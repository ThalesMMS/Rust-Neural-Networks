use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rust_neural_networks::sweep::{generate_configs, load_sweep_config, SweepResult};

// Hyperparameter sweep orchestrator - runs training with multiple configurations
// and aggregates results for comparison.

/// Results from a single training run
#[derive(Debug, Clone)]
struct RunResult {
    config_num: usize,
    log_file: String,
    final_epoch: usize,
    train_loss: f32,
    val_loss: f32,
    val_accuracy: f32,
    training_time: f32,
    _learning_rate: f32,
    // Config parameters for reference
    config_lr: f32,
    config_batch_size: usize,
    config_epochs: usize,
    config_scheduler: String,
    // Store the full config for later serialization
    config: rust_neural_networks::config::TrainingConfig,
}

fn print_usage() {
    eprintln!("Usage: hyperparameter_sweep --target <binary> --sweep <sweep_config.json>");
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  --target <binary>     Target binary to run (e.g., mnist_mlp, mnist_cnn)");
    eprintln!("  --sweep <path>        Path to sweep configuration JSON file");
    eprintln!("  --quick               (Optional) Use reduced epochs for quick testing");
    eprintln!();
    eprintln!("Example:");
    eprintln!("  cargo run --release --bin hyperparameter_sweep -- \\");
    eprintln!("    --target mnist_mlp \\");
    eprintln!("    --sweep config/sweeps/mnist_mlp_sweep.json");
}

fn parse_args() -> Result<(String, String, bool), String> {
    let args: Vec<String> = env::args().collect();

    // Check for --help flag
    if args.len() > 1 && (args[1] == "--help" || args[1] == "-h") {
        print_usage();
        process::exit(0);
    }

    if args.len() < 5 {
        return Err("Insufficient arguments".to_string());
    }

    let mut target_binary = None;
    let mut sweep_config = None;
    let mut quick_mode = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => {
                if i + 1 >= args.len() {
                    return Err("--target requires a value".to_string());
                }
                target_binary = Some(args[i + 1].clone());
                i += 2;
            }
            "--sweep" => {
                if i + 1 >= args.len() {
                    return Err("--sweep requires a value".to_string());
                }
                sweep_config = Some(args[i + 1].clone());
                i += 2;
            }
            "--quick" => {
                quick_mode = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_usage();
                process::exit(0);
            }
            _ => {
                return Err(format!("Unknown argument: {}", args[i]));
            }
        }
    }

    let target = target_binary.ok_or("--target argument is required")?;
    let sweep = sweep_config.ok_or("--sweep argument is required")?;

    Ok((target, sweep, quick_mode))
}

fn main() {
    println!("Hyperparameter Sweep Utility");
    println!("============================\n");

    // Parse command-line arguments
    let (target_binary, sweep_config_path, quick_mode) = match parse_args() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("Error: {}\n", err);
            print_usage();
            process::exit(1);
        }
    };

    // Load sweep configuration
    println!("Loading sweep configuration from: {}", sweep_config_path);
    let sweep_config = match load_sweep_config(&sweep_config_path) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("Error loading sweep config: {}", err);
            process::exit(1);
        }
    };

    // Display sweep parameters
    println!("\nTarget binary: {}", target_binary);
    println!("Quick mode: {}", quick_mode);
    println!("\nSweep configuration loaded successfully!");
    println!("Base config: {}", sweep_config.base_config);

    // Count total configurations to generate
    let mut total_configs = 1;
    if let Some(ref lrs) = sweep_config.learning_rate {
        total_configs *= lrs.len();
        println!("Learning rates to sweep: {:?}", lrs);
    }
    if let Some(ref bss) = sweep_config.batch_size {
        total_configs *= bss.len();
        println!("Batch sizes to sweep: {:?}", bss);
    }
    if let Some(ref epochs) = sweep_config.epochs {
        total_configs *= epochs.len();
        println!("Epochs to sweep: {:?}", epochs);
    }
    if let Some(ref schedulers) = sweep_config.scheduler_type {
        total_configs *= schedulers.len();
        println!("Schedulers to sweep: {:?}", schedulers);
    }

    println!("\nTotal configurations to run: {}", total_configs);

    // Generate all configuration combinations
    println!("\nGenerating configuration combinations...");
    let configs = match generate_configs(&sweep_config) {
        Ok(configs) => configs,
        Err(err) => {
            eprintln!("Error generating configs: {}", err);
            process::exit(1);
        }
    };

    println!("Generated {} configurations", configs.len());

    // Create temp directory for config files
    let temp_dir = create_temp_directory();
    println!("Created temporary directory: {}", temp_dir.display());

    // Track sweep start time
    let sweep_start = Instant::now();
    let mut successful_runs = 0;
    let mut failed_runs = 0;
    let mut run_results: Vec<RunResult> = Vec::new();

    // Run training for each configuration
    println!("\nStarting sweep execution...\n");
    for (idx, config) in configs.iter().enumerate() {
        let config_num = idx + 1;
        println!("==================================================");
        println!("Configuration {}/{}", config_num, configs.len());
        println!("==================================================");

        // Display configuration summary
        print_config_summary(config);

        // Create temporary config file
        let config_filename = format!("sweep_config_{}.json", config_num);
        let config_path = temp_dir.join(&config_filename);

        match save_config_to_file(config, &config_path) {
            Ok(_) => {
                println!("Saved config to: {}", config_path.display());
            }
            Err(err) => {
                eprintln!("Error saving config file: {}", err);
                failed_runs += 1;
                continue;
            }
        }

        // Track log files before training (to find the new one)
        let logs_before = list_log_files("./logs");

        // Run training
        let run_start = Instant::now();
        println!("\nStarting training run...\n");

        let success = run_training(&target_binary, config_path.as_path());

        let run_duration = run_start.elapsed();
        println!("\nRun completed in {:.2}s", run_duration.as_secs_f32());

        if success {
            successful_runs += 1;
            println!("Status: SUCCESS");

            // Find and parse the new log file
            let logs_after = list_log_files("./logs");
            if let Some(new_log_file) = find_new_log_file(&logs_before, &logs_after) {
                println!("Log file created: {}", new_log_file);

                // Parse final metrics from log file
                match parse_log_file(&new_log_file) {
                    Ok((final_epoch, train_loss, val_loss, val_accuracy, training_time, lr)) => {
                        let result = RunResult {
                            config_num,
                            log_file: new_log_file.clone(),
                            final_epoch,
                            train_loss,
                            val_loss,
                            val_accuracy,
                            training_time,
                            _learning_rate: lr,
                            config_lr: config.learning_rate.unwrap_or(0.01),
                            config_batch_size: config.batch_size.unwrap_or(64),
                            config_epochs: config.epochs.unwrap_or(10),
                            config_scheduler: config.scheduler_type.clone(),
                            config: config.clone(),
                        };

                        println!("Final metrics - Epoch: {}, Train Loss: {:.4}, Val Loss: {:.4}, Val Acc: {:.2}%",
                                 final_epoch, train_loss, val_loss, val_accuracy * 100.0);

                        run_results.push(result);
                    }
                    Err(err) => {
                        eprintln!("Warning: Could not parse log file: {}", err);
                    }
                }
            } else {
                eprintln!("Warning: Could not find new log file for this run");
            }
        } else {
            failed_runs += 1;
            println!("Status: FAILED");
        }

        println!();
    }

    // Display final summary
    let sweep_duration = sweep_start.elapsed();
    println!("==================================================");
    println!("Sweep Complete!");
    println!("==================================================");
    println!("Total configurations: {}", configs.len());
    println!("Successful runs: {}", successful_runs);
    println!("Failed runs: {}", failed_runs);
    println!("Total time: {:.2}s", sweep_duration.as_secs_f32());
    println!("\nTemporary configs saved in: {}", temp_dir.display());

    // Display results table
    if !run_results.is_empty() {
        println!("\n==================================================");
        println!("Results Summary");
        println!("==================================================\n");

        print_results_table(&run_results);

        // Find best configuration by validation loss
        if let Some(best) = run_results.iter().min_by(|a, b| {
            a.val_loss
                .partial_cmp(&b.val_loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            println!("\n==================================================");
            println!("Best Configuration (by validation loss)");
            println!("==================================================");
            println!(
                "Config #{}: LR={}, BS={}, Epochs={}, Scheduler={}",
                best.config_num,
                best.config_lr,
                best.config_batch_size,
                best.config_epochs,
                best.config_scheduler
            );
            println!(
                "Val Loss: {:.4}, Val Accuracy: {:.2}%",
                best.val_loss,
                best.val_accuracy * 100.0
            );
            println!("Log file: {}", best.log_file);
        }
    } else {
        println!("\nNote: No results collected (all runs failed)");
    }

    // Write aggregated results to JSON
    if !run_results.is_empty() {
        match write_results_to_json(&run_results) {
            Ok(json_path) => {
                println!("\n==================================================");
                println!("Results exported to: {}", json_path);
                println!("==================================================");
            }
            Err(err) => {
                eprintln!("\nWarning: Could not write results to JSON: {}", err);
            }
        }
    }
}

/// Creates a temporary directory for sweep config files
fn create_temp_directory() -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let temp_dir = PathBuf::from(format!("./logs/sweep_temp_{}", timestamp));

    if let Err(err) = fs::create_dir_all(&temp_dir) {
        eprintln!("Warning: Could not create temp directory: {}", err);
        eprintln!("Falling back to current directory");
        return PathBuf::from(".");
    }

    temp_dir
}

/// Saves a training config to a JSON file
fn save_config_to_file(
    config: &rust_neural_networks::config::TrainingConfig,
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(config)?;
    fs::write(path, json)?;
    Ok(())
}

/// Prints a summary of the configuration parameters
fn print_config_summary(config: &rust_neural_networks::config::TrainingConfig) {
    println!("Parameters:");
    if let Some(lr) = config.learning_rate {
        println!("  Learning rate: {}", lr);
    }
    if let Some(bs) = config.batch_size {
        println!("  Batch size: {}", bs);
    }
    if let Some(ep) = config.epochs {
        println!("  Epochs: {}", ep);
    }
    println!("  Scheduler: {}", config.scheduler_type);
    if let Some(ref act) = config.activation_function {
        println!("  Activation: {}", act);
    }
}

/// Runs a training process for the given binary and config
/// Returns true if successful, false otherwise
fn run_training(target_binary: &str, config_path: &Path) -> bool {
    let mut command = Command::new("cargo");
    command
        .arg("run")
        .arg("--release")
        .arg("--bin")
        .arg(target_binary)
        .arg("--")
        .arg("--config")
        .arg(config_path.to_str().unwrap());

    match command.status() {
        Ok(status) => status.success(),
        Err(err) => {
            eprintln!("Error running training: {}", err);
            false
        }
    }
}

/// Lists all log files in the given directory
fn list_log_files(dir_path: &str) -> Vec<String> {
    let mut log_files = Vec::new();

    if let Ok(entries) = fs::read_dir(dir_path) {
        for entry in entries.flatten() {
            if let Ok(file_name) = entry.file_name().into_string() {
                // Look for training log files (various patterns)
                if file_name.starts_with("training_loss_")
                    || file_name.contains("_train_")
                    || (file_name.starts_with("mnist_") && file_name.ends_with(".csv"))
                    || (file_name.starts_with("cifar") && file_name.ends_with(".csv"))
                {
                    log_files.push(format!("{}/{}", dir_path, file_name));
                }
            }
        }
    }

    log_files
}

/// Finds the new log file by comparing before and after lists
fn find_new_log_file(before: &[String], after: &[String]) -> Option<String> {
    for file in after {
        if !before.contains(file) {
            return Some(file.clone());
        }
    }
    None
}

/// Parses a CSV log file and returns the final epoch's metrics
/// Returns: (epoch, train_loss, val_loss, val_accuracy, training_time, learning_rate)
fn parse_log_file(log_path: &str) -> Result<(usize, f32, f32, f32, f32, f32), String> {
    let file = fs::File::open(log_path).map_err(|e| format!("Could not open log file: {}", e))?;

    let reader = BufReader::new(file);
    let mut last_line = String::new();

    // Read all lines, keeping track of the last non-header line
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("Error reading line: {}", e))?;

        // Skip header line (first line) and empty lines
        if idx > 0 && !line.trim().is_empty() {
            last_line = line;
        }
    }

    if last_line.is_empty() {
        return Err("No data lines found in log file".to_string());
    }

    // Parse the CSV line: epoch,train_loss,train_time,val_loss,val_accuracy,learning_rate
    let parts: Vec<&str> = last_line.split(',').collect();
    if parts.len() < 6 {
        return Err(format!(
            "Invalid CSV format: expected 6 columns, got {}",
            parts.len()
        ));
    }

    let epoch = parts[0]
        .trim()
        .parse::<usize>()
        .map_err(|_| "Could not parse epoch")?;

    let train_loss = parts[1]
        .trim()
        .parse::<f32>()
        .map_err(|_| "Could not parse train_loss")?;

    let training_time = parts[2]
        .trim()
        .parse::<f32>()
        .map_err(|_| "Could not parse train_time")?;

    let val_loss = parts[3]
        .trim()
        .parse::<f32>()
        .map_err(|_| "Could not parse val_loss")?;

    let val_accuracy = parts[4]
        .trim()
        .parse::<f32>()
        .map_err(|_| "Could not parse val_accuracy")?;

    let learning_rate = parts[5]
        .trim()
        .parse::<f32>()
        .map_err(|_| "Could not parse learning_rate")?;

    Ok((
        epoch,
        train_loss,
        val_loss,
        val_accuracy,
        training_time,
        learning_rate,
    ))
}

/// Prints a formatted table of run results
fn print_results_table(results: &[RunResult]) {
    println!(
        "{:<8} {:<10} {:<12} {:<10} {:<12} {:<12} {:<12} {:<15}",
        "Config", "LR", "Batch Size", "Epochs", "Train Loss", "Val Loss", "Val Acc %", "Scheduler"
    );
    println!("{}", "-".repeat(110));

    for result in results {
        println!(
            "{:<8} {:<10} {:<12} {:<10} {:<12.4} {:<12.4} {:<12.2} {:<15}",
            format!("#{}", result.config_num),
            result.config_lr,
            result.config_batch_size,
            result.final_epoch,
            result.train_loss,
            result.val_loss,
            result.val_accuracy * 100.0,
            result.config_scheduler
        );
    }
}

/// Converts a RunResult to a SweepResult for JSON serialization
fn convert_to_sweep_result(run_result: &RunResult) -> SweepResult {
    SweepResult {
        config_id: run_result.config_num,
        learning_rate: run_result.config_lr,
        batch_size: run_result.config_batch_size,
        epochs_completed: run_result.final_epoch,
        scheduler_type: run_result.config_scheduler.clone(),
        activation_function: run_result.config.activation_function.clone(),
        final_train_loss: run_result.train_loss,
        final_val_loss: run_result.val_loss,
        final_val_accuracy: run_result.val_accuracy,
        total_training_time: run_result.training_time,
        log_file: run_result.log_file.clone(),
        step_size: run_result.config.step_size,
        gamma: run_result.config.gamma,
        decay_rate: run_result.config.decay_rate,
        min_lr: run_result.config.min_lr,
        T_max: run_result.config.T_max,
        validation_split: run_result.config.validation_split,
        early_stopping_patience: run_result.config.early_stopping_patience,
        early_stopping_min_delta: run_result.config.early_stopping_min_delta,
        leaky_relu_alpha: run_result.config.leaky_relu_alpha,
        elu_alpha: run_result.config.elu_alpha,
    }
}

/// Writes aggregated sweep results to a JSON file
/// Returns the path to the created JSON file
fn write_results_to_json(results: &[RunResult]) -> Result<String, Box<dyn std::error::Error>> {
    // Generate timestamp for filename
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

    // Create logs directory if it doesn't exist
    fs::create_dir_all("./logs")?;

    // Generate output filename
    let json_filename = format!("./logs/sweep_results_{}.json", timestamp);

    // Convert all RunResults to SweepResults
    let sweep_results: Vec<SweepResult> = results.iter().map(convert_to_sweep_result).collect();

    // Serialize to JSON with pretty printing
    let json_content = serde_json::to_string_pretty(&sweep_results)?;

    // Write to file
    fs::write(&json_filename, json_content)?;

    Ok(json_filename)
}

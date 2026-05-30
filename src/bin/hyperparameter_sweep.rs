use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rust_neural_networks::experiment_registry::RunRecord;
use rust_neural_networks::sweep::{generate_configs, load_sweep_config, SweepResult};

// Hyperparameter sweep orchestrator - runs training with multiple configurations
// and aggregates results for comparison.

/// Results from a single training run
#[derive(Debug, Clone)]
struct RunResult {
    config_num: usize,
    log_file: String,
    run_id: Option<String>,
    run_json_path: Option<String>,
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
    eprintln!("  --export <json|csv>    (Optional) Export aggregated summary table as JSON (default) or CSV");
    eprintln!();
    eprintln!("Example:");
    eprintln!("  cargo run --release --bin hyperparameter_sweep -- \\");
    eprintln!("    --target mnist_mlp \\");
    eprintln!("    --sweep config/sweeps/mnist_mlp_sweep.json");
}

fn parse_args() -> Result<(String, String, bool, Option<String>), String> {
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
    let mut export_format: Option<String> = None;

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
            "--export" => {
                if i + 1 >= args.len() {
                    return Err("--export requires a value (json or csv)".to_string());
                }
                export_format = Some(args[i + 1].clone());
                i += 2;
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

    Ok((target, sweep, quick_mode, export_format))
}

fn main() {
    println!("Hyperparameter Sweep Utility");
    println!("============================\n");

    // Parse command-line arguments
    let (target_binary, sweep_config_path, quick_mode, export_format) = match parse_args() {
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

        let runs_before = list_run_records("./runs");

        let success = run_training(&target_binary, config_path.as_path());

        let run_duration = run_start.elapsed();
        println!("\nRun completed in {:.2}s", run_duration.as_secs_f32());

        if success {
            successful_runs += 1;
            println!("Status: SUCCESS");

            // Try to detect the run record (preferred), falling back to log parsing.
            let runs_after = list_run_records("./runs");
            let new_run_record = find_new_run_record(&runs_before, &runs_after);

            if let Some((run_id, run_json_path)) = new_run_record {
                println!("Run record created: {}", run_json_path);

                match load_run_record(Path::new(&run_json_path)) {
                    Ok(record) => {
                        let metrics = match record.metrics.clone() {
                            Some(m) => m,
                            None => {
                                eprintln!(
                                    "Warning: run.json missing metrics; falling back to log parsing"
                                );
                                // Find and parse the new log file
                                let logs_after = list_log_files("./logs");
                                if let Some(new_log_file) =
                                    find_new_log_file(&logs_before, &logs_after)
                                {
                                    println!("Log file created: {}", new_log_file);
                                    match parse_log_file(&new_log_file) {
                                        Ok((
                                            final_epoch,
                                            train_loss,
                                            val_loss,
                                            val_accuracy,
                                            training_time,
                                            lr,
                                        )) => {
                                            let result = RunResult {
                                                config_num,
                                                log_file: new_log_file.clone(),
                                                run_id: Some(run_id.clone()),
                                                run_json_path: Some(run_json_path.clone()),
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
                                            eprintln!("Warning: Could not parse log file: {}", err)
                                        }
                                    }
                                } else {
                                    eprintln!("Warning: Could not find new log file for this run");
                                }
                                continue;
                            }
                        };

                        let result = RunResult {
                            config_num,
                            log_file: record
                                .artifacts
                                .as_ref()
                                .and_then(|a| a.training_log_csv.clone())
                                .unwrap_or_else(|| "".to_string()),
                            run_id: Some(run_id),
                            run_json_path: Some(run_json_path),
                            final_epoch: metrics.epochs_completed,
                            train_loss: metrics.final_train_loss.unwrap_or(0.0) as f32,
                            val_loss: metrics.final_val_loss.unwrap_or(0.0) as f32,
                            val_accuracy: metrics.final_val_accuracy.unwrap_or(0.0) as f32,
                            training_time: metrics.total_training_time_seconds.unwrap_or(0.0)
                                as f32,
                            _learning_rate: 0.0,
                            config_lr: config.learning_rate.unwrap_or(0.01),
                            config_batch_size: config.batch_size.unwrap_or(64),
                            config_epochs: config.epochs.unwrap_or(10),
                            config_scheduler: config.scheduler_type.clone(),
                            config: config.clone(),
                        };

                        println!(
                            "Final metrics - Epochs: {}, Train Loss: {:.4}, Val Loss: {:.4}, Val Acc: {:.2}%",
                            result.final_epoch,
                            result.train_loss,
                            result.val_loss,
                            result.val_accuracy * 100.0
                        );

                        run_results.push(result);
                    }
                    Err(err) => {
                        eprintln!(
                            "Warning: Could not load run record ({}): {}; falling back to log parsing",
                            run_json_path, err
                        );
                    }
                }
            }

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
                            run_id: None,
                            run_json_path: None,
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

    // Export aggregated summary table
    if !run_results.is_empty() {
        match export_format.as_deref() {
            Some("csv") => match write_results_to_csv(&run_results) {
                Ok(path) => {
                    println!("\n==================================================");
                    println!("Results exported to: {}", path);
                    println!("==================================================");
                }
                Err(err) => {
                    eprintln!("\nWarning: Could not write results to CSV: {}", err);
                }
            },
            Some("json") | None => match write_results_to_json(&run_results) {
                Ok(path) => {
                    println!("\n==================================================");
                    println!("Results exported to: {}", path);
                    println!("==================================================");
                }
                Err(err) => {
                    eprintln!("\nWarning: Could not write results to JSON: {}", err);
                }
            },
            Some(other) => {
                eprintln!(
                    "\nWarning: Unknown export format '{}'; use 'json' or 'csv'",
                    other
                );
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

/// Lists all run record JSON files in the given registry directory.
fn list_run_records(registry_dir_path: &str) -> Vec<String> {
    let mut records = Vec::new();

    let runs_dir = Path::new(registry_dir_path);
    if let Ok(entries) = fs::read_dir(runs_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let run_json = path.join("run.json");
            if run_json.is_file() {
                records.push(run_json.to_string_lossy().to_string());
            }
        }
    }

    records
}

/// Returns (run_id, run_json_path) for the first new run record created.
fn find_new_run_record(before: &[String], after: &[String]) -> Option<(String, String)> {
    for run_json in after {
        if !before.contains(run_json) {
            let run_id = Path::new(run_json)
                .parent()
                .and_then(|p| p.file_name())
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "".to_string());
            return Some((run_id, run_json.clone()));
        }
    }
    None
}

fn load_run_record(path: &Path) -> Result<RunRecord, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)?;
    let record: RunRecord = serde_json::from_str(&contents)?;
    Ok(record)
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

/// Locate the first entry in `after` that is not present in `before`.
///
/// Returns `Some(String)` with the first new file path from `after`, or `None` if no new file is found.
///
/// # Examples
///
/// ```
/// let before = vec!["./logs/training_loss_1.csv".to_string()];
/// let after = vec![
///     "./logs/training_loss_1.csv".to_string(),
///     "./logs/training_loss_2.csv".to_string(),
/// ];
/// assert_eq!(
///     find_new_log_file(&before, &after),
///     Some("./logs/training_loss_2.csv".to_string())
/// );
/// ```
fn find_new_log_file(before: &[String], after: &[String]) -> Option<String> {
    for file in after {
        if !before.contains(file) {
            return Some(file.clone());
        }
    }
    None
}

/// Parse a CSV training log and extract the final epoch's metrics.
///
/// The function reads the log at `log_path`, skips the header, locates the last non-empty data line,
/// and parses CSV columns into metrics.
///
/// # Returns
///
/// A tuple containing `(epoch, train_loss, val_loss, val_accuracy, training_time, learning_rate)`.
/// If the log uses a five-column format (no learning rate column), `learning_rate` is `0.0`.
///
/// # Examples
///
/// ```
/// use std::fs;
/// let path = "test_log.csv";
/// let csv = "\
/// epoch,train_loss,train_time,val_loss,val_accuracy,learning_rate\n\
/// 0,1.0,0.5,0.9,0.80,0.01\n\
/// 1,0.8,1.0,0.7,0.85,0.01\n";
/// fs::write(path, csv).unwrap();
/// let (epoch, train_loss, val_loss, val_accuracy, training_time, learning_rate) = parse_log_file(path).unwrap();
/// assert_eq!(epoch, 1);
/// assert!((train_loss - 0.8).abs() < 1e-6);
/// assert!((val_loss - 0.7).abs() < 1e-6);
/// assert!((val_accuracy - 0.85).abs() < 1e-6);
/// assert!((training_time - 1.0).abs() < 1e-6);
/// assert!((learning_rate - 0.01).abs() < 1e-6);
/// ```
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

    // Parse the CSV line: epoch,train_loss,train_time,val_loss,val_accuracy
    let parts: Vec<&str> = last_line.split(',').collect();
    if parts.len() < 5 {
        return Err(format!(
            "Invalid CSV format: expected 5 columns, got {}",
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

    let learning_rate = if parts.len() >= 6 {
        parts[5].trim().parse::<f32>().map_err(|_| {
            format!(
                "Could not parse learning_rate from column 6: '{}'",
                parts[5].trim()
            )
        })?
    } else {
        0.0
    };

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

/// Writes aggregated sweep results to a CSV file (summary table)
/// Returns the path to the created CSV file
fn write_results_to_csv(results: &[RunResult]) -> Result<String, Box<dyn std::error::Error>> {
    // Generate timestamp for filename
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

    // Create logs directory if it doesn't exist
    fs::create_dir_all("./logs")?;

    // Generate output filename
    let csv_filename = format!("./logs/sweep_results_{}.csv", timestamp);

    let sweep_results: Vec<SweepResult> = results.iter().map(convert_to_sweep_result).collect();

    let mut out = String::new();
    out.push_str("config_id,learning_rate,batch_size,epochs_completed,scheduler_type,activation_function,final_train_loss,final_val_loss,final_val_accuracy,total_training_time,log_file,step_size,gamma,decay_rate,min_lr,T_max,validation_split,early_stopping_patience,early_stopping_min_delta,leaky_relu_alpha,elu_alpha\n");

    for r in sweep_results {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.config_id,
            r.learning_rate,
            r.batch_size,
            r.epochs_completed,
            csv_escape(&r.scheduler_type),
            csv_escape_opt(r.activation_function.as_deref()),
            r.final_train_loss,
            r.final_val_loss,
            r.final_val_accuracy,
            r.total_training_time,
            csv_escape(&r.log_file),
            opt_to_csv(r.step_size),
            opt_to_csv(r.gamma),
            opt_to_csv(r.decay_rate),
            opt_to_csv(r.min_lr),
            opt_to_csv(r.T_max),
            opt_to_csv(r.validation_split),
            opt_to_csv(r.early_stopping_patience),
            opt_to_csv(r.early_stopping_min_delta),
            opt_to_csv(r.leaky_relu_alpha),
            opt_to_csv(r.elu_alpha)
        ));
    }

    fs::write(&csv_filename, out)?;

    Ok(csv_filename)
}

fn csv_escape(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn csv_escape_opt(s: Option<&str>) -> String {
    match s {
        Some(v) => csv_escape(v),
        None => String::new(),
    }
}

fn opt_to_csv<T: std::fmt::Display>(v: Option<T>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}

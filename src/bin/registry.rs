use rust_neural_networks::experiment_registry::load_run_record;
use std::collections::BTreeMap;
use std::env;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage_and_exit();
    }

    let cmd = args[1].as_str();
    match cmd {
        "list" => list_cmd(&args[2..]),
        "compare" => compare_cmd(&args[2..]),
        "export-sweep" => export_sweep_cmd(&args[2..]),
        _ => print_usage_and_exit(),
    }
}

fn list_cmd(args: &[String]) {
    // Supported syntax (minimal):
    //   cargo run --bin registry -- list [--registry-dir <dir>]
    let mut registry_dir = "runs".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--registry-dir" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing value for --registry-dir");
                    std::process::exit(2);
                }
                registry_dir = args[i].clone();
            }
            s if s.starts_with("--") => {
                eprintln!("Unknown flag: {s}");
                std::process::exit(2);
            }
            other => {
                eprintln!("Unexpected arg: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let entries = match std::fs::read_dir(&registry_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to read registry dir {registry_dir:?}: {e}");
            std::process::exit(1);
        }
    };

    // Stable order.
    let mut run_ids: Vec<String> = entries
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            if !path.is_dir() {
                return None;
            }
            let run_json = path.join("run.json");
            if !run_json.exists() {
                return None;
            }
            path.file_name().map(|s| s.to_string_lossy().to_string())
        })
        .collect();
    run_ids.sort();

    for run_id in run_ids {
        println!("{run_id}");
    }
}

fn compare_cmd(args: &[String]) {
    // Supported syntax (minimal):
    //   cargo run --bin registry -- compare <run_id> <run_id> [--registry-dir <dir>]
    let mut registry_dir = "runs".to_string();
    let mut run_ids: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--registry-dir" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing value for --registry-dir");
                    std::process::exit(2);
                }
                registry_dir = args[i].clone();
            }
            s if s.starts_with("--") => {
                eprintln!("Unknown flag: {s}");
                std::process::exit(2);
            }
            run_id => run_ids.push(run_id.to_string()),
        }
        i += 1;
    }

    if run_ids.len() < 2 {
        eprintln!("compare requires at least two run_ids");
        std::process::exit(2);
    }

    let mut records = Vec::new();
    for run_id in &run_ids {
        let path = Path::new(&registry_dir).join(run_id).join("run.json");
        match load_run_record(&path) {
            Ok(r) => records.push(r),
            Err(e) => {
                eprintln!("Failed to load {path:?}: {e}");
                std::process::exit(1);
            }
        }
    }

    // Metrics table
    println!("METRICS");
    println!("run_id\tmodel_type\tepochs\ttrain_loss\tval_loss\tval_acc\ttotal_time_s");
    for r in &records {
        let m = r.metrics.as_ref();
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r.run_id,
            r.model_type,
            m.map(|m| m.epochs_completed.to_string())
                .unwrap_or_else(|| "".to_string()),
            m.and_then(|m| m.final_train_loss)
                .map(fmt_f64)
                .unwrap_or_default(),
            m.and_then(|m| m.final_val_loss)
                .map(fmt_f64)
                .unwrap_or_default(),
            m.and_then(|m| m.final_val_accuracy)
                .map(fmt_f64)
                .unwrap_or_default(),
            m.and_then(|m| m.total_training_time_seconds)
                .map(fmt_f64)
                .unwrap_or_default(),
        );
    }

    println!();
    println!("CONFIG (raw) diff");
    for r in &records {
        let raw = r.config.raw.as_deref().unwrap_or("");
        println!("----- run_id={} -----", r.run_id);
        println!("{raw}");
    }

    // Quick key overview from parsed config, if present.
    println!();
    println!("CONFIG (parsed keys)");
    println!("run_id\tkey\tvalue");
    for r in &records {
        if let Some(parsed) = &r.config.parsed {
            if let Some(obj) = parsed.as_object() {
                let mut keys: BTreeMap<&String, &serde_json::Value> = BTreeMap::new();
                for (k, v) in obj {
                    keys.insert(k, v);
                }
                for (k, v) in keys {
                    let s = match v {
                        serde_json::Value::String(s) => s.clone(),
                        _ => v.to_string(),
                    };
                    println!("{}\t{}\t{}", r.run_id, k, s);
                }
            }
        }
    }
}

fn fmt_f64(v: f64) -> String {
    format!("{:.6}", v)
}

fn export_sweep_cmd(args: &[String]) {
    // Supported syntax (minimal):
    //   cargo run --bin registry -- export-sweep --registry-dir <dir> [--format json|csv]
    // Prints to stdout.
    let mut registry_dir = "runs".to_string();
    let mut format = "json".to_string();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--registry-dir" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing value for --registry-dir");
                    std::process::exit(2);
                }
                registry_dir = args[i].clone();
            }
            "--format" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing value for --format");
                    std::process::exit(2);
                }
                format = args[i].clone();
            }
            s if s.starts_with("--") => {
                eprintln!("Unknown flag: {s}");
                std::process::exit(2);
            }
            other => {
                eprintln!("Unexpected arg: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let entries = match std::fs::read_dir(&registry_dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to read registry dir {registry_dir:?}: {e}");
            std::process::exit(1);
        }
    };

    let mut results: Vec<serde_json::Value> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let run_json = path.join("run.json");
        if !run_json.exists() {
            continue;
        }
        let record = match load_run_record(&run_json) {
            Ok(r) => r,
            Err(_) => continue,
        };

        // Map to the schema expected by compare_sweep_results.py.
        // We keep required keys present even if unknown.
        let (
            epochs_completed,
            final_train_loss,
            final_val_loss,
            final_val_accuracy,
            total_training_time,
        ) = record
            .metrics
            .as_ref()
            .map(|m| {
                (
                    serde_json::Value::from(m.epochs_completed),
                    serde_json::Value::from(m.final_train_loss.unwrap_or(0.0)),
                    serde_json::Value::from(m.final_val_loss.unwrap_or(0.0)),
                    serde_json::Value::from(m.final_val_accuracy.unwrap_or(0.0)),
                    serde_json::Value::from(m.total_training_time_seconds.unwrap_or(0.0)),
                )
            })
            .unwrap_or_else(|| {
                (
                    serde_json::Value::from(0u64),
                    serde_json::Value::from(0.0),
                    serde_json::Value::from(0.0),
                    serde_json::Value::from(0.0),
                    serde_json::Value::from(0.0),
                )
            });

        let parsed = record.config.parsed.as_ref();
        let config_id = parsed
            .and_then(|v| v.get("config_id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| record.run_id.clone());

        let learning_rate = parsed
            .and_then(|v| v.get("learning_rate"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let batch_size = parsed
            .and_then(|v| v.get("batch_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let scheduler_type = parsed
            .and_then(|v| v.get("scheduler_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        results.push(serde_json::json!({
            "config_id": config_id,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs_completed": epochs_completed,
            "scheduler_type": scheduler_type,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "final_val_accuracy": final_val_accuracy,
            "total_training_time": total_training_time,
            // extra registry linkage
            "run_id": record.run_id,
            "run_json_path": run_json,
            "model_type": record.model_type,
        }));
    }

    if format == "json" {
        println!(
            "{}",
            serde_json::to_string_pretty(&results).unwrap_or_else(|_| "[]".to_string())
        );
        return;
    }

    if format == "csv" {
        // Minimal CSV to stdout.
        println!("config_id,learning_rate,batch_size,epochs_completed,scheduler_type,final_train_loss,final_val_loss,final_val_accuracy,total_training_time,run_id,run_json_path,model_type");
        for r in &results {
            let get_s = |k: &str| -> String {
                match r.get(k) {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(v) => v.to_string(),
                    None => "".to_string(),
                }
            };
            println!(
                "{},{},{},{},{},{},{},{},{},{},{},{}",
                csv_escape(&get_s("config_id")),
                get_s("learning_rate"),
                get_s("batch_size"),
                get_s("epochs_completed"),
                csv_escape(&get_s("scheduler_type")),
                get_s("final_train_loss"),
                get_s("final_val_loss"),
                get_s("final_val_accuracy"),
                get_s("total_training_time"),
                csv_escape(&get_s("run_id")),
                csv_escape(&get_s("run_json_path")),
                csv_escape(&get_s("model_type")),
            );
        }
        return;
    }

    eprintln!("Unknown format: {format} (expected json|csv)");
    std::process::exit(2);
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn print_usage_and_exit() -> ! {
    eprintln!(
        "Usage:\n  registry list [--registry-dir <dir>]\n  registry compare <run_id> <run_id> [more run_ids...] [--registry-dir <dir>]\n  registry export-sweep [--registry-dir <dir>] [--format json|csv]  (prints to stdout)\n\nExamples:\n  cargo run --bin registry -- list\n  cargo run --bin registry -- compare 2026-01-01T00-00-00Z_abcd 2026-01-02T00-00-00Z_efgh\n  cargo run --bin registry -- export-sweep --registry-dir runs --format json > logs/sweep_results_from_registry.json"
    );
    std::process::exit(2);
}

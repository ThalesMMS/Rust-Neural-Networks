#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod configs;
mod inference;
mod logs_watch;
mod models;
mod paths;
mod process;
mod runs;
mod status;

use process::ProcessMap;

fn main() {
    tauri::Builder::default()
        .manage(ProcessMap::default())
        .invoke_handler(tauri::generate_handler![
            models::list_models,
            configs::list_configs,
            configs::read_config,
            configs::write_config,
            configs::save_config_as,
            process::start_run,
            process::stop_run,
            logs_watch::find_active_log,
            runs::list_runs,
            runs::list_all_logs,
            runs::read_log_csv,
            runs::list_experiments,
            runs::read_training_series,
            runs::read_gradient_csv,
            runs::read_artifact_preview,
            status::data_status,
            status::checkpoint_status,
            inference::mlp::predict_mlp,
            inference::cnn::predict_cnn,
            inference::attention::predict_attention,
            inference::cifar10::predict_cifar10,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

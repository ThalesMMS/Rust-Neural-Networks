import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export type ArgStyle = "config_flag" | "positional";
export type DataRequirement = "mnist" | "cifar10" | "none";

export interface ModelDescriptor {
  id: string;
  display_name: string;
  description: string;
  category: string;
  bin_name: string;
  default_config_path: string | null;
  default_arch_path: string | null;
  arg_style: ArgStyle;
  supports_step: boolean;
  supports_registry_flags: boolean;
  data_requirement: DataRequirement;
  checkpoints: string[];
  caveats: string[];
}

export interface ConfigEntry {
  relative_path: string;
  group: string;
  file_name: string;
}

export interface StartRunArgs {
  model_id: string;
  config_path?: string | null;
  arch_path?: string | null;
  step: boolean;
  run_name?: string | null;
}

export interface StartedRun {
  run_token: string;
  started_unix_secs: number;
  argv: string[];
}

export interface LogLineEvent {
  run_token: string;
  stream: "stdout" | "stderr" | "system";
  line: string;
}

export interface RunFinishedEvent {
  run_token: string;
  success: boolean;
  message: string;
}

export interface RunRecord {
  schema_version: string;
  run_id: string;
  run_name?: string;
  timestamp_start: string;
  timestamp_end?: string;
  model_type: string;
  command?: string;
  status: "completed" | "failed";
  seed: number;
  config: {
    config_path?: string;
    config_format?: string;
    raw?: string;
    parsed?: unknown;
  };
  dataset?: {
    name: string;
    train_size?: number;
    val_size?: number;
    test_size?: number;
  };
  metrics?: {
    epochs_completed: number;
    final_train_loss?: number;
    final_val_loss?: number;
    final_val_accuracy?: number;
    total_training_time_seconds?: number;
  };
  artifacts?: {
    training_log_csv?: string;
    checkpoints: string[];
    plots: string[];
  };
  environment?: {
    git?: { commit?: string; dirty?: boolean };
    crate_version?: string;
    rustc_version?: string;
    os?: string;
  };
}

export interface LogSummary {
  file_name: string;
  relative_path: string;
  row_count: number;
  last_epoch?: number;
  last_train_loss?: number;
  last_val_loss?: number;
  last_val_accuracy?: number;
  modified_unix_secs: number;
}

export interface CsvRow {
  epoch: number;
  train_loss: number;
  train_time: number;
  val_loss: number;
  val_accuracy: number;
}

export interface DataStatus {
  mnist_present: boolean;
  cifar10_present: boolean;
}

export interface Prediction {
  probabilities: number[];
  predicted_class: number;
}

export const api = {
  listModels: () => invoke<ModelDescriptor[]>("list_models"),

  listConfigs: () => invoke<ConfigEntry[]>("list_configs"),
  readConfig: (relative_path: string) => invoke<string>("read_config", { relativePath: relative_path }),
  writeConfig: (relative_path: string, contents: string) =>
    invoke<void>("write_config", { relativePath: relative_path, contents }),
  saveConfigAs: (relative_path: string, contents: string) =>
    invoke<void>("save_config_as", { relativePath: relative_path, contents }),

  startRun: (args: StartRunArgs) => invoke<StartedRun>("start_run", { args }),
  stopRun: (run_token: string) => invoke<void>("stop_run", { runToken: run_token }),

  findActiveLog: (since_unix_secs: number) =>
    invoke<string | null>("find_active_log", { sinceUnixSecs: since_unix_secs }),

  listRuns: () => invoke<RunRecord[]>("list_runs"),
  listAllLogs: () => invoke<LogSummary[]>("list_all_logs"),
  readLogCsv: (relative_path: string) => invoke<CsvRow[]>("read_log_csv", { relativePath: relative_path }),

  dataStatus: () => invoke<DataStatus>("data_status"),
  checkpointStatus: (relative_paths: string[]) =>
    invoke<boolean[]>("checkpoint_status", { relativePaths: relative_paths }),

  predictMlp: (checkpoint_path: string, pixels: number[]) =>
    invoke<Prediction>("predict_mlp", { checkpointPath: checkpoint_path, pixels }),
  predictCnn: (checkpoint_path: string, pixels: number[]) =>
    invoke<Prediction>("predict_cnn", { checkpointPath: checkpoint_path, pixels }),
  predictAttention: (checkpoint_path: string, pixels: number[]) =>
    invoke<Prediction>("predict_attention", { checkpointPath: checkpoint_path, pixels }),
  predictCifar10: (checkpoint_path: string, pixels: number[]) =>
    invoke<Prediction>("predict_cifar10", { checkpointPath: checkpoint_path, pixels }),
};

export function onTrainingLog(cb: (payload: LogLineEvent) => void): Promise<UnlistenFn> {
  return listen<LogLineEvent>("training-log", (e) => cb(e.payload));
}

export function onTrainingFinished(cb: (payload: RunFinishedEvent) => void): Promise<UnlistenFn> {
  return listen<RunFinishedEvent>("training-finished", (e) => cb(e.payload));
}

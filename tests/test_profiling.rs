//! Integration tests for the profiling module.
//!
//! Tests the public API of [`rust_neural_networks::profiling`]:
//!
//! - [`LayerProfile`] helper methods (`total_ms`, `total_flops`, `param_memory_kib`)
//! - [`EpochProfile`] aggregation (`total_forward_ms`, `total_flops`, etc.)
//! - [`Profiler`] accumulation across multiple epochs
//! - [`Profiler::to_csv_string`] format correctness
//! - [`Profiler::summary`] average timing correctness
//! - Serde round-trips for all public structs
//! - File export methods (`save_csv`, `generate_html_report`)

#[cfg(target_os = "macos")]
extern crate blas_src;
#[cfg(any(target_os = "linux", target_os = "windows"))]
extern crate openblas_src;

use rust_neural_networks::layers::{Conv2DLayer, DenseLayer, Layer};
use rust_neural_networks::profiling::{EpochProfile, LayerProfile, Profiler};
use rust_neural_networks::utils::rng::SimpleRng;

// ============================================================================
// Test helpers
// ============================================================================

/// Build a [`LayerProfile`] with explicit values for assertions.
fn make_layer(name: &str, fwd: f64, bwd: f64, upd: f64, flops: u64, mem: usize) -> LayerProfile {
    LayerProfile {
        layer_name: name.to_string(),
        forward_ms: fwd,
        backward_ms: bwd,
        update_ms: upd,
        flops_forward: flops,
        flops_backward: flops,
        param_memory_bytes: mem,
    }
}

// ============================================================================
// LayerProfile tests
// ============================================================================

mod layer_profile_tests {
    use super::*;

    #[test]
    fn test_total_ms_sums_all_phases() {
        let lp = make_layer("Dense", 1.0, 2.5, 0.5, 0, 0);
        assert!((lp.total_ms() - 4.0).abs() < 1e-9, "total_ms should be 4.0");
    }

    #[test]
    fn test_total_ms_with_zeros() {
        let lp = make_layer("Dense", 0.0, 0.0, 0.0, 0, 0);
        assert!(
            (lp.total_ms()).abs() < 1e-9,
            "total_ms of all-zero should be 0"
        );
    }

    #[test]
    fn test_total_flops_sums_forward_and_backward() {
        let lp = make_layer("Dense", 0.0, 0.0, 0.0, 500_000, 0);
        assert_eq!(lp.total_flops(), 1_000_000, "total_flops = fwd + bwd");
    }

    #[test]
    fn test_total_flops_zero() {
        let lp = make_layer("Dense", 0.0, 0.0, 0.0, 0, 0);
        assert_eq!(lp.total_flops(), 0);
    }

    #[test]
    fn test_param_memory_kib_conversion() {
        // 1024 bytes = 1 KiB
        let lp = make_layer("Dense", 0.0, 0.0, 0.0, 0, 1024);
        assert!((lp.param_memory_kib() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_param_memory_kib_large() {
        // 1 608 704 bytes ≈ 1571 KiB (typical for 784→512 DenseLayer)
        let lp = make_layer("Dense", 0.0, 0.0, 0.0, 0, 1_608_704);
        let expected_kib = 1_608_704.0 / 1024.0;
        assert!((lp.param_memory_kib() - expected_kib).abs() < 1e-3);
    }

    #[test]
    fn test_layer_profile_serde_roundtrip() {
        let original = make_layer("Dense 784→512", 1.5, 2.3, 0.4, 100_000_000, 1_608_704);
        let json = serde_json::to_string(&original).expect("serialize LayerProfile");
        let restored: LayerProfile = serde_json::from_str(&json).expect("deserialize LayerProfile");
        assert_eq!(restored.layer_name, original.layer_name);
        assert!((restored.forward_ms - original.forward_ms).abs() < 1e-9);
        assert!((restored.backward_ms - original.backward_ms).abs() < 1e-9);
        assert_eq!(restored.flops_forward, original.flops_forward);
        assert_eq!(restored.param_memory_bytes, original.param_memory_bytes);
    }
}

// ============================================================================
// EpochProfile tests
// ============================================================================

mod epoch_profile_tests {
    use super::*;

    fn make_epoch(epoch: usize, layers: Vec<LayerProfile>) -> EpochProfile {
        EpochProfile { epoch, layers }
    }

    #[test]
    fn test_total_forward_ms_sums_all_layers() {
        let ep = make_epoch(
            1,
            vec![
                make_layer("A", 1.0, 0.0, 0.0, 0, 0),
                make_layer("B", 3.5, 0.0, 0.0, 0, 0),
            ],
        );
        assert!((ep.total_forward_ms() - 4.5).abs() < 1e-9);
    }

    #[test]
    fn test_total_backward_ms_sums_all_layers() {
        let ep = make_epoch(
            1,
            vec![
                make_layer("A", 0.0, 2.0, 0.0, 0, 0),
                make_layer("B", 0.0, 3.0, 0.0, 0, 0),
            ],
        );
        assert!((ep.total_backward_ms() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_total_update_ms_sums_all_layers() {
        let ep = make_epoch(
            1,
            vec![
                make_layer("A", 0.0, 0.0, 0.5, 0, 0),
                make_layer("B", 0.0, 0.0, 1.5, 0, 0),
            ],
        );
        assert!((ep.total_update_ms() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_total_ms_sums_all_phases_and_layers() {
        let ep = make_epoch(
            1,
            vec![
                make_layer("A", 1.0, 2.0, 0.5, 0, 0),
                make_layer("B", 3.0, 4.0, 1.0, 0, 0),
            ],
        );
        // 1+2+0.5 + 3+4+1 = 11.5
        assert!((ep.total_ms() - 11.5).abs() < 1e-9);
    }

    #[test]
    fn test_total_flops_sums_forward_and_backward_for_all_layers() {
        let ep = make_epoch(
            1,
            vec![
                make_layer("A", 0.0, 0.0, 0.0, 1_000_000, 0),
                make_layer("B", 0.0, 0.0, 0.0, 500_000, 0),
            ],
        );
        // A: 2×1_000_000 = 2_000_000, B: 2×500_000 = 1_000_000 → 3_000_000
        assert_eq!(ep.total_flops(), 3_000_000);
    }

    #[test]
    fn test_total_param_memory_bytes_sums_all_layers() {
        let ep = make_epoch(
            1,
            vec![
                make_layer("A", 0.0, 0.0, 0.0, 0, 4096),
                make_layer("B", 0.0, 0.0, 0.0, 0, 2048),
            ],
        );
        assert_eq!(ep.total_param_memory_bytes(), 6144);
    }

    #[test]
    fn test_epoch_profile_empty_layers() {
        let ep = make_epoch(1, vec![]);
        assert!((ep.total_forward_ms()).abs() < 1e-9);
        assert_eq!(ep.total_flops(), 0);
        assert_eq!(ep.total_param_memory_bytes(), 0);
    }

    #[test]
    fn test_epoch_profile_serde_roundtrip() {
        let ep = make_epoch(3, vec![make_layer("Dense", 1.0, 2.0, 0.5, 100_000, 4096)]);
        let json = serde_json::to_string(&ep).expect("serialize EpochProfile");
        let restored: EpochProfile = serde_json::from_str(&json).expect("deserialize EpochProfile");
        assert_eq!(restored.epoch, 3);
        assert_eq!(restored.layers.len(), 1);
        assert!((restored.layers[0].forward_ms - 1.0).abs() < 1e-9);
    }
}

// ============================================================================
// Profiler accumulation tests
// ============================================================================

mod profiler_accumulation_tests {
    use super::*;

    #[test]
    fn test_new_profiler_is_empty() {
        let profiler = Profiler::new();
        assert_eq!(profiler.epoch_count(), 0);
        assert!(profiler.epochs().is_empty());
    }

    #[test]
    fn test_single_epoch_single_layer_recorded() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(make_layer("Dense", 1.0, 2.0, 0.5, 1_000_000, 4096));
        profiler.end_epoch();

        assert_eq!(profiler.epoch_count(), 1);
        let ep = &profiler.epochs()[0];
        assert_eq!(ep.epoch, 1);
        assert_eq!(ep.layers.len(), 1);
        assert_eq!(ep.layers[0].layer_name, "Dense");
        assert!((ep.layers[0].forward_ms - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_multiple_epochs_accumulate_in_order() {
        let mut profiler = Profiler::new();
        for epoch in 1..=5 {
            profiler.start_epoch(epoch);
            profiler.record_layer(make_layer("Dense", epoch as f64, 0.0, 0.0, 0, 0));
            profiler.end_epoch();
        }

        assert_eq!(profiler.epoch_count(), 5);
        for (i, ep) in profiler.epochs().iter().enumerate() {
            assert_eq!(ep.epoch, i + 1);
            assert!((ep.layers[0].forward_ms - (i + 1) as f64).abs() < 1e-9);
        }
    }

    #[test]
    fn test_multiple_layers_per_epoch() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(make_layer("Hidden", 1.5, 2.3, 0.4, 100_000, 4096));
        profiler.record_layer(make_layer("Output", 0.5, 0.7, 0.1, 10_000, 256));
        profiler.end_epoch();

        let ep = &profiler.epochs()[0];
        assert_eq!(ep.layers.len(), 2);
        assert_eq!(ep.layers[0].layer_name, "Hidden");
        assert_eq!(ep.layers[1].layer_name, "Output");
    }

    #[test]
    fn test_start_epoch_clears_previous_uncommitted_layers() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(make_layer("Orphan", 9.9, 9.9, 9.9, 0, 0));
        // Re-start without calling end_epoch – epoch 1 is discarded.
        profiler.start_epoch(2);
        profiler.record_layer(make_layer("Dense", 1.0, 2.0, 0.5, 0, 0));
        profiler.end_epoch();

        assert_eq!(profiler.epoch_count(), 1);
        assert_eq!(profiler.epochs()[0].epoch, 2);
        assert_eq!(profiler.epochs()[0].layers[0].layer_name, "Dense");
    }

    #[test]
    fn test_epoch_count_matches_end_epoch_calls() {
        let mut profiler = Profiler::new();
        for i in 1..=7 {
            profiler.start_epoch(i);
            profiler.end_epoch();
        }
        assert_eq!(profiler.epoch_count(), 7);
    }
}

// ============================================================================
// CSV format tests
// ============================================================================

mod csv_format_tests {
    use super::*;

    #[test]
    fn test_to_csv_string_has_header() {
        let profiler = Profiler::new();
        let csv = profiler.to_csv_string();
        assert!(
            csv.starts_with("epoch,layer_name,forward_ms,backward_ms,update_ms,flops_forward,flops_backward,param_memory_bytes"),
            "CSV must start with correct header"
        );
    }

    #[test]
    fn test_to_csv_string_empty_profiler_contains_only_header() {
        let profiler = Profiler::new();
        let csv = profiler.to_csv_string();
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines.len(),
            1,
            "empty profiler CSV should have only the header line"
        );
    }

    #[test]
    fn test_to_csv_string_data_row_format() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(LayerProfile {
            layer_name: "Dense".to_string(),
            forward_ms: 1.5,
            backward_ms: 2.3,
            update_ms: 0.4,
            flops_forward: 500_000,
            flops_backward: 500_000,
            param_memory_bytes: 4096,
        });
        profiler.end_epoch();

        let csv = profiler.to_csv_string();
        assert!(
            csv.contains("1,Dense,1.5,2.3,0.4,500000,500000,4096"),
            "data row must contain all fields in CSV order"
        );
    }

    #[test]
    fn test_to_csv_string_multiple_epochs_multiple_rows() {
        let mut profiler = Profiler::new();
        for epoch in 1..=3 {
            profiler.start_epoch(epoch);
            profiler.record_layer(make_layer("L", 1.0, 2.0, 0.5, 10_000, 1024));
            profiler.end_epoch();
        }

        let csv = profiler.to_csv_string();
        let data_rows = csv.lines().skip(1).count();
        assert_eq!(data_rows, 3, "should have one data row per layer×epoch");
    }

    #[test]
    fn test_to_csv_string_multiple_layers_per_epoch() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(make_layer("Hidden", 1.0, 2.0, 0.5, 10_000, 1024));
        profiler.record_layer(make_layer("Output", 0.5, 1.0, 0.2, 5_000, 512));
        profiler.end_epoch();

        let csv = profiler.to_csv_string();
        let data_rows = csv.lines().skip(1).count();
        assert_eq!(data_rows, 2, "two layers in one epoch → two data rows");
        assert!(csv.contains("Hidden"), "CSV must contain Hidden layer name");
        assert!(csv.contains("Output"), "CSV must contain Output layer name");
    }

    #[test]
    fn test_save_csv_creates_file_with_correct_content() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(make_layer("Dense", 1.5, 2.3, 0.4, 1_000_000, 4096));
        profiler.end_epoch();

        let path = std::env::temp_dir().join("test_profiling_integration_save.csv");
        let path_str = path.to_string_lossy().to_string();
        profiler
            .save_csv(&path_str)
            .expect("save_csv should succeed");

        let contents = std::fs::read_to_string(&path).expect("should read CSV file");
        assert!(
            contents.contains("epoch,layer_name"),
            "header must be present"
        );
        assert!(contents.contains("Dense"), "layer name must be present");
        let _ = std::fs::remove_file(&path);
    }
}

// ============================================================================
// Summary / average timing tests
// ============================================================================

mod summary_tests {
    use super::*;

    #[test]
    fn test_summary_empty_profiler_returns_empty_string() {
        let profiler = Profiler::new();
        assert!(
            profiler.summary().is_empty(),
            "empty profiler summary must be empty string"
        );
    }

    #[test]
    fn test_summary_non_empty_contains_layer_name() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(make_layer("Dense 784→512", 1.5, 2.3, 0.4, 100_000, 4096));
        profiler.end_epoch();

        let s = profiler.summary();
        assert!(!s.is_empty());
        assert!(
            s.contains("Dense 784"),
            "summary must contain the layer name"
        );
    }

    #[test]
    fn test_summary_computes_average_over_epochs() {
        // Record three epochs with forward_ms = 1.0, 3.0, 5.0 → avg = 3.0.
        let mut profiler = Profiler::new();
        for (epoch, fwd) in [(1, 1.0f64), (2, 3.0), (3, 5.0)] {
            profiler.start_epoch(epoch);
            profiler.record_layer(make_layer("Dense", fwd, 0.0, 0.0, 0, 0));
            profiler.end_epoch();
        }

        let s = profiler.summary();
        // The formatted avg fwd column should contain "3.000" (three decimal places).
        assert!(
            s.contains("3.000"),
            "summary should show avg forward_ms of 3.000 across 3 epochs"
        );
    }

    #[test]
    fn test_summary_lists_all_unique_layer_names() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(make_layer("Hidden", 1.0, 2.0, 0.5, 0, 0));
        profiler.record_layer(make_layer("Output", 0.5, 1.0, 0.2, 0, 0));
        profiler.end_epoch();

        let s = profiler.summary();
        assert!(s.contains("Hidden"), "summary must mention Hidden layer");
        assert!(s.contains("Output"), "summary must mention Output layer");
    }

    #[test]
    fn test_summary_single_epoch_avg_equals_value() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(make_layer("Dense", 4.567, 8.901, 0.123, 0, 0));
        profiler.end_epoch();

        let s = profiler.summary();
        // With a single epoch, avg = recorded value.
        assert!(
            s.contains("4.567"),
            "single-epoch avg fwd should equal the recorded value"
        );
    }
}

// ============================================================================
// HTML report tests
// ============================================================================

mod html_report_tests {
    use super::*;

    /// Helper: create a profiler with one epoch and one layer, generate HTML to a temp file,
    /// return the HTML contents and clean up the file.
    fn generate_html_to_tmp(layer_name: &str, file_suffix: &str) -> String {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(LayerProfile {
            layer_name: layer_name.to_string(),
            forward_ms: 1.5,
            backward_ms: 2.3,
            update_ms: 0.4,
            flops_forward: 1_000_000,
            flops_backward: 1_000_000,
            param_memory_bytes: 4096,
        });
        profiler.end_epoch();

        let path = std::env::temp_dir().join(format!("test_html_{}.html", file_suffix));
        let path_str = path.to_string_lossy().to_string();
        profiler
            .generate_html_report(&path_str)
            .expect("generate_html_report should succeed");
        let contents = std::fs::read_to_string(&path).expect("should read HTML file");
        let _ = std::fs::remove_file(&path);
        contents
    }

    #[test]
    fn test_html_contains_doctype_declaration() {
        let html = generate_html_to_tmp("Dense", "doctype");
        assert!(
            html.contains("<!DOCTYPE html>"),
            "must have <!DOCTYPE html>"
        );
    }

    #[test]
    fn test_html_contains_chartjs_cdn_reference() {
        let html = generate_html_to_tmp("Dense", "cdn");
        assert!(
            html.contains("cdn.jsdelivr.net") || html.contains("chart.js"),
            "must reference Chart.js CDN"
        );
    }

    #[test]
    fn test_html_contains_layer_name_in_labels_json_array() {
        // The generated HTML embeds layer names in a JS `const labels = [...]` array.
        let html = generate_html_to_tmp("Dense 784", "labels");
        assert!(html.contains("labels"), "must have 'labels' JS variable");
        assert!(
            html.contains("Dense 784"),
            "layer name must appear in the labels array"
        );
    }

    #[test]
    fn test_html_has_stacked_bar_chart_canvas() {
        let html = generate_html_to_tmp("Dense", "bar");
        assert!(
            html.contains("timingChart"),
            "must include timing chart canvas id"
        );
    }

    #[test]
    fn test_html_has_doughnut_flops_chart_canvas() {
        let html = generate_html_to_tmp("Dense", "doughnut");
        assert!(
            html.contains("flopsChart"),
            "must include FLOPS doughnut chart canvas id"
        );
    }

    #[test]
    fn test_html_memory_table_contains_layer_name() {
        let html = generate_html_to_tmp("Hidden Layer", "mem_table");
        assert!(
            html.contains("Hidden Layer"),
            "memory table must contain the layer name"
        );
    }

    #[test]
    fn test_html_embeds_epoch_count() {
        let mut profiler = Profiler::new();
        for epoch in 1..=3 {
            profiler.start_epoch(epoch);
            profiler.record_layer(LayerProfile {
                layer_name: "Dense".to_string(),
                forward_ms: 1.0,
                backward_ms: 2.0,
                update_ms: 0.5,
                flops_forward: 500_000,
                flops_backward: 500_000,
                param_memory_bytes: 2048,
            });
            profiler.end_epoch();
        }

        let path = std::env::temp_dir().join("test_html_epoch_count.html");
        let path_str = path.to_string_lossy().to_string();
        profiler
            .generate_html_report(&path_str)
            .expect("generate_html_report should succeed");
        let html = std::fs::read_to_string(&path).expect("should read HTML file");
        let _ = std::fs::remove_file(&path);

        // The report includes the number of recorded epochs.
        assert!(html.contains("3"), "HTML should embed the epoch count '3'");
    }

    #[test]
    fn test_html_contains_multiple_layer_names() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(LayerProfile {
            layer_name: "Hidden 784→512".to_string(),
            forward_ms: 1.5,
            backward_ms: 2.3,
            update_ms: 0.4,
            flops_forward: 1_000_000,
            flops_backward: 1_000_000,
            param_memory_bytes: 4096,
        });
        profiler.record_layer(LayerProfile {
            layer_name: "Output 512→10".to_string(),
            forward_ms: 0.3,
            backward_ms: 0.5,
            update_ms: 0.1,
            flops_forward: 10_240,
            flops_backward: 10_240,
            param_memory_bytes: 256,
        });
        profiler.end_epoch();

        let path = std::env::temp_dir().join("test_html_multi_layer.html");
        let path_str = path.to_string_lossy().to_string();
        profiler
            .generate_html_report(&path_str)
            .expect("generate_html_report should succeed");
        let html = std::fs::read_to_string(&path).expect("should read HTML file");
        let _ = std::fs::remove_file(&path);

        assert!(
            html.contains("Hidden 784"),
            "HTML must contain Hidden layer name"
        );
        assert!(
            html.contains("Output 512"),
            "HTML must contain Output layer name"
        );
    }

    #[test]
    fn test_generate_html_report_creates_file() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(make_layer("Dense 784→512", 1.5, 2.3, 0.4, 1_000_000, 4096));
        profiler.end_epoch();

        let path = std::env::temp_dir().join("test_profiling_report.html");
        let path_str = path.to_string_lossy().to_string();
        profiler
            .generate_html_report(&path_str)
            .expect("generate_html_report should succeed");

        assert!(
            path.exists(),
            "HTML file must exist after generate_html_report"
        );
        let contents = std::fs::read_to_string(&path).expect("should read HTML file");
        assert!(
            contents.contains("<!DOCTYPE html>"),
            "must be a valid HTML document"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_html_report_contains_chart_js() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(make_layer("Dense", 1.0, 2.0, 0.5, 500_000, 2048));
        profiler.end_epoch();

        let path = std::env::temp_dir().join("test_profiling_chartjs.html");
        let path_str = path.to_string_lossy().to_string();
        profiler
            .generate_html_report(&path_str)
            .expect("generate_html_report should succeed");

        let contents = std::fs::read_to_string(&path).expect("should read HTML file");
        assert!(
            contents.contains("chart.js"),
            "report must reference Chart.js"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_html_report_embeds_layer_name() {
        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(make_layer("Dense 784→512", 1.5, 2.3, 0.4, 0, 0));
        profiler.end_epoch();

        let path = std::env::temp_dir().join("test_profiling_layer_embed.html");
        let path_str = path.to_string_lossy().to_string();
        profiler
            .generate_html_report(&path_str)
            .expect("generate_html_report should succeed");

        let contents = std::fs::read_to_string(&path).expect("should read HTML file");
        assert!(
            contents.contains("Dense 784"),
            "HTML report must embed the layer name"
        );
        let _ = std::fs::remove_file(&path);
    }
}

// ============================================================================
// FLOPS and memory tests (DenseLayer and Conv2DLayer)
// ============================================================================

mod flops_and_memory_tests {
    use super::*;

    // ---- DenseLayer FLOPS ----

    #[test]
    fn test_dense_layer_flops_forward_formula() {
        // Formula: 2 * batch_size * input_size * output_size
        let mut rng = SimpleRng::new(42);
        let layer = DenseLayer::new(784, 512, &mut rng);
        let batch_size = 64;
        let expected = 2 * batch_size as u64 * 784 * 512;
        assert_eq!(
            layer.flops_forward(batch_size),
            expected,
            "DenseLayer forward FLOPS = 2 * batch * in * out"
        );
    }

    #[test]
    fn test_dense_layer_flops_backward_equals_forward() {
        // Backward FLOPS match forward FLOPS for DenseLayer.
        let mut rng = SimpleRng::new(1);
        let layer = DenseLayer::new(512, 10, &mut rng);
        let batch_size = 32;
        assert_eq!(
            layer.flops_backward(batch_size),
            layer.flops_forward(batch_size),
            "DenseLayer backward FLOPS should equal forward FLOPS"
        );
    }

    #[test]
    fn test_dense_layer_flops_scale_linearly_with_batch_size() {
        let mut rng = SimpleRng::new(2);
        let layer = DenseLayer::new(256, 128, &mut rng);
        let flops_1 = layer.flops_forward(1);
        let flops_4 = layer.flops_forward(4);
        assert_eq!(
            flops_4,
            4 * flops_1,
            "FLOPS must scale linearly with batch_size"
        );
    }

    #[test]
    fn test_dense_layer_flops_small_network() {
        // 2-input, 4-hidden layer: forward FLOPS = 2 * batch * 2 * 4 = 16 * batch
        let mut rng = SimpleRng::new(3);
        let layer = DenseLayer::new(2, 4, &mut rng);
        let batch_size = 1;
        assert_eq!(layer.flops_forward(batch_size), 2 * 1 * 2 * 4);
    }

    // ---- DenseLayer parameter memory ----

    #[test]
    fn test_dense_layer_parameter_memory_bytes() {
        // DenseLayer has (input * output) weights + output biases, each f32 (4 bytes).
        let mut rng = SimpleRng::new(4);
        let layer = DenseLayer::new(784, 512, &mut rng);
        let expected_params = 784 * 512 + 512; // weights + biases
        let expected_bytes = expected_params * 4;
        assert_eq!(
            layer.parameter_memory_bytes(),
            expected_bytes,
            "DenseLayer memory = parameter_count * 4"
        );
    }

    #[test]
    fn test_dense_layer_parameter_memory_equals_count_times_four() {
        let mut rng = SimpleRng::new(5);
        let layer = DenseLayer::new(512, 10, &mut rng);
        assert_eq!(
            layer.parameter_memory_bytes(),
            layer.parameter_count() * 4,
            "parameter_memory_bytes must equal parameter_count * sizeof(f32)"
        );
    }

    #[test]
    fn test_dense_layer_parameter_count_formula() {
        let mut rng = SimpleRng::new(6);
        let layer = DenseLayer::new(784, 512, &mut rng);
        // Weights: 784 * 512, Biases: 512
        assert_eq!(
            layer.parameter_count(),
            784 * 512 + 512,
            "DenseLayer parameter_count = input*output + output"
        );
    }

    // ---- Conv2DLayer FLOPS ----

    #[test]
    fn test_conv2d_layer_flops_forward_formula() {
        // Conv2D: 2 * batch * in_channels * k * k * out_channels * out_h * out_w
        // Layer: in_ch=1, out_ch=8, kernel=3, padding=1, stride=1, H=28, W=28
        // out_h = (28 + 2*1 - 3) / 1 + 1 = 28, out_w = 28
        let mut rng = SimpleRng::new(7);
        let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);
        let batch_size = 4;
        let out_h = layer.output_height();
        let out_w = layer.output_width();
        let expected = 2 * batch_size as u64 * 1 * 3 * 3 * 8 * out_h as u64 * out_w as u64;
        assert_eq!(
            layer.flops_forward(batch_size),
            expected,
            "Conv2DLayer forward FLOPS formula mismatch"
        );
    }

    #[test]
    fn test_conv2d_layer_flops_backward_is_two_times_forward() {
        // Conv2D backward is estimated as 2 × forward FLOPS.
        let mut rng = SimpleRng::new(8);
        let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);
        let batch_size = 4;
        assert_eq!(
            layer.flops_backward(batch_size),
            2 * layer.flops_forward(batch_size),
            "Conv2DLayer backward FLOPS = 2 × forward FLOPS"
        );
    }

    #[test]
    fn test_conv2d_layer_output_spatial_dims_with_padding() {
        // With padding=1, stride=1, kernel=3 on 28×28 input → output is 28×28.
        let mut rng = SimpleRng::new(9);
        let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);
        assert_eq!(layer.output_height(), 28);
        assert_eq!(layer.output_width(), 28);
    }

    #[test]
    fn test_conv2d_layer_output_spatial_dims_no_padding() {
        // With padding=0, stride=1, kernel=3 on 28×28 input → output is 26×26.
        let mut rng = SimpleRng::new(10);
        let layer = Conv2DLayer::new(1, 8, 3, 0, 1, 28, 28, &mut rng);
        assert_eq!(layer.output_height(), 26);
        assert_eq!(layer.output_width(), 26);
    }

    // ---- Conv2DLayer parameter memory ----

    #[test]
    fn test_conv2d_layer_parameter_memory_equals_count_times_four() {
        let mut rng = SimpleRng::new(11);
        let layer = Conv2DLayer::new(1, 8, 3, 1, 1, 28, 28, &mut rng);
        assert_eq!(
            layer.parameter_memory_bytes(),
            layer.parameter_count() * 4,
            "Conv2DLayer parameter_memory_bytes = parameter_count * sizeof(f32)"
        );
    }

    // ---- Layer trait integration with Profiler ----

    #[test]
    fn test_profiler_records_layer_flops_from_dense_layer() {
        let mut rng = SimpleRng::new(12);
        let layer = DenseLayer::new(784, 512, &mut rng);
        let batch_size = 64;

        let mut profiler = Profiler::new();
        profiler.start_epoch(1);
        profiler.record_layer(LayerProfile {
            layer_name: "Dense 784→512".to_string(),
            forward_ms: 1.0,
            backward_ms: 2.0,
            update_ms: 0.5,
            flops_forward: layer.flops_forward(batch_size),
            flops_backward: layer.flops_backward(batch_size),
            param_memory_bytes: layer.parameter_memory_bytes(),
        });
        profiler.end_epoch();

        let ep = &profiler.epochs()[0];
        let expected_fwd_flops = 2 * batch_size as u64 * 784 * 512;
        assert_eq!(ep.layers[0].flops_forward, expected_fwd_flops);
        assert_eq!(ep.layers[0].flops_backward, expected_fwd_flops);
        assert_eq!(ep.layers[0].param_memory_bytes, (784 * 512 + 512) * 4);
    }
}

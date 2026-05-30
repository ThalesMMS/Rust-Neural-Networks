use std::process::{Command, ExitCode};

const EXIT_OK: u8 = 0;
const EXIT_FAIL: u8 = 1;
const EXIT_CLI: u8 = 64;

#[derive(Copy, Clone, Debug)]
enum OutputFormat {
    Text,
    Json,
}

#[allow(dead_code)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Status {
    Pass,
    Warn,
    Fail,
    NotApplicable,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Warn => "WARN",
            Status::Fail => "FAIL",
            Status::NotApplicable => "N/A",
        }
    }

    fn ok_for_exit(self, strict: bool) -> bool {
        match self {
            Status::Pass | Status::NotApplicable => true,
            Status::Warn => !strict,
            Status::Fail => false,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct CheckResult {
    name: &'static str,
    status: Status,
    summary: String,
    details: Option<String>,
}

#[allow(dead_code)]
impl CheckResult {
    fn pass(name: &'static str, summary: impl Into<String>) -> Self {
        Self {
            name,
            status: Status::Pass,
            summary: summary.into(),
            details: None,
        }
    }

    fn warn(name: &'static str, summary: impl Into<String>, details: impl Into<String>) -> Self {
        Self {
            name,
            status: Status::Warn,
            summary: summary.into(),
            details: Some(details.into()),
        }
    }

    fn fail(name: &'static str, summary: impl Into<String>, details: impl Into<String>) -> Self {
        Self {
            name,
            status: Status::Fail,
            summary: summary.into(),
            details: Some(details.into()),
        }
    }

    fn na(name: &'static str, summary: impl Into<String>) -> Self {
        Self {
            name,
            status: Status::NotApplicable,
            summary: summary.into(),
            details: None,
        }
    }
}

fn main() -> ExitCode {
    let mut format = OutputFormat::Text;
    let mut strict = false;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--format" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("preflight: --format requires a value (text|json)");
                    return ExitCode::from(EXIT_CLI);
                };
                format = match v.as_str() {
                    "text" => OutputFormat::Text,
                    "json" => OutputFormat::Json,
                    other => {
                        eprintln!("preflight: unknown format '{other}' (expected text|json)");
                        return ExitCode::from(EXIT_CLI);
                    }
                };
                i += 2;
            }
            "--strict" => {
                strict = true;
                i += 1;
            }
            "--help" | "-h" | "help" => {
                print_usage();
                return ExitCode::from(EXIT_OK);
            }
            other => {
                eprintln!("preflight: unknown arg '{other}'\n");
                print_usage();
                return ExitCode::from(EXIT_CLI);
            }
        }
    }

    let checks = vec![
        check_platform(),
        check_rust_toolchain(),
        check_blas_readiness(),
        check_gpu_metal(),
        check_gpu_cuda(),
        check_python_readiness(),
        check_dataset_readiness(),
        check_wasm_pack_readiness(),
    ];

    match format {
        OutputFormat::Text => print_text_report(&checks, strict),
        OutputFormat::Json => print_json_report(&checks, strict),
    }

    let ok = checks.iter().all(|c| c.status.ok_for_exit(strict));
    ExitCode::from(if ok { EXIT_OK } else { EXIT_FAIL })
}

fn print_usage() {
    eprintln!(
        "Usage:\n  cargo run --bin preflight -- [--format text|json] [--strict]\n\nDefaults:\n  --format text\n\nExit codes:\n  0 = pass (warnings allowed unless --strict)\n  1 = fail\n  64 = CLI usage error\n"
    );
}

fn check_platform() -> CheckResult {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;

    CheckResult::pass("platform", format!("{os}/{arch}"))
}

fn recommendations_for_platform() -> Vec<String> {
    let mut recs = vec!["cargo test".to_string()];

    if cfg!(target_os = "macos") {
        recs.push("cargo test --features gpu-metal".to_string());
    }

    if cfg!(target_os = "linux") || cfg!(target_os = "windows") {
        recs.push("cargo test --features gpu-cuda".to_string());
    }

    recs.push("cd wasm && wasm-pack build --target web".to_string());

    recs
}

fn check_rust_toolchain() -> CheckResult {
    let cargo = run_version("cargo");
    let rustc = run_version("rustc");

    match (cargo, rustc) {
        (Some(cargo), Some(rustc)) => CheckResult::pass("rust", format!("{cargo}; {rustc}")),
        (None, Some(rustc)) => CheckResult::fail(
            "rust",
            "cargo missing",
            format!("Found {rustc}. Install Rust via https://rustup.rs/"),
        ),
        (Some(cargo), None) => CheckResult::fail(
            "rust",
            "rustc missing",
            format!("Found {cargo}. Install Rust via https://rustup.rs/"),
        ),
        (None, None) => CheckResult::fail(
            "rust",
            "cargo/rustc missing",
            "Install Rust via https://rustup.rs/",
        ),
    }
}

fn run_version(bin: &str) -> Option<String> {
    let out = Command::new(bin).arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn command_exists(bin: &str) -> bool {
    Command::new(bin)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn check_dataset_readiness() -> CheckResult {
    use std::path::Path;

    let data_dir = Path::new("./data");

    let mnist_dir = data_dir.join("mnist");
    let mnist_ok = mnist_dir.join("train-images-idx3-ubyte").is_file()
        && mnist_dir.join("train-labels-idx1-ubyte").is_file()
        && mnist_dir.join("t10k-images-idx3-ubyte").is_file()
        && mnist_dir.join("t10k-labels-idx1-ubyte").is_file();

    let cifar_dir = data_dir.join("cifar-10-batches-bin");
    let cifar_required = [
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin",
        "test_batch.bin",
    ];
    let cifar_ok = cifar_required.iter().all(|f| cifar_dir.join(f).is_file());

    if mnist_ok && cifar_ok {
        return CheckResult::pass("datasets", "MNIST + CIFAR-10 present under ./data")
            .with_details(
                "Verified expected files under:
  ./data/mnist/
  ./data/cifar-10-batches-bin/
"
                .to_string(),
            );
    }

    if !data_dir.is_dir() {
        return CheckResult::warn(
            "datasets",
            "./data directory missing",
            "Create/populate ./data (or run the helper):
  cargo run --bin dataset-helper -- verify

If datasets are missing, follow:
  docs/cifar10_dataset.md
  docs/tutorials/02_mnist_mlp.md
",
        );
    }

    let mut missing = Vec::new();
    if !mnist_ok {
        missing.push("MNIST (./data/mnist/*.ubyte)".to_string());
    }
    if !cifar_ok {
        missing.push("CIFAR-10 (./data/cifar-10-batches-bin/*.bin)".to_string());
    }

    CheckResult::warn(
        "datasets",
        format!("Missing datasets: {}", missing.join(", ")),
        "Verify what is present:
  cargo run --bin dataset-helper -- verify

If you need to download/prepare datasets, see:
  docs/cifar10_dataset.md
  docs/tutorials/02_mnist_mlp.md
",
    )
}

fn check_blas_readiness() -> CheckResult {
    if cfg!(target_os = "macos") {
        return CheckResult::pass(
            "blas",
            "macOS Accelerate (via blas-src) should be available by default",
        )
        .with_details(
            "If you hit BLAS link errors on macOS:
  - Ensure Xcode Command Line Tools are installed: xcode-select --install
  - Ensure you're using the Apple Clang toolchain (default)
"
            .to_string(),
        );
    }

    if cfg!(target_os = "linux") {
        return CheckResult::pass(
            "blas",
            "Linux OpenBLAS (via openblas-src) is built as part of the Rust build",
        )
        .with_details(
            "If you hit BLAS link errors on Linux:
  - Ensure build essentials are installed (compiler + make)
  - If your environment forbids building OpenBLAS from source, consider installing system OpenBLAS dev packages and adjusting the build accordingly.
"
            .to_string(),
        );
    }

    if cfg!(target_os = "windows") {
        if command_exists("vcpkg") {
            return CheckResult::pass(
                "blas",
                "Windows OpenBLAS expects a system install; vcpkg detected",
            )
            .with_details(
                "If builds fail to link OpenBLAS:
  - Install OpenBLAS with vcpkg:
      vcpkg install openblas
  - Ensure VCPKG_ROOT is set and you're using the MSVC toolchain
"
                .to_string(),
            );
        }

        return CheckResult::warn(
            "blas",
            "Windows OpenBLAS expects a system install; vcpkg not found",
            "Install vcpkg and OpenBLAS:
  - https://github.com/microsoft/vcpkg
  - vcpkg install openblas
Then ensure VCPKG_ROOT is set so openblas-src can find it.
"
            .to_string(),
        );
    }

    CheckResult::na("blas", "unsupported platform")
}

fn check_gpu_metal() -> CheckResult {
    if !cfg!(target_os = "macos") {
        return CheckResult::na("gpu-metal", "not applicable (non-macOS platform)");
    }

    // Best-effort probes. This does not guarantee a Metal-capable GPU, but catches
    // the most common missing prereq for building the Metal backend.
    if Command::new("xcode-select")
        .arg("-p")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        CheckResult::pass(
            "gpu-metal",
            "Xcode Command Line Tools detected; Metal backend should be buildable",
        )
        .with_details(
            "To enable Metal acceleration:
  cargo test --features gpu-metal

If you hit Metal/framework linker errors:
  - Ensure Xcode Command Line Tools are installed: xcode-select --install
  - Ensure you are building on macOS with the Apple toolchain
"
            .to_string(),
        )
    } else {
        CheckResult::warn(
            "gpu-metal",
            "Xcode Command Line Tools not detected; Metal builds may fail",
            "Install Xcode Command Line Tools:
  xcode-select --install

Then try:
  cargo test --features gpu-metal
"
            .to_string(),
        )
    }
}

fn check_gpu_cuda() -> CheckResult {
    if cfg!(target_os = "macos") {
        return CheckResult::na(
            "gpu-cuda",
            "not applicable (CUDA backend is intended for Linux/Windows)",
        );
    }

    let mut findings = Vec::new();

    let nvcc = Command::new("nvcc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() {
                None
            } else {
                Some(s)
            }
        });

    if let Some(nvcc_v) = &nvcc {
        findings.push(format!("nvcc: {nvcc_v}"));
    }

    let nvidia_smi = Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() {
                None
            } else {
                Some(s)
            }
        });

    if let Some(list) = &nvidia_smi {
        findings.push(format!("nvidia-smi: {list}"));
    }

    if nvcc.is_some() {
        let summary = if findings.is_empty() {
            "CUDA toolkit detected (nvcc found)".to_string()
        } else {
            format!("CUDA present ({})", findings.join("; "))
        };

        return CheckResult::pass("gpu-cuda", summary).with_details(
            "To enable CUDA acceleration:
  cargo test --features gpu-cuda

If you hit build errors:
  - Ensure the CUDA Toolkit is installed and `nvcc` is on PATH
  - Ensure an NVIDIA driver is installed (try `nvidia-smi`)
  - See: https://developer.nvidia.com/cuda-downloads
"
            .to_string(),
        );
    }

    CheckResult::warn(
        "gpu-cuda",
        "CUDA toolkit not detected (nvcc not found)",
        "CUDA is optional.

To install CUDA:
  - Linux/Windows: https://developer.nvidia.com/cuda-downloads

After install, verify:
  - nvcc --version
  - (optional) nvidia-smi

Then enable:
  cargo test --features gpu-cuda
"
        .to_string(),
    )
}

fn check_python_readiness() -> CheckResult {
    let python = Command::new("python3")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            let stdout = String::from_utf8_lossy(&o.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&o.stderr).trim().to_string();
            let v = if !stdout.is_empty() { stdout } else { stderr };
            if v.is_empty() {
                None
            } else {
                Some(v)
            }
        });

    let Some(python_v) = python else {
        return CheckResult::warn(
            "python",
            "python3 not found",
            "Python is only required for optional scripts (visualizations, dashboards).

Install Python 3:
  - https://www.python.org/downloads/

If you have `python` but not `python3`, try running with your system's Python launcher.
"
            .to_string(),
        );
    };

    let pip = Command::new("python3")
        .args(["-m", "pip", "--version"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() {
                None
            } else {
                Some(s)
            }
        });

    let venv = Command::new("python3")
        .args(["-m", "venv", "-h"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    let has_requirements = std::path::Path::new("requirements.txt").exists();

    let mut summary = python_v;
    if pip.is_some() {
        summary.push_str("; pip available");
    } else {
        summary.push_str("; pip missing");
    }
    if venv {
        summary.push_str("; venv available");
    } else {
        summary.push_str("; venv missing");
    }

    if !has_requirements {
        return CheckResult::pass("python", summary);
    }

    if pip.is_some() && venv {
        return CheckResult::pass("python", summary).with_details(
            "Optional: set up Python deps for scripts:
  python3 -m venv .venv
  source .venv/bin/activate    # Windows: .venv\\Scripts\\activate
  python3 -m pip install -r requirements.txt
"
            .to_string(),
        );
    }

    CheckResult::warn(
        "python",
        summary,
        "Python is installed, but pip and/or venv support was not detected.

If you want to use Python helper scripts, ensure pip and venv are available, then run:
  python3 -m venv .venv
  source .venv/bin/activate    # Windows: .venv\\Scripts\\activate
  python3 -m pip install -r requirements.txt
"
        .to_string(),
    )
}

fn check_wasm_pack_readiness() -> CheckResult {
    // These are optional, so default to WARN when missing.
    // Report N/A only on targets where wasm32 builds are not meaningful.

    let wasm_pack_ok = command_exists("wasm-pack");

    let wasm_target_ok = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .map(|s| s.lines().any(|l| l.trim() == "wasm32-unknown-unknown"))
        .unwrap_or(false);

    let node_ok = command_exists("node");
    let npm_ok = command_exists("npm");

    if wasm_pack_ok && wasm_target_ok {
        let mut summary =
            "wasm-pack installed; wasm32-unknown-unknown target installed".to_string();
        if node_ok {
            summary.push_str("; node available");
        } else {
            summary.push_str("; node missing");
        }
        if npm_ok {
            summary.push_str("; npm available");
        } else {
            summary.push_str("; npm missing");
        }

        return CheckResult::pass("wasm", summary).with_details(
            "WASM build commands:
  cd wasm
  wasm-pack build --target web

To run the demo (if applicable):
  # any static server works
  python3 -m http.server 8000
"
            .to_string(),
        );
    }

    let mut details = String::from(
        "WASM is optional.

To install wasm-pack:
  cargo install wasm-pack

To install the wasm target:
  rustup target add wasm32-unknown-unknown
",
    );

    if !node_ok || !npm_ok {
        details.push_str(
            "
Optional (for some demo workflows): install Node.js (includes npm):
  https://nodejs.org/
",
        );
    }

    if !wasm_pack_ok && !wasm_target_ok {
        CheckResult::warn(
            "wasm",
            "wasm-pack not found; wasm32 target not installed",
            details,
        )
    } else if !wasm_pack_ok {
        CheckResult::warn("wasm", "wasm-pack not found", details)
    } else {
        CheckResult::warn(
            "wasm",
            "rustup target wasm32-unknown-unknown not installed",
            details,
        )
    }
}

fn print_text_report(checks: &[CheckResult], strict: bool) {
    println!("Rust Neural Networks preflight\n");

    for c in checks {
        println!(
            "[{status}] {name}: {summary}",
            status = c.status.as_str(),
            name = c.name,
            summary = c.summary
        );
        if let Some(d) = &c.details {
            println!("{d}");
        }
    }

    let ok = checks.iter().all(|c| c.status.ok_for_exit(strict));
    println!(
        "\nResult: {result} (strict={strict})",
        result = if ok { "PASS" } else { "FAIL" }
    );

    println!("\nRecommendations:");
    for cmd in recommendations_for_platform() {
        println!("  {cmd}");
    }
}

fn print_json_report(checks: &[CheckResult], strict: bool) {
    fn esc(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    }

    let ok = checks.iter().all(|c| c.status.ok_for_exit(strict));
    let recs = recommendations_for_platform();

    println!(
        "{{\"strict\":{},\"ok\":{},\"checks\":[",
        if strict { "true" } else { "false" },
        if ok { "true" } else { "false" }
    );
    for (idx, c) in checks.iter().enumerate() {
        if idx != 0 {
            println!(",");
        }
        print!(
            "{{\"name\":\"{}\",\"status\":\"{}\",\"summary\":\"{}\"",
            esc(c.name),
            esc(c.status.as_str()),
            esc(&c.summary)
        );
        if let Some(d) = &c.details {
            print!(",\"details\":\"{}\"", esc(d));
        }
        print!("}}");
    }

    println!("],\"recommendations\":[");
    for (idx, r) in recs.iter().enumerate() {
        if idx != 0 {
            println!(",");
        }
        print!("\"{}\"", esc(r));
    }
    println!("]}}");
}

trait WithDetails {
    fn with_details(self, details: String) -> Self;
}

impl WithDetails for CheckResult {
    fn with_details(mut self, details: String) -> Self {
        self.details = Some(details);
        self
    }
}

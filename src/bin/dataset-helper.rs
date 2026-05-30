use std::path::{Path, PathBuf};
use std::process::ExitCode;

const EXIT_OK: u8 = 0;
const EXIT_MISSING: u8 = 2;
const EXIT_MALFORMED: u8 = 3;
const EXIT_CHECKSUM_FAIL: u8 = 4;
const EXIT_CLI: u8 = 64;

#[derive(Copy, Clone, Debug)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Clone, Debug)]
struct DatasetReport {
    name: &'static str,
    ok: bool,
    code: u8,
    summary: String,
    details: Option<String>,
}

impl DatasetReport {
    fn ok(name: &'static str) -> Self {
        Self {
            name,
            ok: true,
            code: EXIT_OK,
            summary: "OK".to_string(),
            details: None,
        }
    }

    fn err(
        name: &'static str,
        code: u8,
        summary: impl Into<String>,
        details: impl Into<String>,
    ) -> Self {
        Self {
            name,
            ok: false,
            code,
            summary: summary.into(),
            details: Some(details.into()),
        }
    }
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);

    let Some(cmd) = args.next() else {
        print_usage();
        return ExitCode::from(EXIT_CLI);
    };

    match cmd.as_str() {
        "verify" => verify_cmd(args.collect()),
        "--help" | "-h" | "help" => {
            print_usage();
            ExitCode::from(EXIT_OK)
        }
        other => {
            eprintln!("dataset-helper: unknown command '{other}'\n");
            print_usage();
            ExitCode::from(EXIT_CLI)
        }
    }
}

fn print_usage() {
    eprintln!(
        "Usage:\n  dataset-helper verify [--mnist|--cifar10|--all] [--data-dir <dir>] [--format text|json]\n\nDefaults:\n  --all\n  --data-dir ./data\n  --format text\n"
    );
}

fn verify_cmd(args: Vec<String>) -> ExitCode {
    let mut data_dir = PathBuf::from("./data");
    let mut check_mnist = false;
    let mut check_cifar10 = false;
    let mut format = OutputFormat::Text;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("dataset-helper verify: --data-dir requires a value");
                    return ExitCode::from(EXIT_CLI);
                };
                data_dir = PathBuf::from(v);
                i += 2;
            }
            "--format" => {
                let Some(v) = args.get(i + 1) else {
                    eprintln!("dataset-helper verify: --format requires a value (text|json)");
                    return ExitCode::from(EXIT_CLI);
                };
                format = match v.as_str() {
                    "text" => OutputFormat::Text,
                    "json" => OutputFormat::Json,
                    other => {
                        eprintln!(
                            "dataset-helper verify: unknown format '{other}' (expected text|json)"
                        );
                        return ExitCode::from(EXIT_CLI);
                    }
                };
                i += 2;
            }
            "--mnist" => {
                check_mnist = true;
                i += 1;
            }
            "--cifar10" => {
                check_cifar10 = true;
                i += 1;
            }
            "--all" => {
                check_mnist = true;
                check_cifar10 = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_usage();
                return ExitCode::from(EXIT_OK);
            }
            other => {
                eprintln!("dataset-helper verify: unknown flag '{other}'");
                return ExitCode::from(EXIT_CLI);
            }
        }
    }

    if !check_mnist && !check_cifar10 {
        // Default is --all
        check_mnist = true;
        check_cifar10 = true;
    }

    let mut reports: Vec<DatasetReport> = Vec::new();

    if check_mnist {
        match verify_mnist(&data_dir) {
            Ok(()) => reports.push(DatasetReport::ok("mnist")),
            Err(e) => reports.push(DatasetReport::err(
                "mnist",
                EXIT_MISSING,
                "missing or malformed dataset",
                e,
            )),
        }
    }

    if check_cifar10 {
        match verify_cifar10(&data_dir) {
            Ok(()) => reports.push(DatasetReport::ok("cifar10")),
            Err(e) => reports.push(DatasetReport::err(
                "cifar10",
                EXIT_MISSING,
                "missing or malformed dataset",
                e,
            )),
        }
    }

    let exit = reports.iter().map(|r| r.code).max().unwrap_or(EXIT_OK);

    match format {
        OutputFormat::Text => print_text_report(&data_dir, &reports),
        OutputFormat::Json => print_json_report(&data_dir, &reports),
    }

    ExitCode::from(exit)
}

fn print_text_report(data_dir: &Path, reports: &[DatasetReport]) {
    eprintln!("dataset-helper verify");
    eprintln!("data dir: {}", data_dir.display());
    for r in reports {
        if r.ok {
            eprintln!("{}: OK", r.name);
        } else {
            eprintln!("{}: ERROR ({}) - {}", r.name, r.code, r.summary);
            if let Some(d) = &r.details {
                eprintln!("{d}");
            }
        }
        eprintln!();
    }
}

fn print_json_report(data_dir: &Path, reports: &[DatasetReport]) {
    fn esc(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 8);
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out
    }

    let mut s = String::new();
    s.push_str("{");
    s.push_str(&format!(
        "\"data_dir\":\"{}\",",
        esc(&data_dir.display().to_string())
    ));
    s.push_str("\"datasets\":[");
    for (idx, r) in reports.iter().enumerate() {
        if idx > 0 {
            s.push(',');
        }
        s.push_str("{");
        s.push_str(&format!("\"name\":\"{}\"", esc(r.name)));
        s.push_str(&format!(",\"ok\":{}", if r.ok { "true" } else { "false" }));
        s.push_str(&format!(",\"code\":{}", r.code));
        s.push_str(&format!(",\"summary\":\"{}\"", esc(&r.summary)));
        if let Some(d) = &r.details {
            s.push_str(&format!(",\"details\":\"{}\"", esc(d)));
        } else {
            s.push_str(",\"details\":null");
        }
        s.push_str("}");
    }
    s.push_str("]}");
    println!("{s}");
}

fn verify_cifar10(data_dir: &Path) -> Result<(), String> {
    let cifar_dir = data_dir.join("cifar-10-batches-bin");
    let required = [
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin",
        "test_batch.bin",
    ];

    let missing: Vec<String> = required
        .iter()
        .filter_map(|f| {
            let p = cifar_dir.join(f);
            if p.is_file() {
                None
            } else {
                Some(p.display().to_string())
            }
        })
        .collect();

    if missing.is_empty() {
        return Ok(());
    }

    // Common mistake 1: tarball downloaded but not extracted.
    let tarball = data_dir.join("cifar-10-binary.tar.gz");
    if tarball.is_file() {
        return Err(format!(
            "found '{}' but CIFAR-10 batches are missing.
Hint: extract the tarball so that '{}' exists and contains the batch .bin files.
Missing: {}",
            tarball.display(),
            cifar_dir.display(),
            missing.join(", ")
        ));
    }

    // Common mistake 2: extracted into an extra nesting directory.
    // e.g. <data_dir>/cifar-10-binary/cifar-10-batches-bin
    for sub in ["cifar-10-binary", "cifar-10", "CIFAR-10", "cifar10"] {
        let candidate = data_dir.join(sub).join("cifar-10-batches-bin");
        if required.iter().all(|f| candidate.join(f).is_file()) {
            return Err(format!(
                "files found under '{}' but expected under '{}' (relative to --data-dir).
Hint: move 'cifar-10-batches-bin' to '{}'.",
                candidate.display(),
                cifar_dir.display(),
                data_dir.display()
            ));
        }
    }

    // Common mistake 3: python version extracted (cifar-10-batches-py)
    let py_dir = data_dir.join("cifar-10-batches-py");
    if py_dir.is_dir() {
        return Err(format!(
            "found '{}' (python version), but this repo expects binary batches under '{}'.
Hint: download 'CIFAR-10 binary version' (cifar-10-binary.tar.gz) and extract it.",
            py_dir.display(),
            cifar_dir.display()
        ));
    }

    // Fallback: list what is missing.
    let mut msg = String::new();
    msg.push_str("missing required CIFAR-10 batch files:\n");
    for m in &missing {
        msg.push_str(&format!("  - {m}\n"));
    }
    msg.push_str("\nExpected layout:\n");
    msg.push_str(&format!("  {}/data_batch_1.bin\n", cifar_dir.display()));
    msg.push_str(&format!("  {}/data_batch_2.bin\n", cifar_dir.display()));
    msg.push_str(&format!("  {}/data_batch_3.bin\n", cifar_dir.display()));
    msg.push_str(&format!("  {}/data_batch_4.bin\n", cifar_dir.display()));
    msg.push_str(&format!("  {}/data_batch_5.bin\n", cifar_dir.display()));
    msg.push_str(&format!("  {}/test_batch.bin\n", cifar_dir.display()));
    msg.push_str("\nOfficial source: https://www.cs.toronto.edu/~kriz/cifar.html\n");
    msg.push_str("Direct download: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz\n");
    msg.push_str("License/terms:   CIFAR-10 is provided by Alex Krizhevsky et al.; review the terms on the official site before redistribution/use.\n");
    Err(msg)
}

fn verify_mnist(data_dir: &Path) -> Result<(), String> {
    let required = [
        "train-images.idx3-ubyte",
        "train-labels.idx1-ubyte",
        "t10k-images.idx3-ubyte",
        "t10k-labels.idx1-ubyte",
    ];

    // Canonical location: <data_dir>/<file>
    let missing: Vec<String> = required
        .iter()
        .filter_map(|f| {
            let p = data_dir.join(f);
            if p.is_file() {
                None
            } else {
                Some(p.display().to_string())
            }
        })
        .collect();

    if missing.is_empty() {
        return Ok(());
    }

    // Common mistake 1: extracted into an extra "mnist" or "MNIST" directory.
    for sub in ["mnist", "MNIST"] {
        let subdir = data_dir.join(sub);
        if required.iter().all(|f| subdir.join(f).is_file()) {
            return Err(format!(
                "files found under '{}' but expected directly under '{}'.\nHint: move the MNIST files up one directory level.",
                subdir.display(),
                data_dir.display()
            ));
        }
    }

    // Common mistake 2: gzip files present but not decompressed.
    let gz_present: Vec<String> = required
        .iter()
        .filter_map(|f| {
            let gz = data_dir.join(format!("{f}.gz"));
            if gz.is_file() {
                Some(gz.display().to_string())
            } else {
                None
            }
        })
        .collect();
    if gz_present.len() == required.len() {
        return Err(format!(
            "found gzip-compressed MNIST files, but the uncompressed .idx*.ubyte files are missing.\nHint: gunzip these files into '{}'.\nMissing: {}",
            data_dir.display(),
            missing.join(", ")
        ));
    }

    // Common mistake 3: wrong filename variant with "-idx" in name.
    let dashed = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ];
    if dashed.iter().all(|f| data_dir.join(f).is_file()) {
        return Err(format!(
            "found MNIST files with names like 'train-images-idx3-ubyte', but this repo expects names like 'train-images.idx3-ubyte'.\nHint: rename files to use a dot before idx (e.g. train-images.idx3-ubyte)."
        ));
    }

    // Fallback: list what is missing.
    let mut msg = String::new();
    msg.push_str("missing required MNIST files:\n");
    for m in &missing {
        msg.push_str(&format!("  - {m}\n"));
    }
    msg.push_str("\nExpected layout:\n");
    msg.push_str(&format!(
        "  {}/train-images.idx3-ubyte\n",
        data_dir.display()
    ));
    msg.push_str(&format!(
        "  {}/train-labels.idx1-ubyte\n",
        data_dir.display()
    ));
    msg.push_str(&format!(
        "  {}/t10k-images.idx3-ubyte\n",
        data_dir.display()
    ));
    msg.push_str(&format!(
        "  {}/t10k-labels.idx1-ubyte\n",
        data_dir.display()
    ));
    msg.push_str("\nOfficial source: http://yann.lecun.com/exdb/mnist/\n");
    msg.push_str("Mirror:          https://storage.googleapis.com/cvdf-datasets/mnist/\n");
    msg.push_str("License/terms:   MNIST is provided by Yann LeCun & collaborators; review the terms on the official site before redistribution/use.\n");
    Err(msg)
}

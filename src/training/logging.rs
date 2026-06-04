use super::metrics::TrainingMetrics;
use std::fs::File;
use std::io::{BufWriter, Write};

fn open_csv_writer(path: &str) -> Result<BufWriter<File>, std::io::Error> {
    Ok(BufWriter::new(File::create(path)?))
}

/// Writes per-epoch training metrics to a CSV file.
///
/// The CSV format matches what all binaries produce:
/// ```text
/// epoch,train_loss,train_time,val_loss,val_accuracy
/// 1,0.354321,12.3,0.312456,91.23
/// ```
///
/// # Examples
///
/// ```no_run
/// use rust_neural_networks::training::{CsvTrainingLogger, TrainingMetrics};
///
/// let mut logger = CsvTrainingLogger::new("./logs/training_loss_mlp.csv").unwrap();
/// logger.write_header().unwrap();
/// let metrics = TrainingMetrics {
///     train_loss: 0.5,
///     val_loss: 0.4,
///     val_accuracy: 88.5,
///     train_time: 10.0,
///     learning_rate: 0.01,
/// };
/// logger.write_epoch(1, &metrics).unwrap();
/// ```
pub struct CsvTrainingLogger {
    writer: BufWriter<File>,
}

impl CsvTrainingLogger {
    /// Creates a new CSV logger that opens (or creates) the file at `path` and buffers writes.
    ///
    /// The directory containing `path` must already exist; create it beforehand if necessary
    /// (for example with `std::fs::create_dir_all("logs")`).
    ///
    /// # Errors
    ///
    /// Returns any `std::io::Error` produced while creating the file.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut logger = CsvTrainingLogger::new("logs/train.csv").unwrap();
    /// logger.write_header().unwrap();
    /// logger.flush().unwrap();
    /// ```
    pub fn new(path: &str) -> Result<Self, std::io::Error> {
        Ok(Self {
            writer: open_csv_writer(path)?,
        })
    }

    /// Writes the CSV header line: `epoch,train_loss,train_time,val_loss,val_accuracy`.
    ///
    /// # Errors
    ///
    /// Returns an `std::io::Error` if writing to the underlying file fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::training::CsvTrainingLogger;
    /// use tempfile::NamedTempFile;
    ///
    /// let tmp = NamedTempFile::new().unwrap();
    /// let mut logger = CsvTrainingLogger::new(tmp.path().to_str().unwrap()).unwrap();
    /// logger.write_header().unwrap();
    /// logger.flush().unwrap();
    /// let contents = std::fs::read_to_string(tmp.path()).unwrap();
    /// assert!(contents.starts_with("epoch,train_loss,train_time,val_loss,val_accuracy\n"));
    /// ```
    pub fn write_header(&mut self) -> Result<(), std::io::Error> {
        writeln!(
            self.writer,
            "epoch,train_loss,train_time,val_loss,val_accuracy"
        )
    }

    /// Appends a CSV row for the given epoch using the supplied metrics.
    ///
    /// The epoch is one-based; the row is written in the format
    /// `epoch,train_loss,train_time,val_loss,val_accuracy`.
    ///
    /// # Parameters
    ///
    /// - `epoch`: One-based epoch index used as the first column in the row.
    /// - `metrics`: TrainingMetrics providing `train_loss`, `train_time`, `val_loss`, and `val_accuracy`.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the row was written successfully, `Err(std::io::Error)` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::training::{CsvTrainingLogger, TrainingMetrics};
    /// use tempfile::NamedTempFile;
    ///
    /// let tmp = NamedTempFile::new().unwrap();
    /// let path = tmp.path().to_str().unwrap();
    /// let mut logger = CsvTrainingLogger::new(path).unwrap();
    /// let metrics = TrainingMetrics {
    ///     train_loss: 0.5,
    ///     val_loss: 0.4,
    ///     val_accuracy: 88.5,
    ///     train_time: 10.0,
    ///     learning_rate: 0.01,
    /// };
    /// logger.write_header().unwrap();
    /// logger.write_epoch(1, &metrics).unwrap();
    /// logger.flush().unwrap();
    /// ```
    pub fn write_epoch(
        &mut self,
        epoch: usize,
        metrics: &TrainingMetrics,
    ) -> Result<(), std::io::Error> {
        writeln!(
            self.writer,
            "{},{},{},{},{}",
            epoch, metrics.train_loss, metrics.train_time, metrics.val_loss, metrics.val_accuracy,
        )
    }

    /// Flushes the internal buffer, forcing any buffered data to be written to the underlying file.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::training::CsvTrainingLogger;
    /// use tempfile::NamedTempFile;
    ///
    /// let tmp = NamedTempFile::new().unwrap();
    /// let mut logger = CsvTrainingLogger::new(tmp.path().to_str().unwrap()).unwrap();
    /// logger.flush().unwrap();
    /// ```
    ///
    /// Returns `Ok(())` on success, or an `std::io::Error` if the flush fails.
    pub fn flush(&mut self) -> Result<(), std::io::Error> {
        self.writer.flush()
    }
}

/// Writes per-epoch per-layer gradient norms to a CSV file.
///
/// The CSV format records one row per layer per epoch, allowing downstream tools
/// to visualise gradient flow and detect vanishing or exploding gradients:
/// ```text
/// epoch,layer_name,grad_norm_weights,grad_norm_biases
/// 1,layer_0_dense,0.042314,0.003217
/// 1,layer_1_dense,0.021756,0.001843
/// ```
///
/// # Examples
///
/// ```ignore
/// use rust_neural_networks::training::CsvGradientLogger;
///
/// let mut logger = CsvGradientLogger::new("logs/gradients.csv").unwrap();
/// logger.write_header().unwrap();
/// logger.write_layer(1, "layer_0_dense", 0.042314, 0.003217).unwrap();
/// logger.write_layer(1, "layer_1_dense", 0.021756, 0.001843).unwrap();
/// logger.flush().unwrap();
/// ```
pub struct CsvGradientLogger {
    writer: BufWriter<File>,
}

impl CsvGradientLogger {
    /// Creates a new CSV logger that opens (or creates) the file at `path` and buffers writes.
    ///
    /// The directory containing `path` must already exist; create it beforehand if necessary
    /// (for example with `std::fs::create_dir_all("logs")`).
    ///
    /// # Errors
    ///
    /// Returns any `std::io::Error` produced while creating the file.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut logger = CsvGradientLogger::new("logs/gradients.csv").unwrap();
    /// logger.write_header().unwrap();
    /// logger.flush().unwrap();
    /// ```
    pub fn new(path: &str) -> Result<Self, std::io::Error> {
        Ok(Self {
            writer: open_csv_writer(path)?,
        })
    }

    /// Writes the CSV header line for gradient-norm logs.
    ///
    /// The header is: `epoch,layer_name,grad_norm_weights,grad_norm_biases`.
    ///
    /// Returns `Ok(())` if the header was written successfully, or an `std::io::Error` if the write fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Create a logger, write the header, and flush it to disk.
    /// let mut logger = CsvGradientLogger::new("gradients.csv").unwrap();
    /// logger.write_header().unwrap();
    /// logger.flush().unwrap();
    /// let contents = std::fs::read_to_string("gradients.csv").unwrap();
    /// assert!(contents.starts_with("epoch,layer_name,grad_norm_weights,grad_norm_biases\n"));
    /// ```
    pub fn write_header(&mut self) -> Result<(), std::io::Error> {
        writeln!(
            self.writer,
            "epoch,layer_name,grad_norm_weights,grad_norm_biases"
        )
    }

    /// Appends a CSV row for the given epoch and layer containing L2 gradient norms.
    ///
    /// The `epoch` is one-based. The written row has the format:
    /// `{epoch},{layer_name},{weight_norm},{bias_norm}`.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err(std::io::Error)` on I/O failure.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut logger = CsvGradientLogger::new("/tmp/gradients.csv").unwrap();
    /// logger.write_header().unwrap();
    /// logger.write_layer(1, "conv1", 0.123, 0.045).unwrap();
    /// logger.flush().unwrap();
    /// ```
    pub fn write_layer(
        &mut self,
        epoch: usize,
        layer_name: &str,
        weight_norm: f32,
        bias_norm: f32,
    ) -> Result<(), std::io::Error> {
        writeln!(
            self.writer,
            "{},{},{},{}",
            epoch, layer_name, weight_norm, bias_norm,
        )
    }

    /// Flushes the internal buffer, forcing any buffered data to be written to the underlying file.
    ///
    /// # Examples
    ///
    /// ```
    /// use rust_neural_networks::training::CsvGradientLogger;
    /// use tempfile::NamedTempFile;
    ///
    /// let tmp = NamedTempFile::new().unwrap();
    /// let mut logger = CsvGradientLogger::new(tmp.path().to_str().unwrap()).unwrap();
    /// logger.flush().unwrap();
    /// ```
    ///
    /// Returns `Ok(())` on success, or an `std::io::Error` if the flush fails.
    pub fn flush(&mut self) -> Result<(), std::io::Error> {
        self.writer.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_training_logger_writes_five_column_header() {
        let file = NamedTempFile::new().expect("create temp file");
        let path = file.path().to_string_lossy().to_string();

        let mut logger = CsvTrainingLogger::new(&path).expect("create logger");
        logger.write_header().expect("write header");
        logger.flush().expect("flush");

        let content = std::fs::read_to_string(&path).expect("read file");
        assert_eq!(
            content.lines().next(),
            Some("epoch,train_loss,train_time,val_loss,val_accuracy")
        );
    }

    #[test]
    fn test_training_logger_writes_five_column_epoch() {
        let file = NamedTempFile::new().expect("create temp file");
        let path = file.path().to_string_lossy().to_string();
        let metrics = TrainingMetrics {
            train_loss: 0.5,
            val_loss: 0.4,
            val_accuracy: 88.5,
            train_time: 10.0,
            learning_rate: 0.01,
        };

        let mut logger = CsvTrainingLogger::new(&path).expect("create logger");
        logger.write_header().expect("write header");
        logger.write_epoch(1, &metrics).expect("write epoch");
        logger.flush().expect("flush");

        let content = std::fs::read_to_string(&path).expect("read file");
        let row = content.lines().nth(1).expect("epoch row");
        assert_eq!(row.split(',').count(), 5);
        assert_eq!(row, "1,0.5,10,0.4,88.5");
    }
}

/// Scalar metrics produced at the end of a training epoch.
///
/// All binaries compute the same five quantities each epoch; this struct
/// avoids repeated long parameter lists.
#[derive(Debug, Clone, Copy)]
pub struct TrainingMetrics {
    /// Average cross-entropy loss over the training set for this epoch.
    pub train_loss: f32,
    /// Average cross-entropy loss over the validation set for this epoch.
    pub val_loss: f32,
    /// Validation accuracy as a percentage (0.0–100.0).
    pub val_accuracy: f32,
    /// Wall-clock time (seconds) spent on the training portion of the epoch.
    pub train_time: f32,
    /// Learning rate that was active during this epoch's parameter updates.
    pub learning_rate: f32,
}

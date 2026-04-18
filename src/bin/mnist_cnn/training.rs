use super::*;

/// Creates a learning-rate scheduler configured for the given learning rate, epoch count, and optional config file path.
///
/// The returned scheduler is configured according to `learning_rate`, `epochs`, and any overrides found at `config_path`.
///
/// # Examples
///
/// ```
/// // Construct a scheduler with a 0.01 base learning rate for 10 epochs and no config file.
/// let _scheduler = scheduler_from_args(0.01_f32, 10_usize, None);
/// ```
pub(crate) fn scheduler_from_args(
    learning_rate: f32,
    epochs: usize,
    config_path: Option<&str>,
) -> Box<dyn LRScheduler> {
    create_scheduler_from_config(learning_rate, epochs, config_path)
}

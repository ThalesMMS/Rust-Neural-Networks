use super::config::SweepConfig;
use crate::config::{load_config, TrainingConfig};
use std::error::Error;

/// Returns 1 when the slice is empty, otherwise its length.
/// Used to include one "pass-through" iteration for sweep dimensions that have no values.
fn nonzero_len<T>(s: &[T]) -> usize {
    s.len().max(1)
}

fn invalid_data(msg: impl Into<String>) -> Box<dyn Error> {
    Box::new(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        msg.into(),
    ))
}

struct IndexProduct<const N: usize> {
    dimensions: [usize; N],
    current: Option<[usize; N]>,
}

impl<const N: usize> IndexProduct<N> {
    fn new(dimensions: [usize; N]) -> Self {
        debug_assert!(dimensions.iter().all(|&dimension| dimension > 0));
        Self {
            dimensions,
            current: Some([0; N]),
        }
    }
}

impl<const N: usize> Iterator for IndexProduct<N> {
    type Item = [usize; N];

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.current?;
        let mut next = item;

        for dimension_idx in (0..N).rev() {
            next[dimension_idx] += 1;
            if next[dimension_idx] < self.dimensions[dimension_idx] {
                self.current = Some(next);
                return Some(item);
            }
            next[dimension_idx] = 0;
        }

        self.current = None;
        Some(item)
    }
}

fn push_scheduler_configs(
    configs: &mut Vec<TrainingConfig>,
    common_config: TrainingConfig,
    scheduler_type_swept: bool,
    step_sizes: &[usize],
    gammas: &[f32],
    decay_rates: &[f32],
    min_lrs: &[f32],
    t_maxes: &[usize],
) -> Result<(), Box<dyn Error>> {
    let scheduler_type = common_config.scheduler_type.clone();

    match scheduler_type.as_str() {
        "step_decay" => {
            if !scheduler_type_swept
                && (!decay_rates.is_empty() || !min_lrs.is_empty() || !t_maxes.is_empty())
            {
                return Err(invalid_data(
                    "scheduler_type 'step_decay' cannot sweep decay_rate, min_lr, or T_max",
                ));
            }
            if scheduler_type_swept && (step_sizes.is_empty() || gammas.is_empty()) {
                return Err(invalid_data(
                    "scheduler_type 'step_decay' requires non-empty step_size and gamma sweep values",
                ));
            }
            for ss_idx in 0..nonzero_len(step_sizes) {
                for g_idx in 0..nonzero_len(gammas) {
                    let mut config = common_config.clone();
                    if !step_sizes.is_empty() {
                        config.step_size = Some(step_sizes[ss_idx]);
                    }
                    if !gammas.is_empty() {
                        config.gamma = Some(gammas[g_idx]);
                    }
                    configs.push(config);
                }
            }
        }
        "exponential" => {
            if !scheduler_type_swept
                && (!step_sizes.is_empty()
                    || !gammas.is_empty()
                    || !min_lrs.is_empty()
                    || !t_maxes.is_empty())
            {
                return Err(invalid_data(
                    "scheduler_type 'exponential' cannot sweep step_size, gamma, min_lr, or T_max",
                ));
            }
            if scheduler_type_swept && decay_rates.is_empty() {
                return Err(invalid_data(
                    "scheduler_type 'exponential' requires non-empty decay_rate sweep values",
                ));
            }
            for dr_idx in 0..nonzero_len(decay_rates) {
                let mut config = common_config.clone();
                if !decay_rates.is_empty() {
                    config.decay_rate = Some(decay_rates[dr_idx]);
                }
                configs.push(config);
            }
        }
        "cosine_annealing" => {
            if !scheduler_type_swept
                && (!step_sizes.is_empty() || !gammas.is_empty() || !decay_rates.is_empty())
            {
                return Err(invalid_data(
                    "scheduler_type 'cosine_annealing' cannot sweep step_size, gamma, or decay_rate",
                ));
            }
            if scheduler_type_swept && (min_lrs.is_empty() || t_maxes.is_empty()) {
                return Err(invalid_data(
                    "scheduler_type 'cosine_annealing' requires non-empty min_lr and T_max sweep values",
                ));
            }
            for mlr_idx in 0..nonzero_len(min_lrs) {
                for tm_idx in 0..nonzero_len(t_maxes) {
                    let mut config = common_config.clone();
                    if !min_lrs.is_empty() {
                        config.min_lr = Some(min_lrs[mlr_idx]);
                    }
                    if !t_maxes.is_empty() {
                        config.T_max = Some(t_maxes[tm_idx]);
                    }
                    configs.push(config);
                }
            }
        }
        _ => {
            return Err(invalid_data(format!(
                "Unknown scheduler_type '{}'",
                scheduler_type
            )));
        }
    }

    Ok(())
}

fn validate_fixed_activation_sweeps(
    activation_function: Option<&str>,
    activation_function_swept: bool,
    leaky_relu_alphas: &[f32],
    elu_alphas: &[f32],
) -> Result<(), Box<dyn Error>> {
    if activation_function_swept {
        return Ok(());
    }

    match activation_function.unwrap_or("relu") {
        "leaky_relu" => {
            if !elu_alphas.is_empty() {
                return Err(invalid_data(
                    "activation_function 'leaky_relu' cannot sweep elu_alpha",
                ));
            }
        }
        "elu" => {
            if !leaky_relu_alphas.is_empty() {
                return Err(invalid_data(
                    "activation_function 'elu' cannot sweep leaky_relu_alpha",
                ));
            }
        }
        activation => {
            if !leaky_relu_alphas.is_empty() || !elu_alphas.is_empty() {
                return Err(invalid_data(format!(
                    "activation_function '{}' cannot sweep leaky_relu_alpha or elu_alpha",
                    activation
                )));
            }
        }
    }

    Ok(())
}

/// Produce all TrainingConfig combinations from a SweepConfig.
///
/// Loads the base configuration specified by `sweep.base_config` and returns one
/// `TrainingConfig` per combination formed by taking the Cartesian product of
/// all provided sweep parameter lists. If a given sweep list is empty or
/// absent, the base value is preserved for that dimension; if all sweep lists
/// are absent, a single config equal to the base config is returned.
///
/// # Errors
///
/// Returns an error if loading or parsing the base config file fails.
///
/// # Examples
///
/// ```ignore
/// let sweep = load_sweep_config("config/sweeps/example_sweep.json").unwrap();
/// let configs = generate_configs(&sweep).unwrap();
/// // If the sweep defines 2 learning rates and 3 batch sizes, 6 configs are produced
/// assert_eq!(configs.len(), 2 * 3);
/// ```
pub fn generate_configs(sweep: &SweepConfig) -> Result<Vec<TrainingConfig>, Box<dyn Error>> {
    let base_config = load_config(&sweep.base_config)?;

    let learning_rates = sweep.learning_rate.as_deref().unwrap_or(&[]);
    let batch_sizes = sweep.batch_size.as_deref().unwrap_or(&[]);
    let epochs_list = sweep.epochs.as_deref().unwrap_or(&[]);
    let validation_splits = sweep.validation_split.as_deref().unwrap_or(&[]);
    let early_stopping_patiences = sweep.early_stopping_patience.as_deref().unwrap_or(&[]);
    let early_stopping_min_deltas = sweep.early_stopping_min_delta.as_deref().unwrap_or(&[]);
    let scheduler_types = sweep.scheduler_type.as_deref().unwrap_or(&[]);
    let step_sizes = sweep.step_size.as_deref().unwrap_or(&[]);
    let gammas = sweep.gamma.as_deref().unwrap_or(&[]);
    let decay_rates = sweep.decay_rate.as_deref().unwrap_or(&[]);
    let min_lrs = sweep.min_lr.as_deref().unwrap_or(&[]);
    let t_maxes = sweep.T_max.as_deref().unwrap_or(&[]);
    let activation_functions = sweep.activation_function.as_deref().unwrap_or(&[]);
    let leaky_relu_alphas = sweep.leaky_relu_alpha.as_deref().unwrap_or(&[]);
    let elu_alphas = sweep.elu_alpha.as_deref().unwrap_or(&[]);

    let num_learning_rates = nonzero_len(learning_rates);
    let num_batch_sizes = nonzero_len(batch_sizes);
    let num_epochs = nonzero_len(epochs_list);
    let num_validation_splits = nonzero_len(validation_splits);
    let num_early_stopping_patiences = nonzero_len(early_stopping_patiences);
    let num_early_stopping_min_deltas = nonzero_len(early_stopping_min_deltas);
    let num_scheduler_types = nonzero_len(scheduler_types);
    let num_activation_functions = nonzero_len(activation_functions);

    let mut configs = Vec::new();

    let dimensions = [
        num_learning_rates,
        num_batch_sizes,
        num_epochs,
        num_validation_splits,
        num_early_stopping_patiences,
        num_early_stopping_min_deltas,
        num_scheduler_types,
        num_activation_functions,
    ];

    for [lr_idx, bs_idx, ep_idx, vs_idx, esp_idx, esmd_idx, st_idx, af_idx] in
        IndexProduct::new(dimensions)
    {
        let mut common_config = base_config.clone();

        if !learning_rates.is_empty() {
            common_config.learning_rate = Some(learning_rates[lr_idx]);
        }
        if !batch_sizes.is_empty() {
            common_config.batch_size = Some(batch_sizes[bs_idx]);
        }
        if !epochs_list.is_empty() {
            common_config.epochs = Some(epochs_list[ep_idx]);
        }
        if !validation_splits.is_empty() {
            common_config.validation_split = Some(validation_splits[vs_idx]);
        }
        if !early_stopping_patiences.is_empty() {
            common_config.early_stopping_patience = Some(early_stopping_patiences[esp_idx]);
        }
        if !early_stopping_min_deltas.is_empty() {
            common_config.early_stopping_min_delta = Some(early_stopping_min_deltas[esmd_idx]);
        }
        if !scheduler_types.is_empty() {
            common_config.scheduler_type = scheduler_types[st_idx].clone();
            common_config.step_size = None;
            common_config.gamma = None;
            common_config.decay_rate = None;
            common_config.min_lr = None;
            common_config.T_max = None;
        }
        if !activation_functions.is_empty() {
            common_config.activation_function = Some(activation_functions[af_idx].clone());
            common_config.leaky_relu_alpha = None;
            common_config.elu_alpha = None;
        }
        validate_fixed_activation_sweeps(
            common_config.activation_function.as_deref(),
            !activation_functions.is_empty(),
            leaky_relu_alphas,
            elu_alphas,
        )?;

        match common_config
            .activation_function
            .as_deref()
            .unwrap_or("relu")
        {
            "leaky_relu" => {
                common_config.elu_alpha = None;
                for lra_idx in 0..nonzero_len(leaky_relu_alphas) {
                    let mut config = common_config.clone();
                    if !leaky_relu_alphas.is_empty() {
                        config.leaky_relu_alpha = Some(leaky_relu_alphas[lra_idx]);
                    }
                    push_scheduler_configs(
                        &mut configs,
                        config,
                        !scheduler_types.is_empty(),
                        step_sizes,
                        gammas,
                        decay_rates,
                        min_lrs,
                        t_maxes,
                    )?;
                }
            }
            "elu" => {
                common_config.leaky_relu_alpha = None;
                for ea_idx in 0..nonzero_len(elu_alphas) {
                    let mut config = common_config.clone();
                    if !elu_alphas.is_empty() {
                        config.elu_alpha = Some(elu_alphas[ea_idx]);
                    }
                    push_scheduler_configs(
                        &mut configs,
                        config,
                        !scheduler_types.is_empty(),
                        step_sizes,
                        gammas,
                        decay_rates,
                        min_lrs,
                        t_maxes,
                    )?;
                }
            }
            _ => {
                common_config.leaky_relu_alpha = None;
                common_config.elu_alpha = None;
                push_scheduler_configs(
                    &mut configs,
                    common_config,
                    !scheduler_types.is_empty(),
                    step_sizes,
                    gammas,
                    decay_rates,
                    min_lrs,
                    t_maxes,
                )?;
            }
        }
    }

    Ok(configs)
}

#[cfg(test)]
mod tests;

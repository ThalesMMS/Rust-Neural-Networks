#!/usr/bin/env python3
"""
Gradient Flow Visualization Tool
Visualizes gradient magnitudes flowing through neural network layers during training
to help understand vanishing/exploding gradient problems
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import sys

# Configuration
LOG_DIR = "./logs"
OUTPUT_FILE = "gradient_flow.png"
ANIMATED_OUTPUT_FILE = "gradient_flow_animated.gif"

# Default gradient log files
GRADIENT_LOGS = {
    'mlp': f"{LOG_DIR}/gradients_mlp.csv",
    'cnn': f"{LOG_DIR}/gradients_cnn.csv",
    'cifar10': f"{LOG_DIR}/gradients_cifar10.csv",
    'attention': f"{LOG_DIR}/gradients_attention.csv",
}

def load_gradient_data(filepath):
    """
    Load gradient data from CSV file.
    Format: epoch,layer_name,grad_norm_weights,grad_norm_biases

    Returns:
        dict: Mapping from layer_name to dict containing:
            - epochs: array of epoch numbers
            - grad_weights: array of weight gradient norms
            - grad_biases: array of bias gradient norms
        None if file doesn't exist or parsing fails
    """
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None

    try:
        # Read file line by line to parse mixed data types
        layers_data = {}

        with open(filepath, 'r') as f:
            # Skip header
            header = f.readline().strip()
            if not header.startswith('epoch'):
                print(f"Warning: Unexpected header format in {filepath}: {header}")
                return None

            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(',')
                if len(parts) != 4:
                    print(f"Warning: Skipping malformed line: {line}")
                    continue

                epoch = int(parts[0])
                layer_name = parts[1].strip()
                grad_norm_weights = float(parts[2])
                grad_norm_biases = float(parts[3])

                # Initialize layer data if first time seeing this layer
                if layer_name not in layers_data:
                    layers_data[layer_name] = {
                        'epochs': [],
                        'grad_weights': [],
                        'grad_biases': []
                    }

                # Append data
                layers_data[layer_name]['epochs'].append(epoch)
                layers_data[layer_name]['grad_weights'].append(grad_norm_weights)
                layers_data[layer_name]['grad_biases'].append(grad_norm_biases)

        # Convert lists to numpy arrays
        for layer_name in layers_data:
            layers_data[layer_name]['epochs'] = np.array(layers_data[layer_name]['epochs'])
            layers_data[layer_name]['grad_weights'] = np.array(layers_data[layer_name]['grad_weights'])
            layers_data[layer_name]['grad_biases'] = np.array(layers_data[layer_name]['grad_biases'])

        return layers_data

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_layer_stats(layers_data):
    """
    Compute statistics for gradient data across all layers.

    Parameters:
        layers_data (dict): Layer data from load_gradient_data()

    Returns:
        dict: Statistics including min, max, mean for each layer
    """
    stats = {}

    for layer_name, data in layers_data.items():
        grad_weights = data['grad_weights']
        grad_biases = data['grad_biases']

        stats[layer_name] = {
            'weights': {
                'min': np.min(grad_weights),
                'max': np.max(grad_weights),
                'mean': np.mean(grad_weights),
                'std': np.std(grad_weights)
            },
            'biases': {
                'min': np.min(grad_biases),
                'max': np.max(grad_biases),
                'mean': np.mean(grad_biases),
                'std': np.std(grad_biases)
            }
        }

    return stats

def detect_gradient_issues(layers_data, vanishing_threshold=1e-5, exploding_threshold=100.0):
    """
    Detect vanishing and exploding gradient problems in the training data.

    Prints warnings when gradients fall below the vanishing threshold or exceed
    the exploding threshold, similar to gradient clipping utilities.

    Parameters:
        layers_data (dict): Layer data from load_gradient_data()
        vanishing_threshold (float): Threshold below which gradients are considered vanishing (default: 1e-5)
        exploding_threshold (float): Threshold above which gradients are considered exploding (default: 100.0)

    Returns:
        dict: Summary of detected issues with counts per layer
    """
    issues = {
        'vanishing': {},
        'exploding': {}
    }

    print("\n" + "="*60)
    print("GRADIENT HEALTH CHECK")
    print("="*60)
    print(f"Vanishing threshold: < {vanishing_threshold:.2e}")
    print(f"Exploding threshold: > {exploding_threshold:.2e}")
    print("-"*60)

    has_issues = False

    for layer_name, data in layers_data.items():
        grad_weights = data['grad_weights']
        grad_biases = data['grad_biases']
        epochs = data['epochs']

        # Check for vanishing gradients in weights
        vanishing_weights = grad_weights < vanishing_threshold
        vanishing_weight_count = np.sum(vanishing_weights)

        # Check for exploding gradients in weights
        exploding_weights = grad_weights > exploding_threshold
        exploding_weight_count = np.sum(exploding_weights)

        # Check for vanishing gradients in biases
        vanishing_biases = grad_biases < vanishing_threshold
        vanishing_bias_count = np.sum(vanishing_biases)

        # Check for exploding gradients in biases
        exploding_biases = grad_biases > exploding_threshold
        exploding_bias_count = np.sum(exploding_biases)

        # Store issue counts
        issues['vanishing'][layer_name] = {
            'weights': vanishing_weight_count,
            'biases': vanishing_bias_count
        }
        issues['exploding'][layer_name] = {
            'weights': exploding_weight_count,
            'biases': exploding_bias_count
        }

        # Print warnings for this layer
        layer_has_issues = False

        if vanishing_weight_count > 0:
            has_issues = True
            layer_has_issues = True
            vanishing_epochs = epochs[vanishing_weights]
            min_grad = np.min(grad_weights[vanishing_weights])
            print(f"\n⚠️  WARNING: Vanishing weight gradients detected in '{layer_name}'")
            print(f"   Occurrences: {vanishing_weight_count}/{len(grad_weights)} epochs")
            print(f"   Affected epochs: {vanishing_epochs.tolist()}")
            print(f"   Minimum gradient: {min_grad:.2e}")

        if exploding_weight_count > 0:
            has_issues = True
            layer_has_issues = True
            exploding_epochs = epochs[exploding_weights]
            max_grad = np.max(grad_weights[exploding_weights])
            print(f"\n⚠️  WARNING: Exploding weight gradients detected in '{layer_name}'")
            print(f"   Occurrences: {exploding_weight_count}/{len(grad_weights)} epochs")
            print(f"   Affected epochs: {exploding_epochs.tolist()}")
            print(f"   Maximum gradient: {max_grad:.2e}")

        if vanishing_bias_count > 0:
            has_issues = True
            layer_has_issues = True
            vanishing_epochs = epochs[vanishing_biases]
            min_grad = np.min(grad_biases[vanishing_biases])
            print(f"\n⚠️  WARNING: Vanishing bias gradients detected in '{layer_name}'")
            print(f"   Occurrences: {vanishing_bias_count}/{len(grad_biases)} epochs")
            print(f"   Affected epochs: {vanishing_epochs.tolist()}")
            print(f"   Minimum gradient: {min_grad:.2e}")

        if exploding_bias_count > 0:
            has_issues = True
            layer_has_issues = True
            exploding_epochs = epochs[exploding_biases]
            max_grad = np.max(grad_biases[exploding_biases])
            print(f"\n⚠️  WARNING: Exploding bias gradients detected in '{layer_name}'")
            print(f"   Occurrences: {exploding_bias_count}/{len(grad_biases)} epochs")
            print(f"   Affected epochs: {exploding_epochs.tolist()}")
            print(f"   Maximum gradient: {max_grad:.2e}")

        if not layer_has_issues:
            print(f"\n✓ '{layer_name}': No gradient issues detected")

    if not has_issues:
        print("\n✓ All layers have healthy gradient magnitudes!")
    else:
        print("\n⚠️  Gradient issues detected - consider:")
        print("   • Using gradient clipping to prevent exploding gradients")
        print("   • Adjusting learning rate or network architecture for vanishing gradients")
        print("   • Reviewing activation functions (ReLU vs sigmoid/tanh)")
        print("   • Adding batch normalization or residual connections")

    print("="*60)

    return issues

def plot_gradient_flow(layers_data, model_type='mlp', output_file=OUTPUT_FILE):
    """
    Create and save gradient magnitude plots showing how gradients flow through layers over epochs.

    Parameters:
        layers_data (dict): Layer data from load_gradient_data()
        model_type (str): Model type for title ('mlp', 'cnn', etc.)
        output_file (str): File path where the generated figure will be saved

    Returns:
        matplotlib.figure.Figure: The Matplotlib Figure containing the plotted graphs
    """
    n_layers = len(layers_data)
    layer_names = sorted(layers_data.keys())

    # Create figure with subplots - one row per layer, two columns (weights + biases)
    fig, axes = plt.subplots(n_layers, 2, figsize=(14, 5 * n_layers))
    fig.suptitle(f'{model_type.upper()} Model - Gradient Flow Analysis',
                 fontsize=16, fontweight='bold')

    # Handle single layer case
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    # Color palette for different layers
    colors = plt.cm.tab10(np.linspace(0, 1, n_layers))

    for idx, layer_name in enumerate(layer_names):
        data = layers_data[layer_name]
        epochs = data['epochs']
        grad_weights = data['grad_weights']
        grad_biases = data['grad_biases']

        # Plot weight gradients
        ax_weights = axes[idx, 0]
        ax_weights.plot(epochs, grad_weights, marker='o', linewidth=2,
                       color=colors[idx], markersize=4, label=layer_name)
        ax_weights.set_xlabel('Epoch', fontsize=11)
        ax_weights.set_ylabel('Gradient Magnitude', fontsize=11)
        ax_weights.set_title(f'{layer_name} - Weight Gradients',
                            fontsize=12, fontweight='bold')
        ax_weights.grid(True, alpha=0.3, linestyle='--')
        ax_weights.set_yscale('log')  # Log scale to handle wide range of values

        # Add mean line
        mean_val = np.mean(grad_weights)
        ax_weights.axhline(y=mean_val, color='red', linestyle='--',
                          linewidth=1, alpha=0.7, label=f'Mean: {mean_val:.2e}')
        ax_weights.legend(fontsize=9)

        # Plot bias gradients
        ax_biases = axes[idx, 1]
        ax_biases.plot(epochs, grad_biases, marker='s', linewidth=2,
                      color=colors[idx], markersize=4, label=layer_name)
        ax_biases.set_xlabel('Epoch', fontsize=11)
        ax_biases.set_ylabel('Gradient Magnitude', fontsize=11)
        ax_biases.set_title(f'{layer_name} - Bias Gradients',
                           fontsize=12, fontweight='bold')
        ax_biases.grid(True, alpha=0.3, linestyle='--')
        ax_biases.set_yscale('log')  # Log scale to handle wide range of values

        # Add mean line
        mean_val = np.mean(grad_biases)
        ax_biases.axhline(y=mean_val, color='red', linestyle='--',
                         linewidth=1, alpha=0.7, label=f'Mean: {mean_val:.2e}')
        ax_biases.legend(fontsize=9)

    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gradient flow visualization saved to: {output_file}")

    return fig

def plot_combined_gradient_flow(layers_data, model_type='mlp'):
    """
    Create combined plots showing all layers on the same axes for comparison.

    Parameters:
        layers_data (dict): Layer data from load_gradient_data()
        model_type (str): Model type for title

    Returns:
        matplotlib.figure.Figure: The combined comparison figure
    """
    layer_names = sorted(layers_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(layer_names)))

    # Create figure with 2 subplots (weights and biases combined)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model_type.upper()} Model - Combined Gradient Flow Comparison',
                 fontsize=14, fontweight='bold')

    # Plot all weight gradients together
    for idx, layer_name in enumerate(layer_names):
        data = layers_data[layer_name]
        epochs = data['epochs']
        grad_weights = data['grad_weights']
        ax1.plot(epochs, grad_weights, marker='o', linewidth=2,
                color=colors[idx], markersize=4, label=layer_name, alpha=0.8)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Gradient Magnitude (log scale)', fontsize=12)
    ax1.set_title('Weight Gradients - All Layers', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    ax1.legend(fontsize=10, loc='best')

    # Plot all bias gradients together
    for idx, layer_name in enumerate(layer_names):
        data = layers_data[layer_name]
        epochs = data['epochs']
        grad_biases = data['grad_biases']
        ax2.plot(epochs, grad_biases, marker='s', linewidth=2,
                color=colors[idx], markersize=4, label=layer_name, alpha=0.8)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Gradient Magnitude (log scale)', fontsize=12)
    ax2.set_title('Bias Gradients - All Layers', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yscale('log')
    ax2.legend(fontsize=10, loc='best')

    plt.tight_layout()

    # Save combined figure
    combined_file = OUTPUT_FILE.replace('.png', '_combined.png')
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    print(f"✓ Combined gradient comparison saved to: {combined_file}")

    return fig

def animate_gradient_flow(layers_data, model_type='mlp', output_file=ANIMATED_OUTPUT_FILE,
                          fps=2, window_size=None):
    """
    Create and save an animated GIF showing gradient evolution over training epochs.

    The animation progressively reveals gradient magnitudes across epochs, helping visualize
    how gradients change during training and identify convergence or instability patterns.

    Parameters:
        layers_data (dict): Layer data from load_gradient_data()
        model_type (str): Model type for title ('mlp', 'cnn', etc.)
        output_file (str): File path where the animated GIF will be saved
        fps (int): Frames per second for the animation (default: 2)
        window_size (int): Optional rolling window size for progressive reveal.
                          If None, shows all data up to current epoch.

    Returns:
        matplotlib.animation.FuncAnimation: The animation object
    """
    print("\nGenerating animated gradient flow visualization...")
    print(f"  Animation parameters: fps={fps}, window_size={window_size}")

    layer_names = sorted(layers_data.keys())
    n_layers = len(layer_names)

    # Get total number of epochs
    first_layer = layer_names[0]
    total_epochs = len(layers_data[first_layer]['epochs'])

    if total_epochs < 2:
        print("Warning: Need at least 2 epochs for animation. Falling back to static plot.")
        return None

    # Create figure with subplots - one row per layer, two columns (weights + biases)
    fig, axes = plt.subplots(n_layers, 2, figsize=(14, 5 * n_layers))
    fig.suptitle(f'{model_type.upper()} Model - Gradient Flow Animation',
                 fontsize=16, fontweight='bold')

    # Handle single layer case
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    # Color palette for different layers
    colors = plt.cm.tab10(np.linspace(0, 1, n_layers))

    # Pre-compute global min/max for consistent y-axis scaling
    all_weight_grads = []
    all_bias_grads = []
    for layer_name in layer_names:
        all_weight_grads.extend(layers_data[layer_name]['grad_weights'])
        all_bias_grads.extend(layers_data[layer_name]['grad_biases'])

    global_weight_min = np.min(all_weight_grads)
    global_weight_max = np.max(all_weight_grads)
    global_bias_min = np.min(all_bias_grads)
    global_bias_max = np.max(all_bias_grads)

    # Add small margins for better visualization
    weight_margin = (global_weight_max - global_weight_min) * 0.1
    bias_margin = (global_bias_max - global_bias_min) * 0.1

    # Initialize line objects for each layer
    line_objects = {'weights': [], 'biases': [], 'mean_weights': [], 'mean_biases': []}

    for idx, layer_name in enumerate(layer_names):
        # Setup weight gradient subplot
        ax_weights = axes[idx, 0]
        line_w, = ax_weights.plot([], [], marker='o', linewidth=2,
                                   color=colors[idx], markersize=4, label=layer_name)
        line_mean_w, = ax_weights.plot([], [], color='red', linestyle='--',
                                       linewidth=1, alpha=0.7, label='Mean')

        ax_weights.set_xlabel('Epoch', fontsize=11)
        ax_weights.set_ylabel('Gradient Magnitude', fontsize=11)
        ax_weights.set_title(f'{layer_name} - Weight Gradients',
                            fontsize=12, fontweight='bold')
        ax_weights.grid(True, alpha=0.3, linestyle='--')
        ax_weights.set_yscale('log')
        ax_weights.set_ylim(max(global_weight_min - weight_margin, 1e-8),
                           global_weight_max + weight_margin)
        ax_weights.legend(fontsize=9, loc='upper right')

        line_objects['weights'].append(line_w)
        line_objects['mean_weights'].append(line_mean_w)

        # Setup bias gradient subplot
        ax_biases = axes[idx, 1]
        line_b, = ax_biases.plot([], [], marker='s', linewidth=2,
                                 color=colors[idx], markersize=4, label=layer_name)
        line_mean_b, = ax_biases.plot([], [], color='red', linestyle='--',
                                      linewidth=1, alpha=0.7, label='Mean')

        ax_biases.set_xlabel('Epoch', fontsize=11)
        ax_biases.set_ylabel('Gradient Magnitude', fontsize=11)
        ax_biases.set_title(f'{layer_name} - Bias Gradients',
                           fontsize=12, fontweight='bold')
        ax_biases.grid(True, alpha=0.3, linestyle='--')
        ax_biases.set_yscale('log')
        ax_biases.set_ylim(max(global_bias_min - bias_margin, 1e-8),
                          global_bias_max + bias_margin)
        ax_biases.legend(fontsize=9, loc='upper right')

        line_objects['biases'].append(line_b)
        line_objects['mean_biases'].append(line_mean_b)

    # Animation update function
    def update_frame(frame_num):
        """Update function called for each animation frame"""
        # Determine which epochs to show in this frame
        if window_size is None:
            # Progressive reveal: show epochs 0 to frame_num
            start_idx = 0
            end_idx = frame_num + 1
        else:
            # Rolling window: show last 'window_size' epochs
            end_idx = frame_num + 1
            start_idx = max(0, end_idx - window_size)

        # Update each layer's plots
        for idx, layer_name in enumerate(layer_names):
            data = layers_data[layer_name]
            epochs = data['epochs'][start_idx:end_idx]
            grad_weights = data['grad_weights'][start_idx:end_idx]
            grad_biases = data['grad_biases'][start_idx:end_idx]

            # Update weight gradient line
            line_objects['weights'][idx].set_data(epochs, grad_weights)

            # Update weight mean line
            if len(grad_weights) > 0:
                mean_val = np.mean(grad_weights)
                line_objects['mean_weights'][idx].set_data(epochs,
                                                           [mean_val] * len(epochs))

            # Update bias gradient line
            line_objects['biases'][idx].set_data(epochs, grad_biases)

            # Update bias mean line
            if len(grad_biases) > 0:
                mean_val = np.mean(grad_biases)
                line_objects['mean_biases'][idx].set_data(epochs,
                                                          [mean_val] * len(epochs))

            # Update x-axis limits dynamically
            if len(epochs) > 0:
                axes[idx, 0].set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
                axes[idx, 1].set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)

        # Update suptitle to show current epoch
        current_epoch = layers_data[first_layer]['epochs'][frame_num]
        fig.suptitle(f'{model_type.upper()} Model - Gradient Flow Animation (Epoch {current_epoch})',
                    fontsize=16, fontweight='bold')

        # Return all line objects that were modified
        all_lines = (line_objects['weights'] + line_objects['biases'] +
                    line_objects['mean_weights'] + line_objects['mean_biases'])
        return all_lines

    # Create animation
    print(f"  Creating animation with {total_epochs} frames...")
    anim = animation.FuncAnimation(fig, update_frame, frames=total_epochs,
                                   interval=1000/fps, blit=True, repeat=True)

    # Save animation
    print(f"  Saving animation to {output_file}...")
    try:
        # Use PillowWriter for GIF output
        writer = animation.PillowWriter(fps=fps)
        anim.save(output_file, writer=writer, dpi=100)
        print(f"\n✓ Animated gradient flow saved to: {output_file}")
        print(f"  Total frames: {total_epochs}")
        print(f"  Duration: {total_epochs / fps:.1f} seconds")
        print(f"  File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    except Exception as e:
        print(f"\n✗ Error saving animation: {e}")
        print("  Make sure Pillow is installed: pip install Pillow")
        return None

    return anim

def main():
    """Main function for gradient visualization"""
    # Parse command-line arguments
    model_type = 'mlp'  # Default
    animate = False
    fps = 2
    window_size = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]

        if arg in ['--animate', '-a']:
            animate = True
            i += 1
        elif arg in ['--fps', '-f']:
            if i + 1 < len(sys.argv):
                try:
                    fps = int(sys.argv[i + 1])
                    i += 2
                except ValueError:
                    print(f"Error: Invalid fps value '{sys.argv[i + 1]}'")
                    sys.exit(1)
            else:
                print("Error: --fps requires a value")
                sys.exit(1)
        elif arg in ['--window', '-w']:
            if i + 1 < len(sys.argv):
                try:
                    window_size = int(sys.argv[i + 1])
                    i += 2
                except ValueError:
                    print(f"Error: Invalid window size '{sys.argv[i + 1]}'")
                    sys.exit(1)
            else:
                print("Error: --window requires a value")
                sys.exit(1)
        elif arg in ['--model', '-m']:
            if i + 1 < len(sys.argv):
                model_type = sys.argv[i + 1]
                i += 2
            else:
                print("Error: --model requires a value")
                sys.exit(1)
        elif arg in ['--help', '-h']:
            print("Gradient Flow Visualization Tool")
            print("\nUsage: python visualize_gradients.py [options]")
            print("\nOptions:")
            print("  --model, -m MODEL      Model type to visualize (mlp, cnn, cifar10, attention)")
            print("  --animate, -a          Generate animated GIF showing gradient evolution")
            print("  --fps, -f FPS          Frames per second for animation (default: 2)")
            print("  --window, -w SIZE      Rolling window size for animation (default: progressive)")
            print("  --help, -h             Show this help message")
            print("\nExamples:")
            print("  python visualize_gradients.py --model mlp")
            print("  python visualize_gradients.py --animate")
            print("  python visualize_gradients.py --animate --fps 4 --window 10")
            sys.exit(0)
        elif arg in GRADIENT_LOGS:
            model_type = arg
            i += 1
        else:
            print(f"Error: Unknown argument '{arg}'")
            print("Use --help for usage information")
            sys.exit(1)

    gradient_file = GRADIENT_LOGS.get(model_type)
    if not gradient_file:
        print(f"Error: Unknown model type '{model_type}'")
        print(f"Available models: {list(GRADIENT_LOGS.keys())}")
        sys.exit(1)

    # Load gradient data
    print(f"\nLoading gradient data from: {gradient_file}")
    layers_data = load_gradient_data(gradient_file)

    if not layers_data:
        print("Error: No gradient data loaded. Please check the log file.")
        sys.exit(1)

    # Compute statistics
    stats = get_layer_stats(layers_data)

    # Print statistics
    print("\n" + "="*60)
    print("GRADIENT STATISTICS")
    print("="*60)
    for layer_name, layer_stats in stats.items():
        print(f"\n{layer_name}:")
        print(f"  Weights: min={layer_stats['weights']['min']:.6f}, "
              f"max={layer_stats['weights']['max']:.6f}, "
              f"mean={layer_stats['weights']['mean']:.6f}")
        print(f"  Biases:  min={layer_stats['biases']['min']:.6f}, "
              f"max={layer_stats['biases']['max']:.6f}, "
              f"mean={layer_stats['biases']['mean']:.6f}")
    print("="*60)

    # Detect vanishing/exploding gradient issues
    gradient_issues = detect_gradient_issues(layers_data)

    print(f"\n✓ Gradient data parsed successfully")
    print(f"  Layers found: {list(layers_data.keys())}")
    print(f"  Total epochs: {len(layers_data[list(layers_data.keys())[0]]['epochs'])}")

    # Create visualizations
    if animate:
        print("\nGenerating animated gradient flow visualization...")
        anim = animate_gradient_flow(layers_data, model_type=model_type,
                                     output_file=ANIMATED_OUTPUT_FILE,
                                     fps=fps, window_size=window_size)
        if anim is not None:
            print("\n" + "="*60)
            print("ANIMATION COMPLETE")
            print("="*60)
            print(f"Animated GIF created: {ANIMATED_OUTPUT_FILE}")
            print(f"View the animation to see gradient evolution over training")
            print("="*60)
    else:
        print("\nGenerating gradient flow visualizations...")
        plot_gradient_flow(layers_data, model_type=model_type)
        plot_combined_gradient_flow(layers_data, model_type=model_type)

        # Show plots
        plt.show()

if __name__ == "__main__":
    main()

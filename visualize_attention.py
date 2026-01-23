#!/usr/bin/env python3
"""
Attention Weight Visualization
Visualizes attention patterns from the MNIST Attention model to show which
image patches attend to which other patches
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Configuration
LOG_DIR = "./logs"
ATTENTION_WEIGHTS_FILE = f"{LOG_DIR}/attention_weights.txt"
OUTPUT_FILE = "attention_visualization.png"

def parse_attention_weights(filepath):
    """
    Load attention weight matrices from a text log file into a mapping of sample IDs to NumPy arrays.
    
    The file is expected to contain blocks per sample beginning with a header line of the form:
    # Sample: <sample_id>
    followed by rows of numeric values for the attention matrix. Rows may be space- or comma-separated. Blank lines are ignored.
    
    Parameters:
        filepath (str): Path to the attention weights log file.
    
    Returns:
        dict[int, numpy.ndarray] | None: Mapping from sample_id to its attention matrix as a NumPy array, or `None` if the file does not exist or parsing fails.
    """
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None

    try:
        samples = {}
        current_sample = None
        current_matrix = []

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    # Check if it's a sample header
                    if line.startswith('# Sample:'):
                        # Save previous sample if exists
                        if current_sample is not None and current_matrix:
                            samples[current_sample] = np.array(current_matrix)
                        # Start new sample
                        current_sample = int(line.split(':')[1].strip())
                        current_matrix = []
                    continue

                # Parse matrix row (space or comma separated)
                if ',' in line:
                    row = [float(x.strip()) for x in line.split(',') if x.strip()]
                else:
                    row = [float(x.strip()) for x in line.split() if x.strip()]

                if row:
                    current_matrix.append(row)

        # Save last sample
        if current_sample is not None and current_matrix:
            samples[current_sample] = np.array(current_matrix)

        return samples

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def generate_sample_attention_weights():
    """
    Generate three synthetic attention weight matrices for a 7x7 token grid (49 tokens) for demonstration and visualization.
    
    Each returned matrix is row-normalized so each query's attention distribution sums to 1. The produced patterns are:
    - 0: Diagonal with local neighborhood attention (strong self-attention plus nearby tokens).
    - 1: Center-focused (tokens strongly attend to the central patch, with some self-attention).
    - 2: Vertical stripe (tokens attend primarily to other tokens in the same column).
    
    Returns:
        dict[int, numpy.ndarray]: Mapping from sample id (0, 1, 2) to a (49, 49) attention matrix with rows summing to 1.
    """
    n_tokens = 49  # 7x7 patches for MNIST

    # Create sample attention patterns
    sample_matrices = {}

    # Sample 0: Strong diagonal attention (token attends mostly to itself)
    diagonal_matrix = np.eye(n_tokens) * 0.5
    # Add some local attention (neighboring patches)
    for i in range(n_tokens):
        for j in range(n_tokens):
            if i != j:
                # Calculate 2D distance between patches
                i_row, i_col = i // 7, i % 7
                j_row, j_col = j // 7, j % 7
                dist = abs(i_row - j_row) + abs(i_col - j_col)
                # Nearby patches get more attention
                diagonal_matrix[i, j] = max(0.05, 0.3 / (1 + dist))
    # Normalize rows to sum to 1
    diagonal_matrix = diagonal_matrix / diagonal_matrix.sum(axis=1, keepdims=True)
    sample_matrices[0] = diagonal_matrix

    # Sample 1: Center-focused attention (all tokens attend to center)
    center_matrix = np.zeros((n_tokens, n_tokens))
    center_idx = n_tokens // 2  # Middle token (24 for 7x7)
    for i in range(n_tokens):
        center_matrix[i, center_idx] = 0.7  # Strong attention to center
        center_matrix[i, i] = 0.2  # Some self-attention
        # Distribute remaining to neighbors
        for j in range(n_tokens):
            if j != i and j != center_idx:
                center_matrix[i, j] = 0.1 / (n_tokens - 2)
    center_matrix = center_matrix / center_matrix.sum(axis=1, keepdims=True)
    sample_matrices[1] = center_matrix

    # Sample 2: Vertical stripe attention (tokens in same column attend to each other)
    stripe_matrix = np.zeros((n_tokens, n_tokens))
    for i in range(n_tokens):
        i_col = i % 7
        for j in range(n_tokens):
            j_col = j % 7
            if i_col == j_col:
                stripe_matrix[i, j] = 0.8 / 7  # Attend to same column
            else:
                stripe_matrix[i, j] = 0.2 / (n_tokens - 7)
    stripe_matrix = stripe_matrix / stripe_matrix.sum(axis=1, keepdims=True)
    sample_matrices[2] = stripe_matrix

    return sample_matrices

def visualize_attention_patterns(attention_samples, output_file=OUTPUT_FILE):
    """
    Create and save a grid of heatmaps showing attention weight matrices for up to six samples.
    
    Parameters:
        attention_samples (dict[int, numpy.ndarray]): Mapping from sample ID to a 2D attention matrix (rows = query tokens, columns = key tokens).
        output_file (str): File path where the generated figure will be saved.
    
    Returns:
        matplotlib.figure.Figure: The Matplotlib Figure containing the plotted heatmaps.
    """
    n_samples = min(len(attention_samples), 6)  # Show up to 6 samples
    sample_ids = sorted(attention_samples.keys())[:n_samples]

    # Create figure with subplots
    if n_samples == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 7))
        axes = [axes]
    elif n_samples <= 3:
        fig, axes = plt.subplots(1, n_samples, figsize=(8 * n_samples, 7))
        if n_samples == 1:
            axes = [axes]
    else:
        rows = (n_samples + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(24, 7 * rows))
        axes = axes.flatten()

    fig.suptitle('MNIST Attention Model - Attention Weight Patterns',
                 fontsize=16, fontweight='bold')

    for idx, sample_id in enumerate(sample_ids):
        ax = axes[idx]
        attention_matrix = attention_samples[sample_id]

        # Plot heatmap
        im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto',
                       interpolation='nearest', vmin=0, vmax=0.15)

        ax.set_xlabel('Key Token (attended to)', fontsize=11)
        ax.set_ylabel('Query Token (attending from)', fontsize=11)
        ax.set_title(f'Sample {sample_id}\nAttention Weights ({attention_matrix.shape[0]} tokens)',
                    fontsize=12, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)

        # Add grid for better readability
        ax.set_xticks(np.arange(0, attention_matrix.shape[1], 7))
        ax.set_yticks(np.arange(0, attention_matrix.shape[0], 7))
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Print statistics
    print("\n" + "="*60)
    print("ATTENTION WEIGHT ANALYSIS")
    print("="*60)
    for sample_id in sample_ids:
        attention_matrix = attention_samples[sample_id]
        print(f"\nSample {sample_id}:")
        print(f"  Shape: {attention_matrix.shape}")
        print(f"  Min weight: {attention_matrix.min():.6f}")
        print(f"  Max weight: {attention_matrix.max():.6f}")
        print(f"  Mean weight: {attention_matrix.mean():.6f}")

        # Check for spatial locality (diagonal dominance)
        diagonal_sum = np.trace(attention_matrix)
        print(f"  Diagonal sum (self-attention): {diagonal_sum:.4f}")

        # Check for uniform vs focused attention
        entropy = -np.sum(attention_matrix * np.log(attention_matrix + 1e-10), axis=1).mean()
        print(f"  Average entropy: {entropy:.4f} (lower = more focused)")

    print("="*60)

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")

    return fig

def visualize_single_token_attention(attention_samples, token_idx=0, output_file="token_attention.png"):
    """
    Visualize the attention distribution from a single source token as a 2D spatial heatmap.
    
    Displays up to four samples in a row, reshaping the selected token's attention row into a square grid when possible (e.g., 7x7); if the token count is not a perfect square, the attention is shown as a 1xN row.
    
    Parameters:
        attention_samples (dict[int, numpy.ndarray]): Mapping from sample ID to attention matrix (queries x keys).
        token_idx (int): Index of the source/query token whose attention distribution will be visualized.
        output_file (str): File path to save the generated visualization image (PNG).
    """
    n_samples = min(len(attention_samples), 4)
    sample_ids = sorted(attention_samples.keys())[:n_samples]

    fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 5))
    if n_samples == 1:
        axes = [axes]

    fig.suptitle(f'Attention from Token {token_idx} (spatial view)',
                 fontsize=14, fontweight='bold')

    for idx, sample_id in enumerate(sample_ids):
        ax = axes[idx]
        attention_matrix = attention_samples[sample_id]

        # Get attention weights for this token
        attention_weights = attention_matrix[token_idx, :]

        # Reshape to 2D grid (assuming 7x7 patches)
        grid_size = int(np.sqrt(len(attention_weights)))
        if grid_size * grid_size == len(attention_weights):
            attention_grid = attention_weights.reshape(grid_size, grid_size)
        else:
            # Fallback for non-square arrangements
            attention_grid = attention_weights.reshape(1, -1)

        # Plot heatmap
        im = ax.imshow(attention_grid, cmap='hot', interpolation='nearest')
        ax.set_title(f'Sample {sample_id}', fontsize=12)
        ax.set_xlabel('Patch column', fontsize=10)
        ax.set_ylabel('Patch row', fontsize=10)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Weight', rotation=270, labelpad=15)

        # Mark the source token position
        if grid_size * grid_size == len(attention_weights):
            source_row, source_col = token_idx // grid_size, token_idx % grid_size
            ax.plot(source_col, source_row, 'b*', markersize=15,
                   markeredgecolor='white', markeredgewidth=1.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Token attention visualization saved to: {output_file}")

def main():
    """
    Orchestrates loading (or generation) of attention weights and produces visualization files.
    
    Attempts to load attention weight matrices from the configured log file; if none are available, generates example attention patterns and saves them to the log. Produces heatmap visualizations for multiple samples and a spatial attention map for token 0, saving output images to disk. Prints concise progress and instructions for exporting real model attention weights.
    """
    print("MNIST Attention Weight Visualizer")
    print("="*60)

    # Try to load attention weights from file
    attention_samples = parse_attention_weights(ATTENTION_WEIGHTS_FILE)

    if attention_samples is None or len(attention_samples) == 0:
        print("\nNo attention weights found in log file.")
        print("Generating sample attention patterns for demonstration...")
        attention_samples = generate_sample_attention_weights()

        # Save sample weights to file for reference
        print(f"\nSaving sample attention weights to: {ATTENTION_WEIGHTS_FILE}")
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(ATTENTION_WEIGHTS_FILE, 'w') as f:
            f.write("# Sample Attention Weights for MNIST Attention Model\n")
            f.write("# Format: Each sample starts with '# Sample: <id>', followed by attention matrix\n")
            f.write("# Each row represents a query token's attention distribution over all key tokens\n\n")

            for sample_id, matrix in sorted(attention_samples.items()):
                f.write(f"# Sample: {sample_id}\n")
                for row in matrix:
                    f.write(" ".join([f"{w:.6f}" for w in row]) + "\n")
                f.write("\n")
        print(f"✓ Sample weights saved to: {ATTENTION_WEIGHTS_FILE}")
    else:
        print(f"\n✓ Loaded {len(attention_samples)} attention weight samples")

    # Create visualizations
    print("\nGenerating attention pattern heatmaps...")
    visualize_attention_patterns(attention_samples)

    # Create spatial attention visualization for first token
    if len(attention_samples) > 0:
        print("\nGenerating spatial attention visualization (token 0)...")
        visualize_single_token_attention(attention_samples, token_idx=0,
                                        output_file="token_0_attention.png")

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print("\nTo export real attention weights from the model:")
    print("1. Modify mnist_attention_pool.rs to save attention weights")
    print("2. Add code after softmax in forward pass:")
    print("   // Save attention weights for visualization")
    print("   save_attention_weights(&attention_weights, sample_idx);")
    print("\n3. Run the model and use this script to visualize patterns")
    print("="*60)

if __name__ == "__main__":
    main()
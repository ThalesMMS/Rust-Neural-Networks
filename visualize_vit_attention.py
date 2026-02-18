#!/usr/bin/env python3
"""
Vision Transformer Attention Map Visualization

Visualizes attention patterns from the CIFAR-10 ViT model, showing which
image patches attend to which other patches across multiple attention heads.

Usage:
    python visualize_vit_attention.py [--input PATH] [--output PATH]

Input format (logs/vit_attention_maps.csv):
    Lines starting with '# Image: N' mark a new image.
    Each data line contains a comma-separated flattened attention matrix
    of shape [num_heads * seq_len * seq_len].
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


# ViT constants (must match cifar10_vit.rs)
NUM_HEADS = 4
GRID = 8  # 8x8 patch grid
NUM_PATCHES = GRID * GRID  # 64 tokens


def parse_attention_maps(filepath):
    """Parse ViT attention maps from CSV file.

    Returns a list of dicts, one per image, each containing:
        'image_id': int
        'weights': np.ndarray of shape [num_heads, seq_len, seq_len]
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return []

    samples = []
    current_image_id = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("# Image:"):
                current_image_id = int(line.split(":")[1].strip())
                continue

            if line.startswith("#"):
                continue

            if current_image_id is None:
                continue

            # Parse comma-separated values
            values = [float(x) for x in line.split(",") if x.strip()]
            arr = np.array(values)

            # Reshape to [num_heads, seq_len, seq_len]
            expected_len = NUM_HEADS * NUM_PATCHES * NUM_PATCHES
            if len(arr) == expected_len:
                weights = arr.reshape(NUM_HEADS, NUM_PATCHES, NUM_PATCHES)
                samples.append({"image_id": current_image_id, "weights": weights})
            else:
                print(
                    f"Warning: Image {current_image_id} has {len(arr)} values, "
                    f"expected {expected_len}. Skipping."
                )

    return samples


def plot_attention_heatmaps(samples, output_file, max_images=4):
    """Plot attention heatmaps for each head across multiple images."""
    n_images = min(len(samples), max_images)
    if n_images == 0:
        print("No samples to plot.")
        return

    fig, axes = plt.subplots(
        n_images, NUM_HEADS, figsize=(4 * NUM_HEADS, 4 * n_images)
    )
    if n_images == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "ViT Attention Maps (CIFAR-10) — First Transformer Block",
        fontsize=14,
        fontweight="bold",
    )

    for img_idx in range(n_images):
        sample = samples[img_idx]
        weights = sample["weights"]  # [num_heads, seq_len, seq_len]

        for head_idx in range(NUM_HEADS):
            ax = axes[img_idx, head_idx]
            attn = weights[head_idx]  # [seq_len, seq_len]

            im = ax.imshow(attn, cmap="viridis", aspect="auto", interpolation="nearest")
            ax.set_xlabel("Key Token")
            if head_idx == 0:
                ax.set_ylabel(f"Image {sample['image_id']}\nQuery Token")
            if img_idx == 0:
                ax.set_title(f"Head {head_idx}", fontsize=11, fontweight="bold")

            # Grid lines at patch boundaries
            ax.set_xticks(np.arange(0, NUM_PATCHES, GRID))
            ax.set_yticks(np.arange(0, NUM_PATCHES, GRID))
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"Saved attention heatmaps to: {output_file}")


def plot_spatial_attention(samples, output_file, query_token=0, max_images=4):
    """Plot spatial attention overlay: reshape attention to 8x8 grid."""
    n_images = min(len(samples), max_images)
    if n_images == 0:
        return

    fig, axes = plt.subplots(
        n_images, NUM_HEADS, figsize=(4 * NUM_HEADS, 4 * n_images)
    )
    if n_images == 1:
        axes = axes[np.newaxis, :]

    query_row, query_col = query_token // GRID, query_token % GRID
    fig.suptitle(
        f"Spatial Attention from Patch ({query_row}, {query_col}) — "
        f"Token {query_token}",
        fontsize=14,
        fontweight="bold",
    )

    for img_idx in range(n_images):
        sample = samples[img_idx]
        weights = sample["weights"]

        for head_idx in range(NUM_HEADS):
            ax = axes[img_idx, head_idx]
            attn_row = weights[head_idx, query_token, :]  # [seq_len]
            attn_grid = attn_row.reshape(GRID, GRID)

            im = ax.imshow(attn_grid, cmap="hot", interpolation="nearest")

            # Mark the query patch
            ax.plot(
                query_col,
                query_row,
                "c*",
                markersize=15,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )

            if head_idx == 0:
                ax.set_ylabel(f"Image {sample['image_id']}")
            if img_idx == 0:
                ax.set_title(f"Head {head_idx}", fontsize=11, fontweight="bold")

            ax.set_xlabel("Patch Column")
            if head_idx == 0:
                ax.set_ylabel(f"Image {sample['image_id']}\nPatch Row")

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    spatial_path = output_file.replace(".png", "_spatial.png")
    plt.tight_layout()
    plt.savefig(spatial_path, dpi=200, bbox_inches="tight")
    print(f"Saved spatial attention to: {spatial_path}")


def print_statistics(samples):
    """Print attention weight statistics."""
    print("\n" + "=" * 60)
    print("ATTENTION WEIGHT STATISTICS")
    print("=" * 60)

    for sample in samples[:4]:
        weights = sample["weights"]
        print(f"\nImage {sample['image_id']}:")
        for h in range(NUM_HEADS):
            attn = weights[h]
            diag_mean = np.trace(attn) / NUM_PATCHES
            entropy = -np.sum(attn * np.log(attn + 1e-10), axis=1).mean()
            print(
                f"  Head {h}: min={attn.min():.4f}, max={attn.max():.4f}, "
                f"mean={attn.mean():.4f}, diag_mean={diag_mean:.4f}, "
                f"entropy={entropy:.4f}"
            )

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ViT attention maps from CIFAR-10"
    )
    parser.add_argument(
        "--input",
        default="logs/vit_attention_maps.csv",
        help="Path to attention maps CSV file",
    )
    parser.add_argument(
        "--output",
        default="plots/vit_attention_maps.png",
        help="Output PNG file path",
    )
    args = parser.parse_args()

    print("ViT Attention Map Visualizer")
    print("=" * 60)

    # Parse attention maps
    samples = parse_attention_maps(args.input)
    if not samples:
        print(f"\nNo attention maps found in {args.input}")
        print("Run the ViT model first:")
        print("  cargo run --release --bin cifar10_vit -- config/cifar10_vit_small.json")
        sys.exit(1)

    print(f"Loaded {len(samples)} attention map samples")

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Generate visualizations
    plot_attention_heatmaps(samples, args.output)
    plot_spatial_attention(samples, args.output)

    # Print statistics
    print_statistics(samples)

    print(f"\nVisualization complete. Output: {args.output}")


if __name__ == "__main__":
    main()

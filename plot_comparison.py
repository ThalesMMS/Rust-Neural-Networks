#!/usr/bin/env python3
"""
Training Performance Plotter
Plots training loss and time from log files
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
LOG_DIR = "./logs"
OUTPUT_FILE = "training_comparison.png"

# Log file path
LOG_FILE = f"{LOG_DIR}/training_loss_c.txt"

def load_training_data(filepath, max_epochs=10):
    """
    Load training data from CSV file.
    Format: epoch,loss,time
    Returns: epochs, losses, times
    """
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None, None, None

    try:
        data = np.loadtxt(filepath, delimiter=',')
        if data.ndim == 1:  # Single epoch
            data = data.reshape(1, -1)

        # Limit to max_epochs
        if len(data) > max_epochs:
            data = data[:max_epochs]

        epochs = data[:, 0].astype(int)
        losses = data[:, 1]
        times = data[:, 2]
        return epochs, losses, times
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None, None

def main():
    """Plot training loss and time from log file"""
    epochs, losses, times = load_training_data(LOG_FILE)
    
    if epochs is None:
        print("Error: No data loaded. Please check log file.")
        return

    cumulative_time = np.cumsum(times)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MNIST MLP Training Performance', fontsize=14, fontweight='bold')

    # Plot 1: Training Loss vs Epoch
    ax1 = axes[0, 0]
    ax1.plot(epochs, losses, marker='o', linewidth=2, color='#1f77b4', markersize=6)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Plot 2: Time per Epoch
    ax2 = axes[0, 1]
    ax2.plot(epochs, times, marker='s', linewidth=2, color='#ff7f0e', markersize=6)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Time (seconds)', fontsize=11)
    ax2.set_title('Time per Epoch', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Plot 3: Cumulative Time
    ax3 = axes[1, 0]
    ax3.plot(epochs, cumulative_time, marker='D', linewidth=2, color='#2ca02c', markersize=6)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Cumulative Time (seconds)', fontsize=11)
    ax3.set_title('Cumulative Training Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Plot 4: Summary stats as text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    total_time = np.sum(times)
    avg_time = np.mean(times)
    final_loss = losses[-1]
    
    summary_text = f"""
    Training Summary
    ────────────────────
    Total Time:      {total_time:.2f}s
    Avg Time/Epoch:  {avg_time:.2f}s
    Final Loss:      {final_loss:.6f}
    Epochs:          {len(epochs)}
    """
    ax4.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Average Time/Epoch:  {avg_time:.2f} seconds")
    print(f"Final Loss:          {final_loss:.6f}")
    print("="*50)

    # Save figure
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graph saved to: {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Hyperparameter Sweep Results Comparison Tool
Compares and visualizes results from hyperparameter sweeps
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def load_sweep_results(filepath: str) -> List[Dict[str, Any]]:
    """
    Load sweep results from JSON file.

    Args:
        filepath: Path to the sweep results JSON file

    Returns:
        List of sweep result dictionaries

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the JSON is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Sweep results file not found: {filepath}")

    try:
        with open(filepath, 'r') as f:
            results = json.load(f)

        if not isinstance(results, list):
            raise ValueError("Expected JSON file to contain a list of results")

        if len(results) == 0:
            raise ValueError("No results found in JSON file")

        return results
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in {filepath}: {e.msg}",
            e.doc,
            e.pos
        )


def validate_sweep_results(results: List[Dict[str, Any]]) -> None:
    """
    Validate that sweep results have required fields.

    Args:
        results: List of sweep result dictionaries

    Raises:
        ValueError: If required fields are missing
    """
    required_fields = [
        'config_id',
        'learning_rate',
        'batch_size',
        'epochs_completed',
        'scheduler_type',
        'final_train_loss',
        'final_val_loss',
        'final_val_accuracy',
        'total_training_time'
    ]

    for i, result in enumerate(results):
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            raise ValueError(
                f"Result {i} (config_id={result.get('config_id', 'unknown')}) "
                f"missing required fields: {', '.join(missing_fields)}"
            )


def generate_comparison_plots(results: List[Dict[str, Any]], output_path: str = None) -> str:
    """
    Generate comparison plots showing parameter effects on metrics.

    Args:
        results: List of sweep result dictionaries
        output_path: Optional path to save the plot (auto-generated if None)

    Returns:
        Path to the saved plot file
    """
    # Generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"logs/sweep_comparison_{timestamp}.png"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Extract data for plotting
    learning_rates = [r['learning_rate'] for r in results]
    batch_sizes = [r['batch_size'] for r in results]
    val_losses = [r['final_val_loss'] for r in results]
    val_accuracies = [r['final_val_accuracy'] * 100 for r in results]
    train_times = [r['total_training_time'] for r in results]
    schedulers = [r['scheduler_type'] for r in results]

    # Create figure with 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Hyperparameter Sweep Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Learning Rate vs Validation Loss
    ax1 = axes[0, 0]
    ax1.scatter(learning_rates, val_losses, s=100, alpha=0.6, color='#1f77b4', edgecolors='black', linewidth=1)
    ax1.set_xlabel('Learning Rate', fontsize=11)
    ax1.set_ylabel('Validation Loss', fontsize=11)
    ax1.set_title('Learning Rate vs Validation Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log')

    # Plot 2: Learning Rate vs Validation Accuracy
    ax2 = axes[0, 1]
    ax2.scatter(learning_rates, val_accuracies, s=100, alpha=0.6, color='#ff7f0e', edgecolors='black', linewidth=1)
    ax2.set_xlabel('Learning Rate', fontsize=11)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax2.set_title('Learning Rate vs Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log')

    # Plot 3: Batch Size vs Validation Loss
    ax3 = axes[0, 2]
    ax3.scatter(batch_sizes, val_losses, s=100, alpha=0.6, color='#2ca02c', edgecolors='black', linewidth=1)
    ax3.set_xlabel('Batch Size', fontsize=11)
    ax3.set_ylabel('Validation Loss', fontsize=11)
    ax3.set_title('Batch Size vs Validation Loss', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')

    # Plot 4: Batch Size vs Validation Accuracy
    ax4 = axes[1, 0]
    ax4.scatter(batch_sizes, val_accuracies, s=100, alpha=0.6, color='#d62728', edgecolors='black', linewidth=1)
    ax4.set_xlabel('Batch Size', fontsize=11)
    ax4.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax4.set_title('Batch Size vs Validation Accuracy', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')

    # Plot 5: Scheduler Type Comparison
    ax5 = axes[1, 1]
    unique_schedulers = sorted(set(schedulers))
    scheduler_colors = {'step_decay': '#9467bd', 'exponential': '#8c564b', 'cosine_annealing': '#e377c2'}

    for scheduler in unique_schedulers:
        indices = [i for i, s in enumerate(schedulers) if s == scheduler]
        color = scheduler_colors.get(scheduler, '#7f7f7f')
        ax5.scatter(
            [val_losses[i] for i in indices],
            [val_accuracies[i] for i in indices],
            s=100, alpha=0.6, color=color,
            label=scheduler, edgecolors='black', linewidth=1
        )

    ax5.set_xlabel('Validation Loss', fontsize=11)
    ax5.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax5.set_title('Loss vs Accuracy by Scheduler Type', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, linestyle='--')

    # Plot 6: Training Time vs Validation Accuracy
    ax6 = axes[1, 2]
    scatter = ax6.scatter(
        train_times, val_accuracies,
        s=100, alpha=0.6, c=val_losses,
        cmap='viridis_r', edgecolors='black', linewidth=1
    )
    ax6.set_xlabel('Training Time (seconds)', fontsize=11)
    ax6.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax6.set_title('Training Time vs Accuracy (colored by loss)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, linestyle='--')
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Val Loss', fontsize=10)

    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    return output_path


def print_summary(results: List[Dict[str, Any]]) -> None:
    """
    Print summary statistics about the sweep results.

    Args:
        results: List of sweep result dictionaries
    """
    print("\n" + "="*60)
    print("SWEEP RESULTS SUMMARY")
    print("="*60)
    print(f"Total configurations: {len(results)}")

    # Find best configuration by validation loss
    best_config = min(results, key=lambda x: x['final_val_loss'])
    print(f"\nBest configuration (by validation loss):")
    print(f"  Config ID:        {best_config['config_id']}")
    print(f"  Learning Rate:    {best_config['learning_rate']}")
    print(f"  Batch Size:       {best_config['batch_size']}")
    print(f"  Scheduler:        {best_config['scheduler_type']}")
    print(f"  Val Loss:         {best_config['final_val_loss']:.6f}")
    print(f"  Val Accuracy:     {best_config['final_val_accuracy']*100:.2f}%")
    print(f"  Training Time:    {best_config['total_training_time']:.2f}s")

    # Overall statistics
    avg_val_loss = sum(r['final_val_loss'] for r in results) / len(results)
    avg_val_acc = sum(r['final_val_accuracy'] for r in results) / len(results)
    avg_time = sum(r['total_training_time'] for r in results) / len(results)

    print(f"\nAverage metrics across all configurations:")
    print(f"  Validation Loss:  {avg_val_loss:.6f}")
    print(f"  Validation Acc:   {avg_val_acc*100:.2f}%")
    print(f"  Training Time:    {avg_time:.2f}s")
    print("="*60)


def print_ranked_table(results: List[Dict[str, Any]]) -> None:
    """
    Print a ranked table of top configurations by validation loss.

    Args:
        results: List of sweep result dictionaries
    """
    # Sort by validation loss (lower is better)
    sorted_results = sorted(results, key=lambda x: x['final_val_loss'])

    # Show top 5 configurations
    top_n = min(5, len(sorted_results))

    print("\n" + "="*90)
    print("TOP CONFIGURATIONS (Ranked by Validation Loss)")
    print("="*90)

    # Table header
    print(f"{'Rank':<6}{'Config ID':<15}{'LR':<12}{'Batch':<8}{'Scheduler':<18}"
          f"{'Val Loss':<12}{'Val Acc':<10}{'Time(s)':<10}")
    print("-" * 90)

    # Table rows
    for i, result in enumerate(sorted_results[:top_n], 1):
        lr = f"{result['learning_rate']:.4f}"
        batch_size = result['batch_size']
        scheduler = result['scheduler_type']
        val_loss = f"{result['final_val_loss']:.6f}"
        val_acc = f"{result['final_val_accuracy']*100:.2f}%"
        train_time = f"{result['total_training_time']:.2f}"
        config_id = result['config_id']

        print(f"{i:<6}{config_id:<15}{lr:<12}{batch_size:<8}{scheduler:<18}"
              f"{val_loss:<12}{val_acc:<10}{train_time:<10}")

    print("="*90)


def print_recommendations(results: List[Dict[str, Any]]) -> None:
    """
    Print recommendations for best configurations based on different criteria.

    Args:
        results: List of sweep result dictionaries
    """
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    # Best accuracy
    best_acc_config = max(results, key=lambda x: x['final_val_accuracy'])
    print(f"\n🏆 Best Accuracy: {best_acc_config['config_id']}")
    print(f"   Validation Accuracy:  {best_acc_config['final_val_accuracy']*100:.2f}%")
    print(f"   Learning Rate:        {best_acc_config['learning_rate']}")
    print(f"   Batch Size:           {best_acc_config['batch_size']}")
    print(f"   Scheduler:            {best_acc_config['scheduler_type']}")

    # Fastest training
    fastest_config = min(results, key=lambda x: x['total_training_time'])
    print(f"\n⚡ Fastest Training: {fastest_config['config_id']}")
    print(f"   Training Time:        {fastest_config['total_training_time']:.2f}s")
    print(f"   Validation Accuracy:  {fastest_config['final_val_accuracy']*100:.2f}%")
    print(f"   Learning Rate:        {fastest_config['learning_rate']}")
    print(f"   Batch Size:           {fastest_config['batch_size']}")

    # Best balance (lowest loss with reasonable time)
    # Define "reasonable time" as within 150% of fastest time
    time_threshold = fastest_config['total_training_time'] * 1.5
    efficient_configs = [r for r in results if r['total_training_time'] <= time_threshold]

    if efficient_configs:
        best_balance = min(efficient_configs, key=lambda x: x['final_val_loss'])
        print(f"\n⚖️  Best Balance (accuracy + speed): {best_balance['config_id']}")
        print(f"   Validation Loss:      {best_balance['final_val_loss']:.6f}")
        print(f"   Validation Accuracy:  {best_balance['final_val_accuracy']*100:.2f}%")
        print(f"   Training Time:        {best_balance['total_training_time']:.2f}s")
        print(f"   Learning Rate:        {best_balance['learning_rate']}")
        print(f"   Batch Size:           {best_balance['batch_size']}")

    print("\n" + "="*60)


def main():
    """
    Main entry point for the sweep results comparison tool.
    """
    parser = argparse.ArgumentParser(
        description='Compare and visualize hyperparameter sweep results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare results from a sweep
  python compare_sweep_results.py logs/sweep_results_20260211_123456.json

  # Display help
  python compare_sweep_results.py --help
        """
    )

    parser.add_argument(
        'results_file',
        type=str,
        help='Path to sweep results JSON file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for comparison plots (default: auto-generated)'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    try:
        # Load and validate results
        if args.verbose:
            print(f"Loading sweep results from: {args.results_file}")

        results = load_sweep_results(args.results_file)

        if args.verbose:
            print(f"Loaded {len(results)} configurations")
            print("Validating results...")

        validate_sweep_results(results)

        if args.verbose:
            print("Validation successful")

        # Print summary
        print_summary(results)

        # Print ranked table
        print_ranked_table(results)

        # Print recommendations
        print_recommendations(results)

        # Generate comparison plots
        if args.verbose:
            print("\nGenerating comparison plots...")

        plot_path = generate_comparison_plots(results, args.output)

        print(f"\n✓ Comparison plots saved to: {plot_path}")
        print("\n✓ Sweep results analyzed successfully")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/bin/bash
# Generate Architecture Comparison Dashboard data
#
# This script trains all three MNIST models (MLP, CNN, Attention) to produce
# CSV training logs in logs/, then generates the dashboard data JSON file.
#
# Usage:
#   ./scripts/generate_dashboard.sh              # Train all models and generate data
#   ./scripts/generate_dashboard.sh --data-only  # Skip training, only generate data
#
# Output:
#   - logs/training_loss_*.txt (CSV training logs)
#   - demo/dashboard_data.json (dashboard data)

set -e

# show_usage prints the script usage, available options (including --data-only), examples, and the expected output files.
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Generate Architecture Comparison Dashboard data by training MNIST models"
    echo "and creating dashboard JSON from their training logs."
    echo ""
    echo "Options:"
    echo "  --help        Show this help message and exit"
    echo "  --data-only   Skip model training, only generate dashboard data"
    echo "                (requires existing training logs in logs/)"
    echo ""
    echo "Examples:"
    echo "  $0              # Train all models and generate dashboard data"
    echo "  $0 --data-only  # Generate data from existing training logs"
    echo ""
    echo "Output:"
    echo "  - logs/training_loss_*.txt (CSV training logs)"
    echo "  - demo/dashboard_data.json (dashboard data)"
    echo ""
}

# Change to project root directory
cd "$(dirname "$0")/.."

# Parse command line arguments
DATA_ONLY=false
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_usage
    exit 0
elif [ "$1" = "--data-only" ]; then
    DATA_ONLY=true
fi

# Train all three MNIST models to generate CSV logs
if [ "$DATA_ONLY" = false ]; then
    echo "========================================="
    echo "Training MNIST Models for Dashboard"
    echo "========================================="
    echo ""
    echo "This will train three models in sequence:"
    echo "  1. MNIST MLP (784 -> 512 -> 10)"
    echo "  2. MNIST CNN (Conv2D + MaxPool + Dense)"
    echo "  3. MNIST Attention Pool (Transformer-style)"
    echo ""
    echo "Note: Training may take several minutes."
    echo ""

    echo "[1/3] Training MNIST MLP..."
    cargo run --release --bin mnist_mlp
    echo ""
    echo "✓ MLP training complete"
    echo ""

    echo "[2/3] Training MNIST CNN..."
    cargo run --release --bin mnist_cnn
    echo ""
    echo "✓ CNN training complete"
    echo ""

    echo "[3/3] Training MNIST Attention Pool..."
    cargo run --release --bin mnist_attention_pool
    echo ""
    echo "✓ Attention training complete"
    echo ""
else
    echo "Skipping model training (--data-only flag specified)"
    echo ""
fi

# Generate dashboard data from training logs
echo "========================================="
echo "Generating Dashboard Data"
echo "========================================="
echo ""
echo "Building and running dashboard data generator..."
cargo run --release --bin generate_dashboard_data

echo ""
echo "========================================="
echo "Dashboard Generation Complete!"
echo "========================================="
echo ""
echo "Dashboard data generated at: demo/dashboard_data.json"
echo "To view: open demo/architecture_dashboard.html in a browser"
echo ""
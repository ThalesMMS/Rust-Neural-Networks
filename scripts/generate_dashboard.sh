#!/bin/bash
# Generate Architecture Comparison Dashboard data
#
# Prerequisites: Train the models first to produce CSV logs in logs/:
#   cargo run --release --bin mnist_mlp
#   cargo run --release --bin mnist_cnn
#   cargo run --release --bin mnist_attention_pool
#
# Then run this script to generate demo/dashboard_data.json.

set -e

cd "$(dirname "$0")/.."

echo "Building and running dashboard data generator..."
cargo run --release --bin generate_dashboard_data

echo ""
echo "Dashboard data generated at demo/dashboard_data.json"
echo "Open demo/architecture_dashboard.html in a browser."

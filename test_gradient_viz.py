#!/usr/bin/env python3
"""Test gradient visualization logic without matplotlib"""

import numpy as np

def load_gradient_data(filepath):
    """Simplified loader for testing"""
    layers_data = {}
    
    with open(filepath, 'r') as f:
        header = f.readline().strip()
        if not header.startswith('epoch'):
            return None
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) != 4:
                continue
            
            epoch = int(parts[0])
            layer_name = parts[1].strip()
            grad_norm_weights = float(parts[2])
            grad_norm_biases = float(parts[3])
            
            if layer_name not in layers_data:
                layers_data[layer_name] = {
                    'epochs': [],
                    'grad_weights': [],
                    'grad_biases': []
                }
            
            layers_data[layer_name]['epochs'].append(epoch)
            layers_data[layer_name]['grad_weights'].append(grad_norm_weights)
            layers_data[layer_name]['grad_biases'].append(grad_norm_biases)
    
    for layer_name in layers_data:
        layers_data[layer_name]['epochs'] = np.array(layers_data[layer_name]['epochs'])
        layers_data[layer_name]['grad_weights'] = np.array(layers_data[layer_name]['grad_weights'])
        layers_data[layer_name]['grad_biases'] = np.array(layers_data[layer_name]['grad_biases'])
    
    return layers_data

# Test loading
data = load_gradient_data('./logs/gradients_mlp.csv')

if data:
    print("✓ Data loaded successfully")
    print(f"  Layers found: {sorted(data.keys())}")
    for layer in sorted(data.keys()):
        epochs = data[layer]['epochs']
        weights = data[layer]['grad_weights']
        biases = data[layer]['grad_biases']
        print(f"\n  {layer}:")
        print(f"    Epochs: {len(epochs)} ({epochs[0]} to {epochs[-1]})")
        print(f"    Weight gradients: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
        print(f"    Bias gradients: min={biases.min():.4f}, max={biases.max():.4f}, mean={biases.mean():.4f}")
    print("\n✓ All gradient data verified")
    print("✓ Ready for visualization (matplotlib required)")
else:
    print("✗ Failed to load data")
    exit(1)

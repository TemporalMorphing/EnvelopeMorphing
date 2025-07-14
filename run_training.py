#!/usr/bin/env python3
"""
Script to run training for the Simple Audio Morphing Model
"""

import os
import torch
from twin_network import train_model

def main():
    # Configuration parameters
    config = {
        # Data paths - UPDATE THESE TO YOUR ACTUAL PATHS
        'csv_path': 'data/morphing_dataset/train_simple.csv',
        'model_config': 'stable_audio_tools/ckpts/model_config.json',
        'ckpt_path': 'stable_audio_tools/ckpts/model_unwrap_10k_full_train.ckpt',
        
        # Output directory
        'output_dir': './simple_morph_model_output',
        
        # Training parameters
        'num_epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'embedding_dim': 64,
        
        # Device selection
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Print configuration
    print("=" * 60)
    print("SIMPLE AUDIO MORPHING MODEL TRAINING")
    print("=" * 60)
    print(f"CSV Path: {config['csv_path']}")
    print(f"Model Config: {config['model_config']}")
    print(f"Checkpoint Path: {config['ckpt_path']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"Device: {config['device']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Embedding Dim: {config['embedding_dim']}")
    print("=" * 60)
    
    # Check if paths exist
    required_paths = [config['csv_path'], config['model_config'], config['ckpt_path']]
    for path in required_paths:
        if not os.path.exists(path):
            print(f"ERROR: Required file not found: {path}")
            print("Please update the paths in this script to point to your actual files.")
            return
    
    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)
    
    try:
        # Start training
        print("Starting training...")
        model = train_model(
            csv_path=config['csv_path'],
            model_config_path=config['model_config'],
            ckpt_path=config['ckpt_path'],
            output_dir=config['output_dir'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            device=config['device'],
            embedding_dim=config['embedding_dim']
        )
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {config['output_dir']}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
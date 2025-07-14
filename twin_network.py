import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torchaudio
import json
import os
from pathlib import Path
from tqdm import tqdm

from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.autoencoders import AudioAutoencoder
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.utils.torch_common import copy_state_dict
from stable_audio_tools.data.modification import Mono, Stereo

# Simplified Audio Morphing Model
class SimpleAudioMorphModel(nn.Module):
    def __init__(self, embedding_dim=64):
        super(SimpleAudioMorphModel, self).__init__()
        
        # Input dimension: 3 * embedding_dim (for x1+x2, abs(x1-x2), alpha*x1+(1-alpha)*x2)
        input_dim = 3 * embedding_dim
        
        # Simple 3-layer network as specified
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
    
    def forward(self, x1, x2, alpha):
        batch_size = x1.size(0)
        
        # Compute the three feature vectors
        sum_feats = x1 + x2  # 64 dim
        diff_feats = torch.abs(x1 - x2)  # 64 dim
        
        # Weighted combination
        alpha_expanded = alpha.view(batch_size, 1).expand(-1, x1.size(1))
        weighted_sum = alpha_expanded * x1 + (1 - alpha_expanded) * x2  # 64 dim
        
        # Concatenate all features (192 dim total)
        combined = torch.cat([sum_feats, diff_feats, weighted_sum], dim=1)
        
        # Pass through network
        output = self.network(combined)
        
        return output

# Dataset class (same as original)
class AudioTripletDataset(Dataset):
    def __init__(self, csv_path, model_config_path, ckpt_path):
        self.df = pd.read_csv(csv_path)
        
        # Load the pre-trained encoder model
        with open(model_config_path) as f:
            model_config = json.load(f)
        
        self.encoder = create_model_from_config(model_config)
        copy_state_dict(self.encoder, load_ckpt_state_dict(ckpt_path))
        self.encoder.eval()
        
        # Determine input channel requirements
        self.in_ch = self.encoder.in_channels
        self.preprocess = Mono() if self.in_ch == 1 else Stereo()
        self.sr = self.encoder.sample_rate
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get file paths
        x1_path = '/data/' + row['env1_filepath']
        x2_path = '/data/' + row['env2_filepath']
        x3_path = '/data/' + row['env3_filepath']

        # Get alpha value (default to 0.5 if not provided)
        alpha = row.get('alpha', 0.5)
        
        try:
            # Extract embeddings
            x1_embedding = self._get_embedding(x1_path)
            x2_embedding = self._get_embedding(x2_path)
            x3_embedding = self._get_embedding(x3_path)
            
            return {
                'x1': x1_embedding,
                'x2': x2_embedding,
                'x3': x3_embedding,
                'alpha': torch.tensor(alpha, dtype=torch.float32)
            }
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return dummy sample to avoid breaking training
            dummy_emb = torch.zeros(64)
            return {
                'x1': dummy_emb,
                'x2': dummy_emb,
                'x3': dummy_emb,
                'alpha': torch.tensor(0.5, dtype=torch.float32)
            }
    
    def _get_embedding(self, audio_path):
        # Load audio file
        audio, in_sr = self._load_audio(audio_path)
        
        # Preprocess audio
        audio = self.preprocess(audio)
        audio = audio.unsqueeze(0)  # Add batch dimension
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.encoder.encode(audio)
            
        # Return embedding as a flattened vector
        return embedding.reshape(-1)
    
    def _load_audio(self, filename):
        if not filename.lower().endswith('.wav'):
            raise ValueError(f"Unsupported file format for {filename}. Only '.wav' files are supported.")
        audio, in_sr = torchaudio.load(filename)
        
        # Handle sample rate mismatch if needed
        if in_sr != self.sr:
            resample_tf = torchaudio.transforms.Resample(in_sr, self.sr)
            audio = resample_tf(audio)
            
        return audio, self.sr

# Simplified training function
def train_model(
    csv_path,
    model_config_path,
    ckpt_path,
    output_dir,
    num_epochs=100,
    batch_size=32,
    learning_rate=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    embedding_dim=64
):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset and dataloader
    print("Loading dataset...")
    dataset = AudioTripletDataset(csv_path, model_config_path, ckpt_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Create model
    model = SimpleAudioMorphModel(embedding_dim=embedding_dim).to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Cosine similarity for evaluation
    cos = nn.CosineSimilarity(dim=1)
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_cos_sim = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get embeddings and alpha
            x1 = batch['x1'].to(device, non_blocking=True)
            x2 = batch['x2'].to(device, non_blocking=True)
            x3 = batch['x3'].to(device, non_blocking=True)
            alpha = batch['alpha'].to(device, non_blocking=True)
            
            # Forward pass
            morphed = model(x1, x2, alpha)
            
            # Calculate loss
            loss = criterion(morphed, x3)
            
            # Calculate cosine similarity for monitoring
            similarity = cos(morphed, x3).mean().item()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            running_cos_sim += similarity
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cos_sim': f"{similarity:.4f}"
            })
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(dataloader)
        epoch_cos_sim = running_cos_sim / len(dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, "
              f"Cosine Similarity: {epoch_cos_sim:.4f}")
        
        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f"{output_dir}/model_epoch_{epoch+1}.pt")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, f"{output_dir}/final_model.pt")
    
    print(f"Training complete. Final model saved to {output_dir}/final_model.pt")
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Simple Audio Morphing Model")
    parser.add_argument('--csv_path', type=str, required=True, 
                       help="Path to the CSV file with triplet file paths and alpha values")
    parser.add_argument('--model_config', type=str, required=True, 
                       help="Path to the encoder model config file")
    parser.add_argument('--ckpt_path', type=str, required=True, 
                       help="Path to the encoder checkpoint file")
    parser.add_argument('--output_dir', type=str, default="./twin_network", 
                       help="Directory to save the model")
    parser.add_argument('--num_epochs', type=int, default=100, 
                       help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, 
                       help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help="Learning rate")
    parser.add_argument('--embedding_dim', type=int, default=64,
                       help="Embedding dimension")
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        csv_path=args.csv_path,
        model_config_path=args.model_config,
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim
    )
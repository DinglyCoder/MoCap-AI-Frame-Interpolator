import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import time
import json
from dataloader import recover_global_motion
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataloader import CMUMotionDataset
from visualization import *

class MotionAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for Motion Data as described in the paper
    "Learning Motion Manifolds with Convolutional Autoencoders"
    """
    def __init__(self, input_dim=117):
        super(MotionAutoencoder, self).__init__()
        
        # Encoder network with 3 convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=15, padding=7, stride=2),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=15, padding=7, stride=2),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 256, kernel_size=15, padding=7, stride=2),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2)
        )
        
        # The decoder needs to upsample to restore the original dimensions
        self.decoder = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='linear'),
            nn.ConvTranspose1d(256, 128, kernel_size=15, padding=7, stride=2, output_padding=1),
            nn.ReLU(),

            # nn.Upsample(scale_factor=2, mode='linear'),
            nn.ConvTranspose1d(128, 64, kernel_size=15, padding=7, stride=2, output_padding=1),
            nn.ReLU(),

            # nn.Upsample(scale_factor=2, mode='linear'),
            nn.ConvTranspose1d(64, input_dim, kernel_size=15, padding=7, stride=2, output_padding=1),
            nn.Tanh(),
        )
    
    def encode(self, x):
        """Project onto the manifold (Φ operation)"""
        return self.encoder(x)
    
    def decode(self, z):
        """Inverse projection from the manifold (Φ† operation)"""
        return self.decoder(z)
    
    def forward(self, x, corrupt_input=False, corruption_prob=0.1):
        """Forward pass with optional denoising"""
        if corrupt_input and self.training:
            # Create corruption mask (randomly set values to zero with probability corruption_prob)
            mask = torch.bernoulli(torch.ones_like(x) * (1 - corruption_prob))
            x = x * mask
            
        # Forward pass for the autoencoder.
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


class MotionManifoldTrainer:
    """Trainer for the Motion Manifold Convolutional Autoencoder"""
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
        epochs: int = 25,
        fine_tune_epochs: int = 25,
        learning_rate: float = 0.5,
        fine_tune_lr: float = 0.01,
        sparsity_weight: float = 0.01,
        window_size: int = 160,
        val_split: float = 0.1,
        device: str = None
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir if cache_dir else os.path.join(data_dir, "cache")
        self.batch_size = batch_size
        self.epochs = epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.learning_rate = learning_rate
        self.fine_tune_lr = fine_tune_lr
        self.sparsity_weight = sparsity_weight
        self.window_size = window_size
        self.val_split = val_split
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        
        # Set device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        # Load dataset
        self._load_dataset()
        
        # Initialize model
        self._init_model()
        
    def _load_dataset(self):
        """Load the CMU Motion dataset and create training/validation splits"""
        # Create dataset
        from dataloader import CMUMotionDataset
        
        self.dataset = CMUMotionDataset(
            data_dir=self.data_dir,
            cache_dir=self.cache_dir,
            frame_rate=30,
            window_size=self.window_size,
            overlap=0.5,
            include_velocity=True,
            include_foot_contact=True
        )
        
        # Split into training and validation sets
        val_size = int(self.val_split * len(self.dataset))
        train_size = len(self.dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4
        )
        
        print(f"Dataset loaded with {len(self.dataset)} windows from {len(self.dataset.motion_data)} files")
        print(f"Training samples: {train_size}, Validation samples: {val_size}")
        
        # Get mean and std for normalization
        self.mean_pose = torch.tensor(self.dataset.get_mean_pose(), device=self.device, dtype=torch.float32)
        self.std = torch.tensor(self.dataset.get_std(), device=self.device, dtype=torch.float32)
        self.joint_names = self.dataset.get_joint_names()
        self.joint_parents = self.dataset.get_joint_parents()
        
    def _init_model(self):
        """Initialize the motion autoencoder model"""
        # Get a sample to determine dimensions
        sample = self.dataset[0]
        
        # Get the flattened positions with velocities for the paper's approach
        # We'll use positions_flat which has global transforms removed
        if "positions_flat" in sample:
            positions_flat = sample["positions_flat"]
            
            # Check if we need to add velocities to the input 
            # The paper mentions including rotational velocity around Y and translational velocity in XZ
            if "trans_vel_xz" in sample and "rot_vel_y" in sample:
                # Get velocity data
                trans_vel_xz = sample["trans_vel_xz"]
                rot_vel_y = sample["rot_vel_y"]
                
                # Create input with features as separate channels (matches paper description)
                input_dim = positions_flat.shape[1] + trans_vel_xz.shape[1] + 1  # positions + trans_vel_xz + rot_vel_y
                print(f"Input includes positions ({positions_flat.shape[1]} dims) and velocities ({trans_vel_xz.shape[1] + 1} dims!!!)")
            else:
                # Just use positions if velocities aren't available
                input_dim = positions_flat.shape[1]
                print(f"Input only includes positions ({input_dim} dims)")
        else:
            # Fallback to original positions if flattened positions aren't available
            positions = sample["positions"]
            # Calculate input dimension from sample
            # For the paper's approach, we need to flatten joints and dimensions
            # positions is [time, joints, 3], we need to get joints*3
            input_dim = positions.shape[1] * positions.shape[2]
            print(f"Using fallback input dimension: {input_dim}")
        
        # Create model
        self.model = MotionAutoencoder(input_dim=input_dim).to(self.device)
        print(f"Created model with input dimension: {input_dim}")
        
    def train(self):
        """Train the motion autoencoder in two phases: initial training and fine-tuning"""
        # Training phases for the motion autoencoder.
        initial_stats = self._train_phase(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            corruption_prob=0.1,  # Add noise during initial training for robustness
            sparsity_weight=0.0,  # No sparsity constraint in initial phase
            phase_name="initial"
        )

        # Phase 2: Fine-tuning with lower learning rate and sparsity constraint
        fine_tune_stats = self._train_phase(
            epochs=self.fine_tune_epochs,
            learning_rate=self.fine_tune_lr,
            corruption_prob=0.0,  # No corruption during fine-tuning
            sparsity_weight=self.sparsity_weight,  # Add sparsity constraint
            phase_name="fine_tune"
        )
        
        # Combine statistics: you can return stats here as a dictionary and use our plotting function to plot the training curves.
        all_stats = {"initial_training": initial_stats, "fine_tuning": fine_tune_stats}
        
        # Save training statistics
        with open(os.path.join(self.output_dir, "training_stats.json"), "w") as f:
            json.dump(all_stats, f, indent=2)
            
        # Save final model
        self._save_model()
            
        # Save normalization parameters
        self._save_normalization_params()
        
        # Plot training curves
        self._plot_training_curves(all_stats)
        
        return all_stats

    # This is a sample of what you can use in the training phase. You are not required to follow it as long as you can provide the training statistics we required.
    def _train_phase(self, epochs, learning_rate, corruption_prob, sparsity_weight, phase_name):
        """Train the model for a specific phase (initial training or fine-tuning)"""
        print(f"\n===== {phase_name.capitalize()} Training Phase! =====")
        
        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        reconstruction_criterion = nn.MSELoss()
        
        # Training stats
        stats = {
            "train_loss": [],
            "val_loss": [],
        }
        
        # Track training checkpoints
        best_val_loss = float("inf")
        
        # Train for specified epochs
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for batch in progress_bar:
                # Training loop for the motion autoencoder.
                x_pos = batch["positions_normalized_flat"].to(self.device)                
                x_vel = batch["trans_vel_normalized"].to(self.device)                  
                x_rot = batch["rot_vel_normalized"].to(self.device).unsqueeze(-1)
                x = torch.cat([x_pos, x_vel, x_rot], dim=-1).permute(0,2,1)

                # Forward pass with optional corruption
                optimizer.zero_grad()
                x_recon = self.model(x, corrupt_input=True, corruption_prob=corruption_prob)

                # print("x mean:", x.mean().item(), "std:", x.std().item())
                # print("x_recon mean:", x_recon.mean().item(), "std:", x_recon.std().item())

                # Compute reconstruction loss
                recon_loss = reconstruction_criterion(x_recon, x)

                # Add sparsity constraint if in fine-tuning phase
                if sparsity_weight > 0:
                    # Get the latent representation
                    latent = self.model.encode(x)
                    # L1 regularization on the latent representation
                    l1_loss = torch.mean(torch.abs(latent))
                    loss = recon_loss + sparsity_weight * l1_loss
                else:
                    loss = recon_loss

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.6f}",
                })
            stats["train_loss"].append(train_loss / len(self.train_loader))


            # Validation, skip validation eval for first 10 epochs
            if epoch >= 0:
                self.model.eval()
                val_loss = 0
                
                with torch.no_grad():       # No grad since we are in eval mode
                    progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                    for batch in progress_bar:
                        # validation loop for the motion autoencoder.
                        x_pos = batch["positions_normalized_flat"].to(self.device)                
                        x_vel = batch["trans_vel_normalized"].to(self.device)                  
                        x_rot = batch["rot_vel_normalized"].to(self.device).unsqueeze(-1)
                        x = torch.cat([x_pos, x_vel, x_rot], dim=-1).permute(0,2,1)

                        # Forward pass without corruption
                        x_recon = self.model(x, corrupt_input=False)

                        # Compute validation loss
                        loss = reconstruction_criterion(x_recon, x)
                        val_loss += loss.item()

                        # Update progress bar
                        progress_bar.set_postfix({"val_loss": f"{val_loss:.6f}"})

                avg_val_loss = val_loss / len(self.val_loader)
                stats["val_loss"].append(avg_val_loss)
                                         
                if epoch % 5 == 0:
                    # Save if best model. You can also save checkpoints each epoch if needed.
                    if avg_val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self._save_checkpoint(epoch, end=f'_valloss_{val_loss:.6f}', phase_name=phase_name)
                        print(f"  Saved checkpoint with val_loss: {val_loss:.6f}")
        
        return stats
    
    def _save_checkpoint(self, epoch, end, phase_name):
        """Save a model checkpoint"""
        checkpoint_path = os.path.join(
            self.output_dir, "checkpoints", f"{phase_name}_epoch_{epoch+1}{end}.pt"
        )
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
        }, checkpoint_path)
    
    def _save_model(self):
        """Save the trained model"""
        model_path = os.path.join(self.output_dir, "models", "motion_autoencoder.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def _save_normalization_params(self):
        """Save normalization parameters for inference if needed"""
        norm_data = {
            "mean_pose": self.mean_pose.cpu().numpy(),
            "std": self.std.cpu().numpy(),
            "joint_names": self.joint_names,
            "joint_parents": self.joint_parents
        }
        
        np.save(os.path.join(self.output_dir, "normalization.npy"), norm_data)
        print(f"Normalization parameters saved to {self.output_dir}/normalization.npy")
    
    def _plot_training_curves(self, stats):
        """Plot training curves for one or more training phases"""
        if not isinstance(stats[list(stats.keys())[0]], dict):
            stats = {"train": stats}
            
        n_p = len(list(stats.keys()))
        plt.figure(figsize=(12, 4 * n_p))
        # Multiple training phases
        for i, (phase_name, phase_stats) in enumerate(stats.items()):
            plt.subplot(n_p, 1, i+1)
            for key, values in phase_stats.items():
                if values is None:
                    continue  # Skip None entries like sparsity_loss
                plt.plot(values, label=key)
            plt.title(f"{phase_name.capitalize()} Training Phase")
            plt.xlabel("Epoch")
            plt.ylabel("Statistics")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "plots", "training_curves.png"))
        plt.close()
        
        print(f"Training curves saved to {self.output_dir}/plots/training_curves.png")


class MotionManifoldSynthesizer:
    """Synthesizer for generating, fixing, and analyzing motion using the learned manifold"""
    def __init__(
        self,
        model_path: str,
        dataset: CMUMotionDataset,
        device: str = None
    ):
        # Set device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {self.device}")
        
        # Load normalization parameters
        self._load_normalization(dataset)
        
        # Load model
        self._load_model(model_path)
    
    def _load_normalization(self, dataset: CMUMotionDataset):
        """Load normalization parameters from dataset"""
        self.mean_pose = torch.tensor(dataset.mean_pose, device=self.device, dtype=torch.float32)
        self.std = torch.tensor(dataset.std, device=self.device, dtype=torch.float32)
        self.joint_names = dataset.joint_names
        self.joint_parents = dataset.joint_parents

        self.mean_trans_vel = torch.tensor(dataset.mean_trans_vel, device=self.device, dtype=torch.float32)
        self.std_trans_vel = torch.tensor(dataset.std_trans_vel, device=self.device, dtype=torch.float32)
        self.mean_rot_vel = torch.tensor(dataset.mean_rot_vel, device=self.device, dtype=torch.float32)
        self.std_rot_vel = torch.tensor(dataset.std_rot_vel, device=self.device, dtype=torch.float32)

    
    def _load_model(self, model_path):
        """Load trained model"""
        if os.path.exists(model_path):
            # Determine input dimension from the model's saved state
            model_state = torch.load(model_path, map_location=self.device)
            
            # Try to infer input dimension from the first layer weights
            first_layer_weight = None
            for key in model_state.keys():
                if 'encoder.0.weight' in key:
                    first_layer_weight = model_state[key]
                    break
            
            if first_layer_weight is not None:
                input_dim = first_layer_weight.shape[1]
                print(f"Inferred input dimension {input_dim} from model weights")
            else:
                # Fallback to calculating from mean_pose if we can't find the weights
                input_dim = self.mean_pose.shape[0] * self.mean_pose.shape[1]
                print(f"Using fallback input dimension: {input_dim}")
                
            # Create model
            self.model = MotionAutoencoder(input_dim=input_dim).to(self.device)
            
            # Load weights
            self.model.load_state_dict(model_state)
            
            # Set to evaluation mode
            self.model.eval()
            
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
    
    def fix_corrupted_motion(self, motion, corruption_type='zero', corruption_params=None):
        """
        Fix corrupted motion by projecting onto the manifold and recovering global motion
        
        Args:
            motion: tensor of shape [batch_size, time_steps, joints, dims]
            corruption_type: Type of corruption to apply ('zero', 'noise', or 'missing')
            corruption_params: Parameters for corruption
                    
        Returns:
            Tuple of (corrupted_motion, fixed_motion)
        """
        positions = motion['positions'].to(self.device)
        trans_vel = motion['trans_vel_xz'].to(self.device)
        rot_vel = motion['rot_vel_y'].to(self.device)
        # Store original shape
        original_shape = positions.shape
        batch_size, time_steps, joints, dims = original_shape
        
        # Apply corruption if not already corrupted
        if corruption_params is not None:
            corrupted_motion = self._apply_corruption(positions, corruption_type, corruption_params)
        else:
            corrupted_motion = positions.clone()
            
        # Fix the corrupted motion by the model.
        corrupted_norm = (corrupted_motion - self.mean_pose) / (self.std + 1e-8)
        corrupted_norm_flat = corrupted_norm.view(batch_size, time_steps, -1)

        trans_vel_norm = (trans_vel - self.mean_trans_vel) / (self.std_trans_vel + 1e-8)
        rot_vel_norm = ((rot_vel - self.mean_rot_vel) / (self.std_rot_vel + 1e-8)).unsqueeze(-1)

        input_features = torch.cat([corrupted_norm_flat, trans_vel_norm, rot_vel_norm], dim=-1)
        input_features = input_features.permute(0, 2, 1)  # [B, C, T]

        # Step 5: Pass through autoencoder
        with torch.no_grad():
            latent = self.model.encode(input_features)
            recon = self.model.decode(latent)

        # revert to original shape
        recon = recon.permute(0, 2, 1)  # [B, T, C]
        pos_dim = joints * dims
        recon_pos_flat = recon[:, :, :pos_dim]
        recon_positions_norm = recon_pos_flat.view(batch_size, time_steps, joints, dims)

        # unnormalize
        recon_positions = recon_positions_norm * (self.std + 1e-8) + self.mean_pose

        fixed_motion = recover_global_motion(recon_positions, trans_vel, rot_vel)
        
        # Return corrupted motion and fixed motion with global transform applied
        return corrupted_motion, fixed_motion
    
    def _apply_corruption(self, motion, corruption_type, params):
        """Apply corruption to motion data"""
        corrupted = motion.clone()
        
        if corruption_type == 'zero':
            # Randomly set values to zero
            prob = params.get('prob', 0.5)
            mask = torch.bernoulli(torch.ones_like(corrupted) * (1 - prob))
            corrupted = corrupted * mask
            
        elif corruption_type == 'noise':
            # Add Gaussian noise
            noise_scale = params.get('scale', 0.1)
            noise = torch.randn_like(corrupted) * noise_scale
            corrupted = corrupted + noise
            
        elif corruption_type == 'missing':
            # Set specific joint to zero
            joint_idx = params.get('joint_idx', 0)
            corrupted[:, :, joint_idx, :] = 0.0
            
        return corrupted
    
    def interpolate_motions(self, positions1, trans_vel1, rot_vel1, positions2, trans_vel2, rot_vel2, t):
        """
        Interpolate between two motions on the manifold, handling global transforms
        
        Args:
            positions1, positions2: [1, T, J*3] (normalized + flattened)
            trans_vel1, trans_vel2: [1, T, 2] (normalized)
            rot_vel1, rot_vel2: [1, T]        (normalized)
            t: Interpolation parameter (0 to 1)
                    
        Returns:
            Interpolated motion as tensor of shape [batch_size, time_steps, joints, dims]
        """
        # Implement motion interpolation on the manifold.
        # Concatenate features: [1, T, D]

        rot_vel1 = rot_vel1.unsqueeze(-1)  # [1, T, 1]
        rot_vel2 = rot_vel2.unsqueeze(-1)
        motion1 = torch.cat([positions1, trans_vel1, rot_vel1], dim=-1)  # [1, T, D]
        motion2 = torch.cat([positions2, trans_vel2, rot_vel2], dim=-1)

        # Permute to [1, D, T] for model
        motion1 = motion1.permute(0, 2, 1)
        motion2 = motion2.permute(0, 2, 1)

        # Encode → Interpolate → Decode
        z1 = self.model.encode(motion1)
        z2 = self.model.encode(motion2)
        z_interp = (1 - t) * z1 + t * z2
        x_interp = self.model.decode(z_interp)  # [1, D, T]

        x_interp = x_interp.permute(0, 2, 1)

        # Split input data back into positions and velocities
        pos_dim = positions1.shape[2]
        trans_dim = trans_vel1.shape[2]

        pos_interp = x_interp[:, :, :pos_dim]               # [1, T, J*3]
        trans_interp = x_interp[:, :, pos_dim:pos_dim+trans_dim]  # [1, T, 2]
        rot_interp = x_interp[:, :, -1]

        pos_interp = pos_interp * self.std.view(1, 1, -1) + self.mean_pose.view(1, 1, -1)

        # Reshape positions: [1, T, J*3] -> [1, T, J, 3]
        B, T, flat_dim = pos_interp.shape
        pos_interp = pos_interp.view(B, T, -1, 3)

        return {
            "positions": pos_interp,        # [1, T, J, 3]
            "trans_vel": trans_interp,      # [1, T, 2]
            "rot_vel": rot_interp           # [1, T]
        }
    
    # You can add more functions for Extra Credit.
    def prepare_inputs(self, motion):
        x_pos = motion["positions_normalized_flat"].to(self.device)                
        x_vel = motion["trans_vel_normalized"].to(self.device)                  
        x_rot = motion["rot_vel_normalized"].to(self.device).unsqueeze(-1)
        x = torch.cat([x_pos, x_vel, x_rot], dim=-1).permute(0,2,1)

        return x
    
    def advanced_motion_synthesis(self, motion, gap_type='middle', gap_params=None):
        '''
        Filling in or extending motion sequences

        Args:
            motion: dict containing 'positions', 'trans_vel_xz', and 'rot_vel_y'
            gap_type: 'middle' for adding in missing frames within a squence
                        'end' for extending the end of a sequence
            gap_params: parameters for gap creation
        '''
        
        corrupted_pos = motion['positions'].clone().to(self.device)
        corrupted_pos_norm = motion['positions_normalized_flat'].clone().to(self.device)
        corrupted_rot_norm = motion['rot_vel_normalized'].clone().to(self.device).unsqueeze(-1)
        corrupted_trans_norm = motion['trans_vel_normalized'].clone().to(self.device)   

        batch_size, time_steps, joints, dims = motion['positions'].shape

        if gap_type == 'middle':
            # Create a gap in the middle of the sequence
            gap_start = gap_params.get('start', 70)
            gap_end = gap_params.get('end', 90)
            corrupted_pos[:, gap_start:gap_end, :, :] = 0.0
            model_input = self.prepare_inputs(motion)
        elif gap_type == 'end':
            # Create a gap at the end of the sequence
            gap_length = gap_params.get('length', 60)
            corrupted_pos[:, -gap_length:, :, :] = 0.0
            corrupted_pos_norm[:, -gap_length:, :, :] = 0.0
            corrupted_trans_norm[:, -gap_length:, :] = 0.0
            corrupted_rot_norm[:, -gap_length:, :] = 0.0
            model_input = torch.cat([corrupted_pos_norm, corrupted_trans_norm, corrupted_rot_norm], dim=-1).permute(0,2,1)

        # model_input = self.prepare_inputs(corrupted_inputs)

        # Pass through the model
        with torch.no_grad():
            latent = self.model.encode(model_input)
            recon = self.model.decode(latent)

        # revert to original shape
        recon = recon.permute(0, 2, 1)  # [B, T, C]
        pos_dim = joints * dims
        recon_pos_flat = recon[:, :, :pos_dim]
        recon_positions_norm = recon_pos_flat.view(batch_size, time_steps, joints, dims)

        # unnormalize
        recon_positions = recon_positions_norm * (self.std + 1e-8) + self.mean_pose

        # add reconstruction to corrupted motion'
        if gap_type == 'middle':
            completed = corrupted_pos.clone()
            completed[:, gap_start:gap_end, :, :] = recon_positions[:, gap_start:gap_end, :, :]
        elif gap_type == 'end':
            completed = torch.cat([motion['positions'].to(self.device), recon_positions[:, -gap_length:, :, :]], dim=1)
        
        return completed
    
def main():
    """Example usage of the motion manifold training"""
    
    # Training parameters
    data_dir = "path/to/cmu-mocap"
    output_dir = "./output/ae"
    
    trainer = MotionManifoldTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=32,
        epochs=25,              # Initial training epochs
        fine_tune_epochs=25,    # Fine-tuning epochs
        learning_rate=0.001,    # Initial learning rate
        fine_tune_lr=0.001,     # Fine-tuning learning rate
        sparsity_weight=0.01,   # Sparsity constraint weight
        window_size=160,        # Window size (as in paper)
        val_split=0.1           # Validation split
    )
    
    # Train the model
    trainer.train()
    
    # For inference, you can load the dataset and model and use the synthesizer for different tasks. 
    # You can also use the visualization functions to visualize the results following examples in dataloader.py.


if __name__ == "__main__":
    main()
    
    
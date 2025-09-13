import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import pickle
import tqdm
from typing import List, Tuple, Dict, Optional, Union


def recover_global_motion(local_motion, trans_vel_xz, rot_vel_y, frame_rate=30.0):
    """
    Recover global motion by integrating velocities over time.
    
    Args:
        local_motion: Tensor of shape [batch_size, time_steps, joints, dims] with local positions
        trans_vel_xz: Tensor of shape [batch_size, time_steps, 2] with XZ translational velocities
        rot_vel_y: Tensor of shape [batch_size, time_steps] with Y rotational velocities
        frame_rate: Frames per second
    
    Returns:
        Tensor of shape [batch_size, time_steps, joints, dims] with global positions
    """
    batch_size, time_steps, num_joints, dims = local_motion.shape
    global_motion = local_motion.clone()
    
    # Process each batch separately
    for b in range(batch_size):
        # Initialize global transform variables
        global_trans_xz = torch.zeros((time_steps, 2), device=local_motion.device)
        global_rot_y = torch.zeros(time_steps, device=local_motion.device)
        
        # Integrate velocities to recover global transform
        # Start with initial values (typically zero)
        for t in range(1, time_steps):
            # Integrate translational velocity (XZ plane)
            delta_trans = trans_vel_xz[b, t-1] / frame_rate  # Convert velocity to displacement per frame
            global_trans_xz[t] = global_trans_xz[t-1] + delta_trans
            
            # Integrate rotational velocity (Y axis)
            delta_rot = rot_vel_y[b, t-1] / frame_rate  # Convert velocity to rotation per frame
            global_rot_y[t] = global_rot_y[t-1] + delta_rot
        
        # Apply global transforms to local positions
        for t in range(time_steps):
            # 1. Apply Y-axis rotation to all joints
            angle = global_rot_y[t]
            cos_angle = torch.cos(angle)
            sin_angle = torch.sin(angle)
            
            for j in range(num_joints):
                x = local_motion[b, t, j, 0].clone()
                z = local_motion[b, t, j, 2].clone()
                global_motion[b, t, j, 0] = cos_angle * x + sin_angle * z
                global_motion[b, t, j, 2] = -sin_angle * x + cos_angle * z
            
            # 2. Apply XZ translation to all joints
            global_motion[b, t, :, 0] += global_trans_xz[t, 0]  # Add X translation
            global_motion[b, t, :, 2] += global_trans_xz[t, 1]  # Add Z translation
    
    return global_motion

class CMUMotionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        cache_dir: str = None,
        frame_rate: int = 60,
        window_size: int = 240,
        overlap: float = 0.5,
        joint_selection: Optional[List[str]] = None,
        include_velocity: bool = True,
        include_foot_contact: bool = True,
        force_recompute: bool = False
    ):
        """
        Complete dataloader for CMU motion dataset with caching.
        
        Args:
            data_dir: Directory containing BVH files (will search recursively)
            cache_dir: Directory to store preprocessed data (if None, uses data_dir/cache)
            frame_rate: Target frame rate of the motion
            window_size: Number of frames in each window
            overlap: Overlap ratio between consecutive windows
            joint_selection: List of joint names to include (None = all)
            include_velocity: Whether to compute velocity features
            include_foot_contact: Whether to detect foot contacts
            force_recompute: Force recomputation of cached data
        """
        self.data_dir = os.path.join(data_dir, "data")
        self.cache_dir = cache_dir if cache_dir else os.path.join(data_dir, "cache")
        self.frame_rate = frame_rate
        self.window_size = window_size
        self.overlap = overlap
        self.joint_selection = joint_selection
        self.include_velocity = include_velocity
        self.include_foot_contact = include_foot_contact
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Find all BVH files recursively
        self.file_list = glob.glob(os.path.join(data_dir, "**/*.bvh"), recursive=True)
        self.file_list.sort()
        print(f"Found {len(self.file_list)} BVH files in {data_dir}")
        
        # Create a cache identifier based on parameters
        self.cache_id = self._create_cache_id()
        
        # Load or compute the dataset
        self._load_or_compute_dataset(force_recompute)
        
    def _create_cache_id(self) -> str:
        """Create a unique identifier for the cache based on parameters"""
        # Create a string representation of the parameters
        param_str = f"fr{self.frame_rate}_ws{self.window_size}_ol{self.overlap}"
        param_str += f"_vel{int(self.include_velocity)}_fc{int(self.include_foot_contact)}"
        
        if self.joint_selection:
            param_str += f"_js{'_'.join(self.joint_selection)}"
            
        return f"{param_str}"
    
    def _compute_statistics(self, motion_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and standard deviation of (which) joint positions"""
        
        # TODO: Compute the normalization statistics on the necessary part of motion data.
        # Save the mean and std to self.mean and self.std.
        
        # self.mean_pose = np.zeros_like(motion_data[list(motion_data.keys())[0]]['positions'][0])
        # self.std = np.zeros_like(motion_data[list(motion_data.keys())[0]]['positions'][0])

        # sum all values
        all_positions = []
        all_rot_vel = []
        all_trans_vel = []
        for i in motion_data:
            all_positions.append(motion_data[i]['local_positions'])
            all_trans_vel.append(motion_data[i]['trans_vel_xz'])
            all_rot_vel.append(motion_data[i]['rot_vel_xz'])

        all_positions = np.concatenate(all_positions, axis=0) 
        all_trans_vel = np.concatenate(all_trans_vel, axis=0)    
        all_rot_vel = np.concatenate(all_rot_vel, axis=0) 

        # calculate means and stds
        self.mean_pose = np.mean(all_positions, axis=0)           
        self.std = np.std(all_positions, axis=0)        

        self.mean_trans_vel = np.mean(all_trans_vel, axis=0)        
        self.std_trans_vel = np.std(all_trans_vel, axis=0) 

        self.mean_rot_vel = np.mean(all_rot_vel)            
        self.std_rot_vel = np.std(all_rot_vel)   


        # You can also save a copy of the mean and std for later use.
        # stats = {
        #     "mean_pose": self.mean_pose,
        #     "std": self.std,
        #     "mean_trans_vel": self.mean_trans_vel,
        #     "std_trans_vel": self.std_trans_vel,
        #     "mean_rot_vel": self.mean_rot_vel,
        #     "std_rot_vel": self.std_rot_vel
        # }
        # with open(os.path.join(self.cache_dir, f"stats_{self.cache_id}.pkl"), 'wb') as f:
        #     pickle.dump(stats, f)

        return self.mean_pose, self.std, self.mean_trans_vel, self.std_trans_vel, self.mean_rot_vel, self.std_rot_vel
    
    def _load_or_compute_dataset(self, force_recompute: bool = False) -> None:
        """Load dataset from cache if available, otherwise compute and cache it"""
        cache_file = os.path.join(self.cache_dir, f"dataset_{self.cache_id}.pkl")
        
        if os.path.exists(cache_file) and not force_recompute:
            print(f"Loading preprocessed data from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            self.windows = cache_data['windows']
            self.motion_data = cache_data['motion_data']
            self.joint_names = cache_data['joint_names']
            self.joint_parents = cache_data['joint_parents']
            
            print(f"Loaded {len(self.windows)} windows")
        else:
            print("Computing dataset and caching results...")
            self._preprocess_dataset()
            
            # Save to cache
            cache_data = {
                'windows': self.windows,
                'motion_data': self.motion_data,
                'joint_names': self.joint_names,
                'joint_parents': self.joint_parents,
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            print(f"Saved {len(self.windows)} windows to cache: {cache_file}")
        
        cache_staticstics = os.path.join(self.cache_dir, f"stats_{self.cache_id}.pkl")
        if os.path.exists(cache_staticstics):
            with open(cache_staticstics, 'rb') as f:
                stats = pickle.load(f)
                self.mean_pose = stats["mean_pose"]
                self.std = stats["std"]
                self.mean_trans_vel = stats["mean_trans_vel"]
                self.std_trans_vel = stats["std_trans_vel"]
                self.mean_rot_vel = stats["mean_rot_vel"]
                self.std_rot_vel = stats["std_rot_vel"]
        else:
            self._compute_statistics(self.motion_data)
        
    def _preprocess_dataset(self) -> None:
        """Process all BVH files and create windows"""
        self.motion_data = {}
        all_positions = []
        all_local_positions = []
        
        # Process each BVH file
        for file_path in tqdm.tqdm(self.file_list, desc="Processing BVH files"):
            try:
                # Use BVHReader from read_bvh.py to load the data
                from read_bvh import BVHReader
                reader = BVHReader(file_path)
                reader.read()
                
                # Get joint positions using the reader
                joint_positions = reader.get_joint_positions_batch()
                num_frames = joint_positions.shape[0]
                
                # Skip files that are too short
                if num_frames < self.window_size:
                    continue
                
                root_translation = reader.root_translation.numpy()
                root_rotation = reader.root_rotation.numpy()
                
                # Compute additional features
                joint_names = reader.joint_names
                joint_parents = reader.joint_parents.tolist()
                
                # Get joint offsets as fixed positions in T-pose
                joint_offsets = reader.joint_offsets.numpy()
                
                # Convert to desired frame rate if needed
                if abs(1.0 / reader.frame_time - self.frame_rate) > 1e-5:
                    original_fps = 1.0 / reader.frame_time
                    joint_positions = self._resample_positions(joint_positions.numpy(), original_fps)
                    root_rotation = self._resample_positions(root_rotation, original_fps)
                    root_translation = self._resample_positions(root_translation, original_fps)
                else:
                    joint_positions = joint_positions.numpy()
                
                root_rotation = np.deg2rad(root_rotation)
                root_rotation_y = root_rotation[..., 1]
                
                num_frames = joint_positions.shape[0]
                
                # Process motion data to ensure correct Y-up orientation
                # Check if coordinate system seems to be different than expected
                foot_heights = self._get_foot_heights(joint_positions, joint_names)
                if np.mean(foot_heights) > 1.0:  # If feet are very high, coordinate system may be wrong
                    print(f"Warning: Unusual foot heights detected in {file_path}. Check coordinate system.")
                
                local_positions = self._remove_global_transforms(joint_positions, root_translation, root_rotation_y)
                
                # Calculate velocities from actual root motion
                trans_vel_xz = np.zeros((num_frames, 2))
                rot_vel_y = np.zeros(num_frames)
                
                if num_frames > 1:
                    # XZ translational velocity (assuming root motion channels are in world space)
                    trans_vel_xz[1:, 0] = root_translation[1:, 0] - root_translation[:-1, 0]  # X velocity
                    trans_vel_xz[1:, 1] = root_translation[1:, 2] - root_translation[:-1, 2]  # Z velocity
                    
                    # Y rotational velocity (handle angle wrapping)
                    angle_diff = root_rotation_y[1:] - root_rotation_y[:-1]
                    # Normalize to [-π, π]
                    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                    rot_vel_y[1:] = angle_diff
                    
                    # Convert to velocity per second
                    trans_vel_xz *= self.frame_rate
                    rot_vel_y *= self.frame_rate
                    
                # Store the processed data
                bvh_data = {
                    "positions": joint_positions,
                    "local_positions": local_positions,
                    "joint_names": joint_names,
                    "joint_parents": joint_parents,
                    "joint_offsets": joint_offsets,
                    "num_frames": num_frames,
                    "root_translation": root_translation,
                    "root_rotation": root_rotation,
                    'trans_vel_xz': trans_vel_xz,
                    'rot_vel_y': rot_vel_y
                }
                
                # Calculate velocities if needed
                if self.include_velocity:
                    velocities = self._calculate_velocities(joint_positions)
                    bvh_data["velocities"] = velocities
                    
                # Calculate foot contacts if needed
                if self.include_foot_contact:
                    foot_contacts = self._detect_foot_contacts(joint_positions, joint_names)
                    bvh_data["foot_contacts"] = foot_contacts
                
                file_name = file_path.split('/')[-2:]
                file_name = '/'.join(file_name)
                self.motion_data[file_path] = bvh_data
                all_positions.append(joint_positions)
                all_local_positions.append(local_positions)
                
                # Store joint structure from first file (assuming uniform skeleton)
                if not hasattr(self, 'joint_names'):
                    self.joint_names = joint_names
                    self.joint_parents = joint_parents
                    
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        if not all_local_positions:
            raise ValueError("No valid motion files found. Check your data directory.")
        
        # Create windows
        self.windows = self._create_windows()
        
        print(f"Processed {len(self.motion_data)} files with {len(self.windows)} valid windows")
        
    def _resample_positions(self, positions, original_fps):
        """Resample motion data to target frame rate"""
        num_frames = positions.shape[0]
        
        # Calculate new number of frames
        duration = num_frames / original_fps
        new_num_frames = int(duration * self.frame_rate)
        
        # Create new positions array
        new_positions = np.zeros((new_num_frames, *positions.shape[1:]))
        
        # Interpolate each frame
        for i in range(new_num_frames):
            # Calculate source frame index (can be fractional)
            src_idx = i * original_fps / self.frame_rate
            
            # Linear interpolation between adjacent frames
            frame1 = int(np.floor(src_idx))
            frame2 = min(frame1 + 1, num_frames - 1)
            blend = src_idx - frame1
            
            new_positions[i] = positions[frame1] * (1 - blend) + positions[frame2] * blend
            
        return new_positions
    
    def _create_windows(self) -> List[Tuple[str, int]]:
        """Create all windows from the processed motion data, handling shorter sequences"""
        windows = []
        
        for file_path, motion_data in self.motion_data.items():
            num_frames = motion_data["positions"].shape[0]
            
            # Skip files that are too short for even a single window
            if num_frames < self.window_size:
                continue
                
            # Create windows with specified overlap
            stride = int(self.window_size * (1 - self.overlap))
            
            # Create windows that fit completely within the sequence
            for start_frame in range(0, num_frames - self.window_size + 1, stride):
                windows.append((file_path, start_frame))
                    
        return windows
        
    def _calculate_velocities(self, positions):
        """Calculate joint velocities from positions"""
        num_frames = positions.shape[0]
        velocities = np.zeros_like(positions)
        
        if num_frames <= 1:
            # Cannot calculate velocity for single frame, return zeros
            return velocities
        
        # Calculate velocities for frames with predecessor
        velocities[1:] = (positions[1:] - positions[:-1]) * self.frame_rate
        
        # For first frame, use the same velocity as second frame
        velocities[0] = velocities[1]
        
        return velocities
    
    def _get_foot_heights(self, positions, joint_names):
        """Get the average height of feet joints for detecting coordinate system issues"""
        # Find foot joint indices
        left_foot_idx = 4
        right_foot_idx = 10
        
        # If specific joints not found, use alternatives
        if left_foot_idx is None and right_foot_idx is None:
            # If no foot joints found, try to find the lowest joints in the hierarchy
            foot_candidates = [i for i, name in enumerate(joint_names) if 'end' in name.lower()]
            if foot_candidates:
                # Take average height of the lowest 2-4 joints
                foot_heights = np.mean(positions[:, foot_candidates, 1])
                return foot_heights
            return 0.0  # No appropriate joints found
        
        # Calculate average foot heights
        foot_heights = []
        if left_foot_idx is not None:
            foot_heights.append(np.mean(positions[:, left_foot_idx, 1]))
        if right_foot_idx is not None:
            foot_heights.append(np.mean(positions[:, right_foot_idx, 1]))
            
        return np.mean(foot_heights)

    def _detect_foot_contacts(self, positions, joint_names):
        """Detect foot contacts by height and velocity thresholds"""
        num_frames = positions.shape[0]
        
        # Try to find specific foot joints
        left_foot_idx = 4
        right_foot_idx = 10
        left_toe_idx = 5
        right_toe_idx = 11
        
        # Initialize contact array
        contacts = np.zeros((num_frames, 4))  # left heel, left toe, right heel, right toe
        
        # Get minimum height of feet across all frames to determine the "ground" level
        min_left_foot = np.min(positions[:, left_foot_idx, 1])
        min_right_foot = np.min(positions[:, right_foot_idx, 1])
        min_left_toe = np.min(positions[:, left_toe_idx, 1])
        min_right_toe = np.min(positions[:, right_toe_idx, 1])
        ground_level = min(min_left_foot, min_right_foot, min_left_toe, min_right_toe)
        
        # Threshold parameters (adaptive to the data)
        height_threshold = ground_level + 0.05  # 5cm above ground
        velocity_threshold = 0.15  # Meters per second
        
        # Get foot heights (using Y coordinate for Y-up data)
        left_foot_height = positions[:, left_foot_idx, 1]  # Y component
        right_foot_height = positions[:, right_foot_idx, 1]
        left_toe_height = positions[:, left_toe_idx, 1]
        right_toe_height = positions[:, right_toe_idx, 1]
        
        # Calculate vertical velocities
        left_foot_vel = np.zeros_like(left_foot_height)
        right_foot_vel = np.zeros_like(right_foot_height)
        left_toe_vel = np.zeros_like(left_toe_height)
        right_toe_vel = np.zeros_like(right_toe_height)
        
        left_foot_vel[1:] = (left_foot_height[1:] - left_foot_height[:-1]) * self.frame_rate
        right_foot_vel[1:] = (right_foot_height[1:] - right_foot_height[:-1]) * self.frame_rate
        left_toe_vel[1:] = (left_toe_height[1:] - left_toe_height[:-1]) * self.frame_rate
        right_toe_vel[1:] = (right_toe_height[1:] - right_toe_height[:-1]) * self.frame_rate
        
        # Detect contacts
        contacts[:, 0] = (left_foot_height < height_threshold) & (abs(left_foot_vel) < velocity_threshold)
        contacts[:, 1] = (left_toe_height < height_threshold) & (abs(left_toe_vel) < velocity_threshold)
        contacts[:, 2] = (right_foot_height < height_threshold) & (abs(right_foot_vel) < velocity_threshold)
        contacts[:, 3] = (right_toe_height < height_threshold) & (abs(right_toe_vel) < velocity_threshold)
        
        # Convert to -1/1 format as specified in the paper
        contacts = contacts * 2 - 1
        
        return contacts
    
    def _remove_global_transforms(self, positions, root_translation, root_rotation_y):
        """
        Remove global transforms using direct root motion from BVH
        
        Args:
            positions: [frames, joints, 3] joint positions
            root_translation: [frames, 3] root translation (XYZ)
            root_rotation_y: [frames] root rotation around Y axis in radians
            
        Returns:
            local_positions: Positions with global transforms removed
        """
        frames, num_joints, dims = positions.shape
        local_positions = positions.copy()
        # 1. Remove global translation in XZ plane
        local_positions[..., 0] -= root_translation[:, None, 0]
        local_positions[..., 2] -= root_translation[:, None, 2]
        
        # 2. Remove global rotation around Y axis
        for f in range(frames):
            # Get rotation angle
            angle = root_rotation_y[f]
            
            # Apply inverse rotation (negative angle)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            # Apply to all joints
            for j in range(num_joints):
                x = local_positions[f, j, 0]
                z = local_positions[f, j, 2]
                local_positions[f, j, 0] = cos_angle * x - sin_angle * z
                local_positions[f, j, 2] = sin_angle * x + cos_angle * z
        
        return local_positions
    
    def __len__(self):
        """Return the number of windows in the dataset"""
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Get a window of motion data with preprocessing applied consistently.
        Returns normalized local motion along with metadata needed for global reconstruction.
        """
        file_path, start_frame = self.windows[idx]
        motion_data = self.motion_data[file_path]
        
        end_frame = start_frame + self.window_size
        
        # 1. Get pre-computed local positions (global transforms already removed)
        local_positions = motion_data["local_positions"][start_frame:end_frame].copy()
        
        # Align original positions to the same window and align displacement
        original_positions = motion_data["positions"][start_frame:end_frame].copy()
        original_positions[:, :, 0] -= original_positions[:1, :1, 0]
        original_positions[:, :, 2] -= original_positions[:1, :1, 2]
        
        # 2. Get global motion parameters for reconstruction
        trans_vel_xz = motion_data["trans_vel_xz"][start_frame:end_frame].copy()
        rot_vel_y = motion_data["rot_vel_y"][start_frame:end_frame].copy()
        
        # 3. TODO: Apply normalization to necessary part of the motion data to compute positions_normalized.
        positions_normalized = (local_positions - self.mean_pose) / (self.std + 1e-8)
        trans_vel_normalized = (trans_vel_xz - self.mean_trans_vel) / (self.std_trans_vel + 1e-8)
        rot_vel_normalized = (rot_vel_y - self.mean_rot_vel) / (self.std_rot_vel + 1e-8)
        
        # 4. Create flat versions for the model
        # Reshape: [time, joints, 3] -> [time, joints*3]
        time_steps, joints, dims = local_positions.shape
        positions_flat = local_positions.reshape(time_steps, -1)
        positions_normalized_flat = positions_normalized.reshape(time_steps, -1)
        
        # 5. Create result dictionary
        result = {
            "positions": torch.tensor(local_positions, dtype=torch.float32),
            "positions_normalized": torch.tensor(positions_normalized, dtype=torch.float32),
            "positions_flat": torch.tensor(positions_flat, dtype=torch.float32),
            "positions_normalized_flat": torch.tensor(positions_normalized_flat, dtype=torch.float32),
            "trans_vel_xz": torch.tensor(trans_vel_xz, dtype=torch.float32),
            "rot_vel_y": torch.tensor(rot_vel_y, dtype=torch.float32),
            "trans_vel_normalized": torch.tensor(trans_vel_normalized, dtype=torch.float32),
            "rot_vel_normalized": torch.tensor(rot_vel_normalized, dtype=torch.float32),
            "root_positions": torch.tensor(motion_data["positions"][start_frame:end_frame, 0, :], dtype=torch.float32),
            "original_positions": torch.tensor(original_positions, dtype=torch.float32),
        }
        
        # 6. Add additional data if available
        # Add foot contacts if available
        if self.include_foot_contact and "foot_contacts" in motion_data:
            foot_contacts = motion_data["foot_contacts"][start_frame:end_frame]
            result["foot_contacts"] = torch.tensor(foot_contacts, dtype=torch.float32)
        
        # Add velocities if available
        if self.include_velocity and "velocities" in motion_data:
            velocities = motion_data["velocities"][start_frame:end_frame]
            result["velocities"] = torch.tensor(velocities, dtype=torch.float32)
        
        return result
    
    def get_mean_pose(self):
        """Return the mean pose of the dataset"""
        return self.mean_pose
    
    def get_std(self):
        """Return the standard deviation of the dataset"""
        return self.std
    
    def get_joint_names(self):
        """Return the joint names of the skeleton"""
        return self.joint_names
    
    def get_joint_parents(self):
        """Return the parent indices of each joint"""
        return self.joint_parents

# Example usage
if __name__ == "__main__":
    # Create dataset
    data_dir = "./cmu-mocap/"
    output_dir = "./cmu-mocap/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = CMUMotionDataset(
        data_dir=data_dir,
        frame_rate=30,
        window_size=160,
        overlap=0.5
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # Print dataset info
    print(f"Dataset size: {len(dataset)}")
    print(f"Joint names: {dataset.get_joint_names()}")
    
    # Iterate through batches
    for batch_idx, batch in enumerate(dataloader):
        positions = batch["positions"]
        
        print(f"Batch {batch_idx}: Positions shape: {positions.shape}")
        
        # Check if other features are available
        if "foot_contacts" in batch:
            foot_contacts = batch["foot_contacts"]
            print(f"Foot contacts shape: {foot_contacts.shape}")
        
        # Just process a few batches for demonstration
        if batch_idx >= 2:
            break
        
    from visualization import visualize_motion_to_video
    # Visualize multiple samples for testing your output

    # for i in range(3):
    #     sample = dataset[497 + i]
    #     positions = sample["positions"]
    #     output_file = os.path.join(output_dir, f"sample_{497 + i}.mp4")
    #     visualize_motion_to_video(positions, dataset.joint_parents, output_file)
    
    for i in range(3):
        sample = dataset[i]
        positions = sample["positions"]
        output_file = os.path.join(output_dir, f"sample_{i}.mp4")
        visualize_motion_to_video(positions, dataset.joint_parents, output_file)
        positions_normalized = sample["positions_normalized"]
        output_file = os.path.join(output_dir, f"sample_{i}_normalized.mp4")
        visualize_motion_to_video(positions_normalized, dataset.joint_parents, output_file)
        recovered = recover_global_motion(positions.unsqueeze(0), sample['trans_vel_xz'].unsqueeze(0), sample['rot_vel_y'].unsqueeze(0))[0]
        output_file = os.path.join(output_dir, f"sample_{i}_recovered.mp4")
        visualize_motion_to_video(recovered, dataset.joint_parents, output_file)     
        
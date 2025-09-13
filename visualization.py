import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from matplotlib.animation import FFMpegWriter

def visualize_motion_to_video(motion_tensor, joint_parents, output_path, fps=30, view_elevation=20, view_azimuth=45):
    """
    Visualize motion data as a 3D animation and save to MP4 video.
    
    Args:
        motion_tensor: Tensor of shape [batch, frames, joints, 3] containing joint positions
        joint_parents: Dictionary mapping joint indices to parent joint indices
        output_path: Path to save the MP4 video
        fps: Frames per second for the output video
        view_elevation: Elevation angle for the 3D view
        view_azimuth: Azimuth angle for the 3D view
    """
    # Convert to numpy if it's a PyTorch tensor
    if isinstance(motion_tensor, torch.Tensor):
        motion_data = motion_tensor.detach().cpu().numpy()
    else:
        motion_data = motion_tensor
    
    # Remove batch dimension if it's present
    if len(motion_data.shape) == 4 and motion_data.shape[0] == 1:
        motion_data = motion_data[0]
    
    # Get dimensions
    num_frames, num_joints, _ = motion_data.shape
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set some margin around the motion
    min_vals = np.min(motion_data, axis=(0, 1))
    max_vals = np.max(motion_data, axis=(0, 1))
    center = (min_vals + max_vals) / 2
    max_range = np.max(max_vals - min_vals) / 2 + 0.5  # Add some margin
    
    # Create line objects for bones (one per parent-child joint pair)
    lines = []
    for joint_idx in range(num_joints):
        parent_idx = joint_parents[joint_idx]
        parent_idx = parent_idx if parent_idx >= 0 else None
        # parent_idx = joint_parents.get(joint_idx)
        if parent_idx is not None:  # Skip root joint which has no parent
            line, = ax.plot([], [], [], 'b-', linewidth=2)
            lines.append((joint_idx, parent_idx, line))
    
    # Create scatter objects for joints
    joints_scatter = ax.scatter([], [], [], c='r', s=30)
    
    # Set fixed axes limits for consistent view
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Z')  # Swap Y and Z for better visualization
    ax.set_zlabel('Y')
    
    # Set the view angle
    ax.view_init(elev=view_elevation, azim=view_azimuth)
    
    # Set title
    ax.set_title('Motion Visualization')
    
    # Initialize plots with the first frame
    def init():
        for _, _, line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        
        joints_scatter._offsets3d = ([], [], [])
        return [line for _, _, line in lines] + [joints_scatter]
    
    # Update function for each frame
    def update(frame):
        # Update title with frame number
        ax.set_title(f'Motion Visualization (Frame {frame+1}/{num_frames})')
        
        # Update joint positions
        joint_positions = motion_data[frame]
        
        # Update lines (bones)
        for joint_idx, parent_idx, line in lines:
            joint_pos = joint_positions[joint_idx]
            parent_pos = joint_positions[parent_idx]
            line.set_data([parent_pos[0], joint_pos[0]], [parent_pos[2], joint_pos[2]])  # Swap Y and Z
            line.set_3d_properties([parent_pos[1], joint_pos[1]])
        
        # Update joints
        joints_scatter._offsets3d = (
            joint_positions[:, 0],  # X
            joint_positions[:, 2],  # Z (swapped with Y)
            joint_positions[:, 1]   # Y
        )
        
        return [line for _, _, line in lines] + [joints_scatter]
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, blit=True, interval=1000/fps
    )
    
    # Save animation to file
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=5000)
    ani.save(output_path, writer=writer)
    
    plt.close(fig)
    print(f"Animation saved to {output_path}")
    
    # Return path to the saved animation
    return output_path


def visualize_motion_comparison(original_tensor, fixed_tensor, joint_parents, output_path, fps=30):
    """
    Visualize two motions side by side for comparison.
    
    Args:
        original_tensor: Original motion tensor of shape [batch, frames, joints, 3]
        fixed_tensor: Fixed motion tensor of shape [batch, frames, joints, 3]
        joint_parents: Dictionary mapping joint indices to parent joint indices
        output_path: Path to save the MP4 video
        fps: Frames per second for the output video
    """
    # Convert to numpy if they're PyTorch tensors
    if isinstance(original_tensor, torch.Tensor):
        original_data = original_tensor.detach().cpu().numpy()
    else:
        original_data = original_tensor
        
    if isinstance(fixed_tensor, torch.Tensor):
        fixed_data = fixed_tensor.detach().cpu().numpy()
    else:
        fixed_data = fixed_tensor
    
    # Remove batch dimension if present
    if len(original_data.shape) == 4 and original_data.shape[0] == 1:
        original_data = original_data[0]
    if len(fixed_data.shape) == 4 and fixed_data.shape[0] == 1:
        fixed_data = fixed_data[0]
    
    # Get dimensions
    num_frames = max(original_data.shape[0], fixed_data.shape[0])
    num_joints = original_data.shape[1]
    
    # Create figure with two 3D subplots
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Set titles
    ax1.set_title('Original Motion')
    ax2.set_title('Fixed Motion')
    
    # Calculate limits for consistent scale
    all_data = np.concatenate([original_data[:num_frames], fixed_data[:num_frames]], axis=0)
    min_vals = np.min(all_data, axis=(0, 1))
    max_vals = np.max(all_data, axis=(0, 1))
    center = (min_vals + max_vals) / 2
    max_range = np.max(max_vals - min_vals) / 2 + 0.5  # Add margin
    
    # Set fixed axes limits
    for ax in [ax1, ax2]:
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')  # Swap Y and Z
        ax.set_zlabel('Y')
        
        # Set the view angle
        ax.view_init(elev=20, azim=45)
    
    # Create line objects for bones
    lines1 = []
    lines2 = []
    
    for joint_idx in range(num_joints):
        parent_idx = joint_parents[joint_idx]
        parent_idx = parent_idx if parent_idx >= 0 else None
        # parent_idx = joint_parents.get(joint_idx)
        if parent_idx is not None:
            line1, = ax1.plot([], [], [], 'b-', linewidth=2)
            line2, = ax2.plot([], [], [], 'g-', linewidth=2)
            lines1.append((joint_idx, parent_idx, line1))
            lines2.append((joint_idx, parent_idx, line2))
    
    # Create scatter objects for joints
    joints_scatter1 = ax1.scatter([], [], [], c='r', s=30)
    joints_scatter2 = ax2.scatter([], [], [], c='r', s=30)
    
    # Initialize plots
    def init():
        for _, _, line in lines1 + lines2:
            line.set_data([], [])
            line.set_3d_properties([])
        
        joints_scatter1._offsets3d = ([], [], [])
        joints_scatter2._offsets3d = ([], [], [])
        return [line for _, _, line in lines1 + lines2] + [joints_scatter1, joints_scatter2]
    
    # Update function for each frame
    def update(frame):
        # Update frame titles
        ax1.set_title(f'Original Motion (Frame {frame+1}/{num_frames})')
        ax2.set_title(f'Fixed Motion (Frame {frame+1}/{num_frames})')
        
        # Get joint positions for current frame
        original_positions = original_data[frame]
        fixed_positions = fixed_data[frame]
        
        # Update original motion
        for joint_idx, parent_idx, line in lines1:
            joint_pos = original_positions[joint_idx]
            parent_pos = original_positions[parent_idx]
            line.set_data([parent_pos[0], joint_pos[0]], [parent_pos[2], joint_pos[2]])
            line.set_3d_properties([parent_pos[1], joint_pos[1]])
        
        joints_scatter1._offsets3d = (
            original_positions[:, 0],
            original_positions[:, 2],
            original_positions[:, 1]
        )
        
        # Update fixed motion
        for joint_idx, parent_idx, line in lines2:
            joint_pos = fixed_positions[joint_idx]
            parent_pos = fixed_positions[parent_idx]
            line.set_data([parent_pos[0], joint_pos[0]], [parent_pos[2], joint_pos[2]])
            line.set_3d_properties([parent_pos[1], joint_pos[1]])
        
        joints_scatter2._offsets3d = (
            fixed_positions[:, 0],
            fixed_positions[:, 2],
            fixed_positions[:, 1]
        )
        
        return [line for _, _, line in lines1 + lines2] + [joints_scatter1, joints_scatter2]
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, blit=True, interval=1000/fps
    )
    
    # Save animation
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=5000)
    ani.save(output_path, writer=writer)
    
    plt.close(fig)
    print(f"Comparison animation saved to {output_path}")
    
    return output_path


def visualize_interpolation(motion1, motion2, interpolated_motion, joint_parents, output_path, fps=30):
    """
    Visualize interpolation between two motions.
    
    Args:
        motion1: First motion tensor of shape [batch, frames, joints, 3]
        motion2: Second motion tensor of shape [batch, frames, joints, 3]
        interpolated_motion: Interpolated motion tensor of shape [batch, frames, joints, 3]
        joint_parents: Dictionary mapping joint indices to parent joint indices
        output_path: Path to save the MP4 video
        fps: Frames per second for the output video
    """
    # Convert to numpy if they're PyTorch tensors
    if isinstance(motion1, torch.Tensor):
        motion1_data = motion1.detach().cpu().numpy()
    else:
        motion1_data = motion1
        
    if isinstance(motion2, torch.Tensor):
        motion2_data = motion2.detach().cpu().numpy()
    else:
        motion2_data = motion2
        
    if isinstance(interpolated_motion, torch.Tensor):
        interp_data = interpolated_motion.detach().cpu().numpy()
    else:
        interp_data = interpolated_motion
    
    # Remove batch dimension if present
    if len(motion1_data.shape) == 4 and motion1_data.shape[0] == 1:
        motion1_data = motion1_data[0]
    if len(motion2_data.shape) == 4 and motion2_data.shape[0] == 1:
        motion2_data = motion2_data[0]
    if len(interp_data.shape) == 4 and interp_data.shape[0] == 1:
        interp_data = interp_data[0]
    
    # Get dimensions
    num_frames = min(motion1_data.shape[0], motion2_data.shape[0], interp_data.shape[0])
    num_joints = motion1_data.shape[1]
    
    # Create figure with three 3D subplots
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Set titles
    ax1.set_title('Motion 1')
    ax2.set_title('Interpolated Motion')
    ax3.set_title('Motion 2')
    
    # Calculate limits for consistent scale
    all_data = np.concatenate([motion1_data[:num_frames], 
                              motion2_data[:num_frames], 
                              interp_data[:num_frames]], axis=0)
    min_vals = np.min(all_data, axis=(0, 1))
    max_vals = np.max(all_data, axis=(0, 1))
    center = (min_vals + max_vals) / 2
    max_range = np.max(max_vals - min_vals) / 2 + 0.5  # Add margin
    
    # Set fixed axes limits
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')  # Swap Y and Z
        ax.set_zlabel('Y')
        
        # Set the view angle
        ax.view_init(elev=20, azim=45)
    
    # Create line objects for bones
    lines1, lines2, lines3 = [], [], []
    
    for joint_idx in range(num_joints):
        parent_idx = joint_parents[joint_idx]
        parent_idx = parent_idx if parent_idx >= 0 else None
        # parent_idx = joint_parents.get(joint_idx)
        if parent_idx is not None:
            line1, = ax1.plot([], [], [], 'b-', linewidth=2)
            line2, = ax2.plot([], [], [], 'g-', linewidth=2)
            line3, = ax3.plot([], [], [], 'r-', linewidth=2)
            lines1.append((joint_idx, parent_idx, line1))
            lines2.append((joint_idx, parent_idx, line2))
            lines3.append((joint_idx, parent_idx, line3))
    
    # Create scatter objects for joints
    joints_scatter1 = ax1.scatter([], [], [], c='b', s=30)
    joints_scatter2 = ax2.scatter([], [], [], c='g', s=30)
    joints_scatter3 = ax3.scatter([], [], [], c='r', s=30)
    
    # Initialize plots
    def init():
        for _, _, line in lines1 + lines2 + lines3:
            line.set_data([], [])
            line.set_3d_properties([])
        
        joints_scatter1._offsets3d = ([], [], [])
        joints_scatter2._offsets3d = ([], [], [])
        joints_scatter3._offsets3d = ([], [], [])
        return [line for _, _, line in lines1 + lines2 + lines3] + [
            joints_scatter1, joints_scatter2, joints_scatter3]
    
    # Update function for each frame
    def update(frame):
        # Update frame titles
        ax1.set_title(f'Motion 1 (Frame {frame+1}/{num_frames})')
        ax2.set_title(f'Interpolated Motion (Frame {frame+1}/{num_frames})')
        ax3.set_title(f'Motion 2 (Frame {frame+1}/{num_frames})')
        
        # Get joint positions for current frame
        positions1 = motion1_data[frame]
        positions2 = interp_data[frame]
        positions3 = motion2_data[frame]
        
        # Update motion 1
        for joint_idx, parent_idx, line in lines1:
            joint_pos = positions1[joint_idx]
            parent_pos = positions1[parent_idx]
            line.set_data([parent_pos[0], joint_pos[0]], [parent_pos[2], joint_pos[2]])
            line.set_3d_properties([parent_pos[1], joint_pos[1]])
        
        joints_scatter1._offsets3d = (
            positions1[:, 0],
            positions1[:, 2],
            positions1[:, 1]
        )
        
        # Update interpolated motion
        for joint_idx, parent_idx, line in lines2:
            joint_pos = positions2[joint_idx]
            parent_pos = positions2[parent_idx]
            line.set_data([parent_pos[0], joint_pos[0]], [parent_pos[2], joint_pos[2]])
            line.set_3d_properties([parent_pos[1], joint_pos[1]])
        
        joints_scatter2._offsets3d = (
            positions2[:, 0],
            positions2[:, 2],
            positions2[:, 1]
        )
        
        # Update motion 2
        for joint_idx, parent_idx, line in lines3:
            joint_pos = positions3[joint_idx]
            parent_pos = positions3[parent_idx]
            line.set_data([parent_pos[0], joint_pos[0]], [parent_pos[2], joint_pos[2]])
            line.set_3d_properties([parent_pos[1], joint_pos[1]])
        
        joints_scatter3._offsets3d = (
            positions3[:, 0],
            positions3[:, 2],
            positions3[:, 1]
        )
        
        return [line for _, _, line in lines1 + lines2 + lines3] + [
            joints_scatter1, joints_scatter2, joints_scatter3]
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, init_func=init, blit=True, interval=1000/fps
    )
    
    # Save animation
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=5000)
    ani.save(output_path, writer=writer)
    
    plt.close(fig)
    print(f"Interpolation animation saved to {output_path}")
    
    return output_path


import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from pathlib import Path

def load_and_process_data(csv_path):
    """
    Load and process data from CSV file, organized by trajectories.
    Data is grouped by test/env_info (tasks), test/episode (trajectory numbers), 
    and ordered by test/steps (steps within trajectories).
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Filter for specific environment types
        selected_envs = ['[0, 30]', '[1, 1]', '[1, 30]', '[1, 5]']
        df = df[df['test/env_info'].isin(selected_envs)]
        
        # Sort data by environment, episode, and steps to organize by trajectories
        df = df.sort_values(['test/env_info', 'test/episode', 'test/steps'])
        
        # Convert action strings to numpy arrays more robustly
        def parse_array_str(array_str):
            try:
                # Try ast.literal_eval first for safety
                return np.array(ast.literal_eval(array_str))
            except:
                # Fallback to string splitting if needed
                return np.array([float(x) for x in array_str.strip('[]').split()])
        
        # Organize data by trajectories
        trajectory_data = {}
        
        for env_type in selected_envs:
            env_df = df[df['test/env_info'] == env_type]
            trajectory_data[env_type] = {}
            
            for episode in env_df['test/episode'].unique():
                episode_df = env_df[env_df['test/episode'] == episode]
                # Sort by steps within each episode
                episode_df = episode_df.sort_values('test/steps')
                
                # Process trajectory data
                trajectory_actions = np.array([parse_array_str(action) for action in episode_df['test/action']])
                trajectory_causal_embeddings = np.array([parse_array_str(emb) for emb in episode_df['test/causal_embedding']])
                
                # Process observations and extract positions
                observations = episode_df['test/observations'].apply(ast.literal_eval)
                # Extract robot end effector position (first 3 dimensions)
                trajectory_robot_positions = np.array([obs[0][0:3] for obs in observations])
                # Extract cube positions (dimensions 4-6)
                trajectory_cube_positions = np.array([obs[0][4:7] for obs in observations])
                # Extract cube heights (dimension 6)
                trajectory_cube_heights = np.array([[obs[0][6]] for obs in observations])
                # Calculate position differences
                trajectory_position_differences = trajectory_cube_positions - trajectory_robot_positions
                
                # Extract robot constraints and cube position constraints from CSV columns
                trajectory_robot_constraints = np.array([parse_array_str(constraint) for constraint in episode_df['test/robot_constraint']])
                trajectory_cube_pos_constraints = np.array([parse_array_str(constraint) for constraint in episode_df['test/cube_pos_constraint']])
                
                trajectory_data[env_type][episode] = {
                    'actions': trajectory_actions,
                    'causal_embeddings': trajectory_causal_embeddings,
                    'robot_positions': trajectory_robot_positions,
                    'cube_positions': trajectory_cube_positions,
                    'cube_heights': trajectory_cube_heights,
                    'position_differences': trajectory_position_differences,
                    'robot_constraints': trajectory_robot_constraints,
                    'cube_pos_constraints': trajectory_cube_pos_constraints,
                    'steps': episode_df['test/steps'].values,
                    'env_info': episode_df['test/env_info'].values
                }
        
        # Flatten trajectory data for visualization (backward compatibility)
        all_actions = []
        all_causal_embeddings = []
        all_robot_positions = []
        all_cube_positions = []
        all_cube_heights = []
        all_position_differences = []
        all_robot_constraints = []
        all_cube_pos_constraints = []
        all_env_info = []
        all_steps = []
        
        for env_type in trajectory_data:
            for episode in trajectory_data[env_type]:
                traj = trajectory_data[env_type][episode]
                all_actions.append(traj['actions'])
                all_causal_embeddings.append(traj['causal_embeddings'])
                all_robot_positions.append(traj['robot_positions'])
                all_cube_positions.append(traj['cube_positions'])
                all_cube_heights.append(traj['cube_heights'])
                all_position_differences.append(traj['position_differences'])
                all_robot_constraints.append(traj['robot_constraints'])
                all_cube_pos_constraints.append(traj['cube_pos_constraints'])
                all_env_info.extend(traj['env_info'])
                all_steps.extend(traj['steps'])
        
        # Convert to numpy arrays
        actions = np.vstack(all_actions)
        causal_embeddings = np.vstack(all_causal_embeddings)
        robot_positions = np.vstack(all_robot_positions)
        cube_positions = np.vstack(all_cube_positions)
        cube_heights = np.vstack(all_cube_heights)
        position_differences = np.vstack(all_position_differences)
        
        # Reshape robot_constraints and cube_pos_constraints to 2D arrays
        robot_constraints = np.vstack(all_robot_constraints)
        if robot_constraints.ndim == 3:
            robot_constraints = robot_constraints.reshape(robot_constraints.shape[0], -1)
        
        cube_pos_constraints = np.vstack(all_cube_pos_constraints)
        if cube_pos_constraints.ndim == 3:
            cube_pos_constraints = cube_pos_constraints.reshape(cube_pos_constraints.shape[0], -1)
        
        env_info = pd.Series(all_env_info)
        steps = np.array(all_steps)
        
        # Extract height constraints from robot_constraints and cube_pos_constraints
        # Assuming the last dimension is the height constraint
        robot_height_constraints = robot_constraints[:, -1:] if robot_constraints.shape[1] > 0 else robot_constraints[:, :1]
        cube_pos_height_constraints = cube_pos_constraints[:, -1:] if cube_pos_constraints.shape[1] > 0 else cube_pos_constraints[:, :1]
        
        print(f"Loaded data organized by trajectories:")
        print(f"Total environment types: {len(trajectory_data)}")
        for env_type in trajectory_data:
            print(f"  {env_type}: {len(trajectory_data[env_type])} trajectories")
        print(f"Robot constraints shape: {robot_constraints.shape}")
        print(f"Cube position constraints shape: {cube_pos_constraints.shape}")
        print(f"Robot height constraints shape: {robot_height_constraints.shape}")
        print(f"Cube position height constraints shape: {cube_pos_height_constraints.shape}")
        
        return actions, causal_embeddings, cube_positions, position_differences, cube_heights, env_info, df, trajectory_data, robot_constraints, cube_pos_constraints, robot_height_constraints, cube_pos_height_constraints, steps
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def apply_dimensionality_reduction(actions, method='tsne', perplexity=30, max_iter=1000, random_state=42):
    """
    Apply dimensionality reduction using either t-SNE or PCA.
    """
    if method == 'tsne':
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=max_iter,
            random_state=random_state
        )
        return tsne.fit_transform(actions)
    elif method == 'pca':
        pca = PCA(n_components=2, random_state=random_state)
        return pca.fit_transform(actions)

def create_subplot(ax, data, env_info, steps, title, unique_envs, base_colors, is_height=False):
    """
    Create a single subplot for either t-SNE, PCA, or height visualization.
    """
    if is_height:
        # For height data, create a scatter plot with sample index vs height
        for env_type, color in zip(unique_envs, base_colors):
            mask = env_info == env_type
            heights = data[mask].flatten()
            indices = np.arange(len(heights))
            env_steps = steps[mask]
            
            # Calculate alpha values based on steps (normalize to 0.2-1.0 range)
            max_step = np.max(env_steps) if len(env_steps) > 0 else 1
            min_step = np.min(env_steps) if len(env_steps) > 0 else 0
            if max_step > min_step:
                alphas = 0.2 + 0.8 * (env_steps - min_step) / (max_step - min_step)
            else:
                alphas = np.full(len(env_steps), 0.6)
            
            for i, (idx, height, alpha) in enumerate(zip(indices, heights, alphas)):
                ax.scatter(idx, height, c=color, alpha=alpha, s=50)
            
            # Add a dummy point for legend
            ax.scatter([], [], c=color, label=env_type, alpha=0.6, s=50)
                
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Cube Height')
    else:
        # For t-SNE and PCA visualizations
        for env_type, color in zip(unique_envs, base_colors):
            mask = env_info == env_type
            env_data = data[mask]
            env_steps = steps[mask]
            
            # Calculate alpha values based on steps (normalize to 0.2-1.0 range)
            max_step = np.max(env_steps) if len(env_steps) > 0 else 1
            min_step = np.min(env_steps) if len(env_steps) > 0 else 0
            if max_step > min_step:
                alphas = 0.2 + 0.8 * (env_steps - min_step) / (max_step - min_step)
            else:
                alphas = np.full(len(env_steps), 0.6)
            
            # Plot each point with its corresponding alpha
            for i, (point, alpha) in enumerate(zip(env_data, alphas)):
                ax.scatter(point[0], point[1], c=color, alpha=alpha, s=50)
            
            # Add a dummy point for legend
            ax.scatter([], [], c=color, label=env_type, alpha=0.6, s=50)
            
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
    
    ax.set_title(title)
    ax.grid(True)

def create_visualization(actions_tsne, actions_pca, 
                      diff_tsne, diff_pca,
                      cube_tsne, cube_pca, 
                      heights, 
                      causal_tsne, causal_pca,
                      robot_constraint_tsne, robot_constraint_pca,
                      cube_pos_constraint_tsne, cube_pos_constraint_pca,
                      robot_height_constraints, cube_pos_height_constraints,
                      env_info, steps, title=None, output_path=None):
    """
    Create and save the t-SNE and PCA visualizations in a 8x2 grid.
    """
    plt.figure(figsize=(20, 48))
    
    # Create color palette with high contrast colors
    unique_envs = env_info.unique()
    base_colors = [
        '#FF0000',  # Red for [0, 30]
        '#0000FF',  # Blue for [1, 1]
        '#00FF00',  # Green for [1, 30]
        '#FFD700',  # Gold for [1, 5]
    ]
    colors = base_colors[:len(unique_envs)]
    
    # Create subplots in a 8x2 grid
    ax1 = plt.subplot(8,2,1)   # Actions t-SNE
    ax2 = plt.subplot(8,2,2)   # Actions PCA
    ax3 = plt.subplot(8,2,3)   # Position Diff t-SNE
    ax4 = plt.subplot(8,2,4)   # Position Diff PCA
    ax5 = plt.subplot(8,2,5)   # Cube Position t-SNE
    ax6 = plt.subplot(8,2,6)   # Cube Position PCA
    ax7 = plt.subplot(8,2,7)   # Cube Height Distribution
    ax8 = plt.subplot(8,2,8)   # Causal Embedding t-SNE
    ax9 = plt.subplot(8,2,9)   # Causal Embedding PCA
    ax10 = plt.subplot(8,2,10) # Robot Constraint t-SNE
    ax11 = plt.subplot(8,2,11) # Robot Constraint PCA
    ax12 = plt.subplot(8,2,12) # Cube Position Constraint t-SNE
    ax13 = plt.subplot(8,2,13) # Cube Position Constraint PCA
    ax14 = plt.subplot(8,2,14) # Robot Height Constraints
    ax15 = plt.subplot(8,2,15) # Cube Position Height Constraints
    ax16 = plt.subplot(8,2,16) # (Reserved for future use)
    
    # Create action plots (first row)
    create_subplot(ax1, actions_tsne, env_info, steps, 'Actions t-SNE', unique_envs, colors)
    create_subplot(ax2, actions_pca, env_info, steps, 'Actions PCA', unique_envs, colors)
    
    # Create position difference plots (second row)
    create_subplot(ax3, diff_tsne, env_info, steps, 'Position Difference t-SNE', unique_envs, colors)
    create_subplot(ax4, diff_pca, env_info, steps, 'Position Difference PCA', unique_envs, colors)
    
    # Create cube position plots (third row)
    create_subplot(ax5, cube_tsne, env_info, steps, 'Cube Position t-SNE', unique_envs, colors)
    create_subplot(ax6, cube_pca, env_info, steps, 'Cube Position PCA', unique_envs, colors)
    
    # Create cube height plots (fourth row)
    create_subplot(ax7, heights, env_info, steps, 'Cube Height Distribution', unique_envs, colors, is_height=True)
    create_subplot(ax8, causal_tsne, env_info, steps, 'Causal Embedding t-SNE', unique_envs, colors)
    
    # Create causal embedding plots (fifth row)
    create_subplot(ax9, causal_pca, env_info, steps, 'Causal Embedding PCA', unique_envs, colors)
    create_subplot(ax10, robot_constraint_tsne, env_info, steps, 'Robot Constraint t-SNE', unique_envs, colors)
    
    # Create robot constraint plots (sixth row)
    create_subplot(ax11, robot_constraint_pca, env_info, steps, 'Robot Constraint PCA', unique_envs, colors)
    create_subplot(ax12, cube_pos_constraint_tsne, env_info, steps, 'Cube Position Constraint t-SNE', unique_envs, colors)
    
    # Create cube position constraint plots (seventh row)
    create_subplot(ax13, cube_pos_constraint_pca, env_info, steps, 'Cube Position Constraint PCA', unique_envs, colors)
    
    # Create height constraint plots (eighth row)
    create_subplot(ax14, robot_height_constraints, env_info, steps, 'Robot Height Constraints', unique_envs, colors, is_height=True)
    create_subplot(ax15, cube_pos_height_constraints, env_info, steps, 'Cube Position Height Constraints', unique_envs, colors, is_height=True)
    
    
    # Add a common legend
    plt.figlegend(
        ax1.get_legend_handles_labels()[0],
        unique_envs,
        title="Environment Types",
        loc='center right',
        bbox_to_anchor=(1.15, 0.5)
    )
    
    # Add main title if provided
    if title:
        plt.suptitle(title, fontsize=14, y=1.02)
    
    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    for env_type in unique_envs:
        count = sum(env_info == env_type)
        print(f"Environment Type: {env_type}, Count: {count}")
    
    # Save the plot if output path is provided
    if output_path:
        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    
    plt.show()
    plt.close()

def main():
    try:
        # Configuration
        csv_path = "/home/xi/yxh_space/SaMI/SaMI/saved_models/PandaPush_obs22_object_size_6cm_gripper_constraint/SaCCM_trainenv_mix_cube_height_coef_0_1/history.csv"
        output_path = "visualization_comparison.png"
        
        print("Loading and processing data...")
        actions, causal_embeddings, cube_positions, position_differences, cube_heights, env_info, df, trajectory_data, robot_constraints, cube_pos_constraints, robot_height_constraints, cube_pos_height_constraints, steps = load_and_process_data(csv_path)
        
        print("Applying t-SNE dimensionality reduction to actions...")
        actions_tsne = apply_dimensionality_reduction(actions, method='tsne')
        
        print("Applying PCA dimensionality reduction to actions...")
        actions_pca = apply_dimensionality_reduction(actions, method='pca')
        
        print("Applying t-SNE dimensionality reduction to causal embeddings...")
        causal_tsne = apply_dimensionality_reduction(causal_embeddings, method='tsne')
        
        print("Applying PCA dimensionality reduction to causal embeddings...")
        causal_pca = apply_dimensionality_reduction(causal_embeddings, method='pca')
        
        print("Applying t-SNE dimensionality reduction to cube positions...")
        cube_tsne = apply_dimensionality_reduction(cube_positions, method='tsne')
        
        print("Applying PCA dimensionality reduction to cube positions...")
        cube_pca = apply_dimensionality_reduction(cube_positions, method='pca')
        
        print("Applying t-SNE dimensionality reduction to position differences...")
        diff_tsne = apply_dimensionality_reduction(position_differences, method='tsne')
        
        print("Applying PCA dimensionality reduction to position differences...")
        diff_pca = apply_dimensionality_reduction(position_differences, method='pca')
        
        print("Applying t-SNE dimensionality reduction to robot constraints...")
        robot_constraint_tsne = apply_dimensionality_reduction(robot_constraints, method='tsne')
        
        print("Applying PCA dimensionality reduction to robot constraints...")
        robot_constraint_pca = apply_dimensionality_reduction(robot_constraints, method='pca')
        
        print("Applying t-SNE dimensionality reduction to cube position constraints...")
        cube_pos_constraint_tsne = apply_dimensionality_reduction(cube_pos_constraints, method='tsne')
        
        print("Applying PCA dimensionality reduction to cube position constraints...")
        cube_pos_constraint_pca = apply_dimensionality_reduction(cube_pos_constraints, method='pca')
        
        print("Creating visualizations...")
        create_visualization(
            actions_tsne,
            actions_pca,
            diff_tsne,
            diff_pca,
            cube_tsne,
            cube_pca,
            cube_heights,
            causal_tsne,
            causal_pca,
            robot_constraint_tsne,
            robot_constraint_pca,
            cube_pos_constraint_tsne,
            cube_pos_constraint_pca,
            robot_height_constraints,
            cube_pos_height_constraints,
            env_info,
            steps,
            title="Comparison of Actions, Position Differences, Positions, Heights, Embeddings and Constraints",
            output_path=output_path
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

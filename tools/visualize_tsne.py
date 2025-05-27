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
    Load and process data from CSV file.
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Filter for specific environment types
        selected_envs = ['[0, 30]', '[1, 1]', '[1, 30]', '[1, 5]']
        df = df[df['test/env_info'].isin(selected_envs)]
        
        # Convert action strings to numpy arrays more robustly
        def parse_array_str(array_str):
            try:
                # Try ast.literal_eval first for safety
                return np.array(ast.literal_eval(array_str))
            except:
                # Fallback to string splitting if needed
                return np.array([float(x) for x in array_str.strip('[]').split()])
        
        # Process actions and causal embeddings
        actions = np.array([parse_array_str(action) for action in df['test/action']])
        causal_embeddings = np.array([parse_array_str(emb) for emb in df['test/causal_embedding']])
        
        # Process observations and extract positions
        observations = df['test/observations'].apply(ast.literal_eval)
        # Extract robot end effector position (first 3 dimensions)
        robot_positions = np.array([obs[0][0:3] for obs in observations])
        # Extract cube positions (dimensions 4-6)
        cube_positions = np.array([obs[0][4:7] for obs in observations])
        # Extract cube heights (dimension 6)
        cube_heights = np.array([[obs[0][6]] for obs in observations])
        # Calculate position differences
        position_differences = cube_positions - robot_positions
        
        env_info = df['test/env_info']
        
        return actions, causal_embeddings, cube_positions, position_differences, cube_heights, env_info, df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def apply_dimensionality_reduction(actions, method='tsne', perplexity=30, n_iter=1000, random_state=42):
    """
    Apply dimensionality reduction using either t-SNE or PCA.
    """
    if method == 'tsne':
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state
        )
        return tsne.fit_transform(actions)
    elif method == 'pca':
        pca = PCA(n_components=2, random_state=random_state)
        return pca.fit_transform(actions)

def create_subplot(ax, data, env_info, title, unique_envs, base_colors, is_height=False):
    """
    Create a single subplot for either t-SNE, PCA, or height visualization.
    """
    if is_height:
        # For height data, create a scatter plot with sample index vs height
        for env_type, color in zip(unique_envs, base_colors):
            mask = env_info == env_type
            heights = data[mask].flatten()
            indices = np.arange(len(heights))
            ax.scatter(indices, heights, c=[color], label=env_type, alpha=0.6, s=50)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Cube Height')
    else:
        # For t-SNE and PCA visualizations
        for env_type, color in zip(unique_envs, base_colors):
            mask = env_info == env_type
            ax.scatter(
                data[mask, 0],
                data[mask, 1],
                c=[color],
                label=env_type,
                alpha=0.6,
                s=50
            )
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
    
    ax.set_title(title)
    ax.grid(True)

def create_visualization(actions_tsne, actions_pca, 
                      diff_tsne, diff_pca,
                      cube_tsne, cube_pca, 
                      heights, 
                      causal_tsne, causal_pca,
                      env_info, title=None, output_path=None):
    """
    Create and save the t-SNE and PCA visualizations in a 5x2 grid.
    """
    plt.figure(figsize=(20, 35))
    
    # Create color palette with high contrast colors
    unique_envs = env_info.unique()
    base_colors = [
        '#FF0000',  # Red for [0, 30]
        '#0000FF',  # Blue for [1, 1]
        '#00FF00',  # Green for [1, 30]
        '#FFD700',  # Gold for [1, 5]
    ]
    colors = base_colors[:len(unique_envs)]
    
    # Create subplots in a 5x2 grid
    ax1 = plt.subplot(521)  # Actions t-SNE
    ax2 = plt.subplot(522)  # Actions PCA
    ax3 = plt.subplot(523)  # Position Diff t-SNE
    ax4 = plt.subplot(524)  # Position Diff PCA
    ax5 = plt.subplot(525)  # Cube Position t-SNE
    ax6 = plt.subplot(526)  # Cube Position PCA
    ax7 = plt.subplot(527)  # Cube Height Distribution
    ax8 = plt.subplot(528)  # Cube Height Distribution (Duplicate)
    ax9 = plt.subplot(529)  # Causal Embedding t-SNE
    ax10 = plt.subplot(5,2,10)  # Causal Embedding PCA
    
    # Create action plots (first row)
    create_subplot(ax1, actions_tsne, env_info, 'Actions t-SNE', unique_envs, colors)
    create_subplot(ax2, actions_pca, env_info, 'Actions PCA', unique_envs, colors)
    
    # Create position difference plots (second row)
    create_subplot(ax3, diff_tsne, env_info, 'Position Difference t-SNE', unique_envs, colors)
    create_subplot(ax4, diff_pca, env_info, 'Position Difference PCA', unique_envs, colors)
    
    # Create cube position plots (third row)
    create_subplot(ax5, cube_tsne, env_info, 'Cube Position t-SNE', unique_envs, colors)
    create_subplot(ax6, cube_pca, env_info, 'Cube Position PCA', unique_envs, colors)
    
    # Create cube height plots (fourth row)
    create_subplot(ax7, heights, env_info, 'Cube Height Distribution', unique_envs, colors, is_height=True)
    create_subplot(ax8, heights, env_info, 'Cube Height Distribution (Duplicate)', unique_envs, colors, is_height=True)
    
    # Create causal embedding plots (bottom row)
    create_subplot(ax9, causal_tsne, env_info, 'Causal Embedding t-SNE', unique_envs, colors)
    create_subplot(ax10, causal_pca, env_info, 'Causal Embedding PCA', unique_envs, colors)
    
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
        actions, causal_embeddings, cube_positions, position_differences, cube_heights, env_info, df = load_and_process_data(csv_path)
        
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
            env_info,
            title="Comparison of Actions, Position Differences, Positions, Heights and Embeddings",
            output_path=output_path
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

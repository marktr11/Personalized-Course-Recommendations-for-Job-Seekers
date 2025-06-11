"""
Learning Curves Visualization Module

This module provides functions to visualize and compare learning curves from different experiments
with mastery levels and clustering-based reward adjustment.

Key Features:
    - Individual learning curve plots
    - Clustering effect comparison plots
    - Model comparison plots
    - Support for multiple k values (1,2,3,...)
    - Automatic k value detection

Plot Types:
    1. Individual Learning Curves:
        - Shows training progress for each model
        - Includes clustering status
        - Displays average applicable jobs
    
    2. Clustering Comparison:
        - Compares performance with/without clustering
        - Shows impact of reward adjustment
        - Highlights stability improvements
    
    3. Model Comparison:
        - Compares different RL algorithms
        - Shows performance across k values
        - Includes clustering variants

Configuration:
    To change the target branch for plots, modify the BRANCH_NAME variable below.
    Example:
        BRANCH_NAME = "clusteringRL-mastery-levels"   # Current branch
        BRANCH_NAME = "clusteringRL-no-mastery-levels"  # Different branch

Directory Structure:
    Code/results/
    ├── [BRANCH_NAME]/
    │   ├── plots/     # All plots will be saved here
    │   └── data/      # Data files
    └── backups/       # Backup directories

Usage:
    1. Set BRANCH_NAME to your target branch
    2. Run the script:
       python visualize_learning_curves.py
    3. Choose from menu options:
       - Generate all plots
       - Generate only comparison plots
       - Exit

Note:
    - All plots will be saved in Code/results/[BRANCH_NAME]/plots/
    - Existing plots will be skipped unless deleted
    - Each comparison plot has a unique name based on model and k value
    - Plots include mastery levels and clustering information
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import yaml

# Configuration
BRANCH_NAME = "clusteringRL-mastery-levels"  # Change this to your target branch

def get_plots_directory(results_dir):
    """Get the plots directory for the current branch.
    
    Args:
        results_dir (str): Base results directory path
        
    Returns:
        str: Path to the plots directory for current branch
        
    Note:
        Creates the directory if it doesn't exist
    """
    # Convert to absolute path
    results_dir = os.path.abspath(results_dir)
    plots_dir = os.path.join(results_dir, BRANCH_NAME, "plots")
    
    try:
        os.makedirs(plots_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating plots directory: {e}")
        print(f"Attempted to create: {plots_dir}")
        raise
        
    return plots_dir

def load_evaluation_data(file_path):
    """Load evaluation data from a text file.
    
    Args:
        file_path (str): Path to the evaluation data file
        
    Returns:
        tuple: (steps, metrics, start_step) arrays from the data file
    """
    data = np.loadtxt(file_path)
    steps = data[:, 0]
    metrics = data[:, 1]  # Using the first metric column
    start_step = steps[0]  # Store the starting step
    
    # Normalize steps to start from 0 for plotting
    steps = steps - start_step
    
    return steps, metrics, start_step

def get_experiment_title(model_name, k, is_clustered):
    """Generate a descriptive title for the experiment.
    
    Args:
        model_name (str): Name of the model (e.g., 'dqn', 'ppo')
        k (str): Number of clusters
        is_clustered (bool): Whether clustering is used
        
    Returns:
        str: Formatted title for the plot
    """
    # Map model names to full names
    model_map = {
        'dqn': 'DQN',
        'ppo': 'PPO'
    }
    
    # Get formatted names
    model = model_map.get(model_name.lower(), model_name.upper())
    
    # Add clustering information
    clustering_info = "with Clustering" if is_clustered else "without Clustering"
    
    return f"{model} - {clustering_info} (k={k})"

def plot_learning_curves(results_dir):
    """Plot learning curves for all evaluation files in the results directory.
    
    Args:
        results_dir (str): Base results directory path
        
    Note:
        - Plots are saved in Code/results/[BRANCH_NAME]/plots/
        - Existing plots are skipped
        - Each plot shows learning curve for a specific model/k combination
    """
    # Get all evaluation result files from the specified branch
    branch_dir = os.path.join(results_dir, BRANCH_NAME, "data")
    result_files = glob.glob(os.path.join(branch_dir, "all_*.txt"))
    
    if not result_files:
        print(f"\nWarning: No result files found in {branch_dir}")
        print("Please check if:")
        print("1. The branch directory exists")
        print("2. The result files follow the naming pattern: all_*.txt")
        print("3. You are looking in the correct branch")
        return
    
    # Set global font settings
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titlesize': 16,
        'figure.titleweight': 'bold'
    })
    
    # Get plots directory
    plots_dir = get_plots_directory(results_dir)
    
    # Count existing plots
    existing_plots = glob.glob(os.path.join(plots_dir, "*.png"))
    print(f"\nFound {len(existing_plots)} existing plots in {plots_dir}")
    
    print(f"\nFound {len(result_files)} result files to process...")
    print("Files found:")
    for file in result_files:
        print(f"- {os.path.basename(file)}")
    
    # Keep track of new plots
    new_plots = 0
    
    for file_path in result_files:
        # Extract model info from filename
        filename = os.path.basename(file_path)
        
        # Parse model information more carefully
        parts = filename.split('_')
        if len(parts) < 5:  # Skip invalid filenames
            print(f"Skipping invalid filename format: {filename}")
            continue
            
        model_name = parts[1]
        # Extract k value from filename - look for pattern k_X
        k = None
        for i, part in enumerate(parts):
            if part == 'k' and i + 1 < len(parts):
                try:
                    k = parts[i + 1]  # Get the value after 'k'
                    break
                except:
                    continue
        
        if k is None:
            print(f"Could not find k value in filename: {filename}")
            continue
            
        # Check if file has clustering info from filename
        has_clustering = any('cluster' in part.lower() for part in parts)
        
        # Create output filename based on result file's clustering status
        clustering_suffix = "clustered" if has_clustering else "no_clustering"
        output_filename = f"{model_name}_mastery_levels_{clustering_suffix}_k{k}.png"
        output_path = os.path.join(plots_dir, output_filename)
        
        # Skip if plot already exists
        if os.path.exists(output_path):
            print(f"Skipping existing plot: {output_filename}")
            continue
            
        print(f"Creating new plot: {output_filename}...")
        new_plots += 1
        
        # Load data
        steps, metrics, start_step = load_evaluation_data(file_path)
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Plot metrics
        plt.plot(steps, metrics, 'b-', linewidth=2, label='Average Applicable Jobs')
        plt.xlabel('Training Steps', fontsize=14, fontweight='bold')
        plt.ylabel('Average Applicable Jobs', fontsize=14, fontweight='bold')
        
        # Set title using the new format
        title = get_experiment_title(model_name, k, has_clustering)
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.grid(True, alpha=0.3)
        
        # Set legend position to bottom right
        plt.legend(fontsize=12, loc='lower right', bbox_to_anchor=(1.0, 0.0))
        
        # Make y-axis ticks more readable
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # Format y-axis to show plain numbers instead of scientific notation
        plt.ticklabel_format(style='plain', axis='y')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nSummary:")
    print(f"- Existing plots: {len(existing_plots)}")
    print(f"- New plots created: {new_plots}")
    print(f"- Total plots after update: {len(existing_plots) + new_plots}")

def compare_clustering_effect(results_dir, model_name, k):
    """Compare learning curves with and without clustering for specific model and k."""
    # Get plots directory
    plots_dir = get_plots_directory(results_dir)
    
    # Set global font settings
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titlesize': 16,
        'figure.titleweight': 'bold'
    })
    
    # Find result files for both cases in the specified branch
    branch_dir = os.path.join(results_dir, BRANCH_NAME, "data")
    clustered_pattern = f"all_{model_name}_k_{k}_total_steps_*_clusters_auto_run_*.txt"
    no_cluster_pattern = f"all_{model_name}_k_{k}_total_steps_*_run_*.txt"
    
    clustered_files = glob.glob(os.path.join(branch_dir, clustered_pattern))
    no_cluster_files = glob.glob(os.path.join(branch_dir, no_cluster_pattern))
    
    if not clustered_files or not no_cluster_files:
        print(f"\nWarning: Could not find both clustered and non-clustered files for {model_name} k={k}")
        print("Looking for:")
        print(f"- Clustered pattern: {clustered_pattern}")
        print(f"- No cluster pattern: {no_cluster_pattern}")
        print("\nFiles found:")
        if clustered_files:
            print("Clustered files:")
            for f in clustered_files:
                print(f"- {os.path.basename(f)}")
        if no_cluster_files:
            print("Non-clustered files:")
            for f in no_cluster_files:
                print(f"- {os.path.basename(f)}")
        return
    
    # Print debug information
    print("\nDebug Information:")
    print(f"Clustered file: {os.path.basename(clustered_files[0])}")
    print(f"No cluster file: {os.path.basename(no_cluster_files[0])}")
    
    # Check if we're reading the same file
    if clustered_files[0] == no_cluster_files[0]:
        print("\nWARNING: Reading the same file for both clustered and non-clustered data!")
        return
        
    # Load data
    clustered_steps, clustered_metrics, clustered_start = load_evaluation_data(clustered_files[0])
    no_cluster_steps, no_cluster_metrics, no_cluster_start = load_evaluation_data(no_cluster_files[0])
    
    # Print data statistics
    print("\nData Statistics:")
    print("Clustered data:")
    print(f"- Number of points: {len(clustered_steps)}")
    print(f"- Training started at step: {clustered_start:,}")
    print(f"- Steps range: {clustered_start:,} to {clustered_start + clustered_steps[-1]:,}")
    print(f"- Metrics range: {min(clustered_metrics):.4f} to {max(clustered_metrics):.4f}")
    print(f"- First few metrics: {clustered_metrics[:5]}")
    
    print("\nNo cluster data:")
    print(f"- Number of points: {len(no_cluster_steps)}")
    print(f"- Training started at step: {no_cluster_start:,}")
    print(f"- Steps range: {no_cluster_start:,} to {no_cluster_start + no_cluster_steps[-1]:,}")
    print(f"- Metrics range: {min(no_cluster_metrics):.4f} to {max(no_cluster_metrics):.4f}")
    print(f"- First few metrics: {no_cluster_metrics[:5]}")
    
    # Get total steps from filenames
    clustered_total_steps = int(os.path.basename(clustered_files[0]).split('_')[4])
    no_cluster_total_steps = int(os.path.basename(no_cluster_files[0]).split('_')[4])
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Plot both curves
    plt.plot(clustered_steps, clustered_metrics, 'b-', linewidth=2, 
             label=f'With Clustering (Started at {clustered_start:,} steps)')
    plt.plot(no_cluster_steps, no_cluster_metrics, 'r--', linewidth=2, 
             label=f'Without Clustering (Started at {no_cluster_start:,} steps)')
    
    plt.xlabel('Training Steps (Normalized)', fontsize=14, fontweight='bold')
    plt.ylabel('Average Applicable Jobs', fontsize=14, fontweight='bold')
    
    # Set title with training information
    title = (f"{model_name.upper()} k={k} - Clustering Comparison\n"
             f"Clustered: {clustered_start:,} to {clustered_start + clustered_steps[-1]:,} steps, "
             f"No Clustering: {no_cluster_start:,} to {no_cluster_start + no_cluster_steps[-1]:,} steps")
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    
    # Set legend position to bottom right
    plt.legend(fontsize=12, loc='lower right', bbox_to_anchor=(1.0, 0.0))
    
    # Make y-axis ticks more readable
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Format y-axis to show plain numbers instead of scientific notation
    plt.ticklabel_format(style='plain', axis='y')
    
    # Create output filename
    output_filename = f"{model_name}_k{k}_clustering_comparison.png"
    output_path = os.path.join(plots_dir, output_filename)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nCreated comparison plot: {output_filename}")

def compare_models(results_dir):
    """Compare models with and without clustering for different k values."""
    # Get plots directory
    plots_dir = get_plots_directory(results_dir)
    
    # Set global font settings
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.titlesize': 16,
        'figure.titleweight': 'bold'
    })
    
    # Define models to compare
    models = ['dqn', 'ppo']
    
    # Define color scheme and styles
    colors = {
        'dqn': {
            'clustered': '#1f77b4',     # Dark blue
            'non_clustered': '#7fb3d5'  # Light blue
        },
        'ppo': {
            'clustered': '#d62728',     # Dark red
            'non_clustered': '#ff9999'  # Light red
        }
    }
    
    # Find all available k values from result files
    branch_dir = os.path.join(results_dir, BRANCH_NAME, "data")
    k_values = set()
    for model in models:
        for is_clustered in [True, False]:
            if is_clustered:
                pattern = f"all_{model}_k_*_total_steps_*_clusters_auto_run_*.txt"
            else:
                pattern = f"all_{model}_k_*_total_steps_*_run_*.txt"
            
            files = glob.glob(os.path.join(branch_dir, pattern))
            for file in files:
                # Extract k value from filename
                parts = os.path.basename(file).split('_')
                for i, part in enumerate(parts):
                    if part == 'k' and i + 1 < len(parts):
                        k_values.add(parts[i + 1])
    
    if not k_values:
        print("\nNo result files found with k values")
        return
        
    print(f"\nFound k values: {sorted(list(k_values))}")
    
    # Create separate plots for each k value
    for k in sorted(list(k_values)):
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Store last points for text positioning
        last_points = []
        all_steps = []  # Store all steps for x-axis limits
        all_metrics = []  # Store all metrics for y-axis limits
        
        # First pass: collect all last points
        for model in models:
            for is_clustered in [True, False]:
                if is_clustered:
                    pattern = f"all_{model}_k_{k}_total_steps_*_clusters_auto_run_*.txt"
                else:
                    pattern = f"all_{model}_k_{k}_total_steps_*_run_*.txt"
                
                files = glob.glob(os.path.join(branch_dir, pattern))
                if files:
                    steps, metrics, start_step = load_evaluation_data(files[0])
                    all_steps.extend(steps)
                    all_metrics.extend(metrics)
                    last_points.append({
                        'step': steps[-1],
                        'metric': metrics[-1],
                        'model': model,
                        'is_clustered': is_clustered
                    })
        
        if not last_points:
            print(f"\nWarning: No data found for k={k}")
            plt.close()
            continue
            
        # Sort points by metric value
        last_points.sort(key=lambda x: x['metric'])
        
        # Second pass: plot lines and adjust text positions
        for model in models:
            for is_clustered in [True, False]:
                if is_clustered:
                    pattern = f"all_{model}_k_{k}_total_steps_*_clusters_auto_run_*.txt"
                else:
                    pattern = f"all_{model}_k_{k}_total_steps_*_run_*.txt"
                
                files = glob.glob(os.path.join(branch_dir, pattern))
                if files:
                    # Load and plot data
                    steps, metrics, start_step = load_evaluation_data(files[0])
                    
                    # Define line style and label
                    linestyle = '-' if is_clustered else '--'
                    color = colors[model]['clustered' if is_clustered else 'non_clustered']
                    
                    # Add clustering indicator to label
                    clustering_indicator = 'CL-' if is_clustered else ''
                    label = f"{clustering_indicator}{model.upper()}"
                    
                    # Plot the line
                    plt.plot(steps, metrics, color=color, linestyle=linestyle, linewidth=2, label=label)
                    
                    # Add marker for the last point
                    last_step = steps[-1]
                    last_metric = metrics[-1]
                    plt.plot(last_step, last_metric, 'o', color=color, markersize=8)
                    
                    # Find index of current point in sorted list
                    current_idx = next(i for i, p in enumerate(last_points) 
                                     if p['model'] == model and p['is_clustered'] == is_clustered)
                    
                    # Calculate base offset
                    x_offset = (max(all_steps) - min(all_steps)) * 0.015
                    y_offset = (max(all_metrics) - min(all_metrics)) * 0.015
                    
                    # Adjust y_offset based on position in sorted list
                    if current_idx > 0:
                        prev_metric = last_points[current_idx - 1]['metric']
                        if abs(last_metric - prev_metric) < (max(all_metrics) - min(all_metrics)) * 0.1:
                            y_offset = (current_idx + 1) * y_offset * 1.5
                    
                    plt.text(last_step + x_offset, last_metric + y_offset, f'{last_metric:.2f}', 
                            color=color, fontsize=15, fontweight='bold',
                            ha='left', va='bottom')
        
        # Set labels and title
        plt.xlabel('Training Steps', fontsize=14, fontweight='bold')
        plt.ylabel('Average Applicable Jobs', fontsize=14, fontweight='bold')
        
        # Set title
        title = f"Models Comparison (k={k})"
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        plt.grid(True, alpha=0.3)
        
        # Increase padding for x and y axis
        plt.xlim(min(all_steps), max(all_steps) * 1.11)
        plt.ylim(min(all_metrics) * 0.95, max(all_metrics) * 1.15)
   
        # Set legend position to bottom right
        plt.legend(fontsize=14, loc='lower right', bbox_to_anchor=(1.0, 0.0))
        
        # Make y-axis ticks more readable
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # Format y-axis to show plain numbers instead of scientific notation
        plt.ticklabel_format(style='plain', axis='y')
        
        # Create output filename
        output_filename = f"models_mastery_levels_k{k}_comparison.png"
        output_path = os.path.join(plots_dir, output_filename)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created comparison plot: {output_filename}")

def show_menu():
    """Display the main menu and get user choice."""
    print("\nLearning Curves Visualization Menu:")
    print("1. Generate all plots")
    print("2. Generate only comparison plots")
    print("3. Exit")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-3): "))
            if choice in [1, 2, 3]:
                return choice
            print("Invalid choice. Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    # Define directories
    results_dir = os.path.join("Code", "results")
    
    print(f"Generating plots for branch: {BRANCH_NAME}")
    print(f"Plots will be saved in: {os.path.join(results_dir, BRANCH_NAME, 'plots')}")
    print(f"Looking for result files in: {results_dir}")
    
    while True:
        choice = show_menu()
        
        if choice == 1:
            # Plot learning curves
            plot_learning_curves(results_dir)
            
            # Compare models for different k values
            compare_models(results_dir)
            
        elif choice == 2:
            # Compare models for different k values
            compare_models(results_dir)
            
        else:  # choice == 3
            break
    
    print("\nLearning curves update completed.") 
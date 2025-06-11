"""
Clustering Module for Course Recommendation System

This module implements clustering functionality to improve RL performance by adjusting rewards
based on course cluster membership. The clustering helps identify similar courses based on their
provided skills, which is then used to modify the reward signal to encourage more stable learning.

The reward adjustment follows these rules:
1. Same cluster & reward increase: Moderate encouragement (x1.1)
   - Encourages the agent to continue exploring within the same cluster when it's working well
   - Reduced from 1.2 to make it easier for k=3 to overcome
2. Same cluster & reward decrease: Light penalty (x0.9)
   - Slightly discourages actions that decrease reward within the same cluster
3. Different cluster & reward increase: Strong encouragement (x1.3)
   - Encourages the agent to explore new clusters when it finds improvements
   - Reduced from 1.5 to prevent over-exploration
4. Different cluster & reward decrease: Moderate penalty (x0.8)
   - Discourages actions that decrease reward when switching clusters
   - Increased from 0.7 to reduce penalty severity

The clustering is based on two key metrics for each course:
1. Skill Coverage: The percentage of total skills that the course provides
2. Skill Diversity: An entropy-based measure of how evenly distributed the course's skills are

Example:
    ```python
    # Initialize clusterer
    clusterer = CourseClusterer(n_clusters=5, random_state=42, auto_clusters=True)
    
    # Fit clusters to courses
    clusterer.fit_course_clusters(courses)
    
    # Adjust reward based on clustering
    adjusted_reward = clusterer.adjust_reward(
        course_idx=0,
        original_reward=1.0,
        prev_reward=0.8
    )
    ```
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

class CourseClusterer:
    """Class for clustering courses and adjusting rewards based on cluster membership.
    
    This class implements a clustering-based reward adjustment mechanism that helps
    stabilize and improve the learning process in the RL environment. It clusters
    courses based on their provided skills and uses this information to modify
    rewards according to predefined rules.
    
    Attributes:
        n_clusters (int): Number of clusters to create
        course_clusters (np.ndarray): Array of cluster assignments for each course
        scaler (StandardScaler): Scaler for normalizing features before clustering
        prev_cluster (int): Cluster of the previous course recommendation
        features (np.ndarray): Original features used for clustering
        random_state (int): Random seed for reproducibility
        auto_clusters (bool): Whether to automatically determine optimal number of clusters
        max_clusters (int): Maximum number of clusters to try when using elbow method
        optimal_k (int): Optimal number of clusters determined by elbow method
        clustering_dir (str): Directory to save clustering results
        reward_multipliers (dict): Dictionary of reward adjustment multipliers
    """
    
    def __init__(self, n_clusters=5, random_state=42, auto_clusters=False, max_clusters=10, config=None):
        """Initialize the clusterer.
        
        Args:
            n_clusters (int): Number of clusters to create
            random_state (int): Random seed for reproducibility
            auto_clusters (bool): Whether to automatically determine optimal number of clusters
            max_clusters (int): Maximum number of clusters to try when using elbow method
            config (dict): Configuration dictionary containing reward multipliers
        """
        self.n_clusters = n_clusters
        self.course_clusters = None
        self.scaler = StandardScaler()
        self.prev_cluster = None
        self.features = None
        self.random_state = random_state
        self.auto_clusters = auto_clusters
        self.max_clusters = max_clusters
        self.optimal_k = None
        
        # Set reward multipliers from config or use defaults
        self.reward_multipliers = {
            'same_cluster_increase': config.get('same_cluster_increase', 1.1) if config else 1.1,
            'same_cluster_decrease': config.get('same_cluster_decrease', 0.9) if config else 0.9,
            'diff_cluster_increase': config.get('diff_cluster_increase', 1.3) if config else 1.3,
            'diff_cluster_decrease': config.get('diff_cluster_decrease', 0.8) if config else 0.8
        }
        
        # Create Clustering directory if it doesn't exist
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.clustering_dir = os.path.join(current_dir, "..", "Clustering")
        os.makedirs(self.clustering_dir, exist_ok=True)
        
    def find_optimal_clusters(self, features_scaled):
        """Find optimal number of clusters using elbow method."""
        plot_path = os.path.join(self.clustering_dir, 'elbow_curve_skillslevels.png')
        print("\nFinding optimal number of clusters...")
        
        inertias = []
        K = range(1, self.max_clusters + 1)
        
        for k in K:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )
            kmeans.fit(features_scaled)
            inertias.append(kmeans.inertia_)
        
        # Calculate the rate of change of inertia
        changes = np.diff(inertias)
        changes_r = np.diff(changes)
        
        # Find the elbow point (where the rate of change changes most)
        optimal_k = np.argmax(changes_r) + 2  # +2 because we took two diffs
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(K, inertias, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
        plt.legend()
        
        # Save plot
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
        
    def visualize_feature_pairs(self, features_scaled):
        """Visualize relationships between features using correlation matrix.
        
        Args:
            features_scaled: Scaled features used for clustering
        """
        # Create DataFrame for easier plotting
        feature_names = ['Coverage', 'Required Entropy', 'Provided Entropy', 
                        'Avg Level Gap', 'Max Level Gap']
        df = pd.DataFrame(features_scaled, columns=feature_names)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create figure with larger size
        plt.figure(figsize=(10, 8))
        
        # Plot correlation matrix
        plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # Add correlation values with bold text
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                        ha='center', va='center',
                        color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                        fontsize=12,
                        fontweight='bold')
        
        # Add colorbar with larger font
        cbar = plt.colorbar(label='Correlation Coefficient')
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_ylabel('Correlation Coefficient', fontsize=14, fontweight='bold')
        
        # Add labels with larger font
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right', fontsize=12, fontweight='bold')
        plt.yticks(range(len(feature_names)), feature_names, fontsize=12, fontweight='bold')
        
        # Add title with larger font
        plt.title('Feature Correlation Matrix', pad=20, fontsize=16, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.clustering_dir, 'feature_correlation.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nSaved correlation matrix to: {plot_path}")
        
    def fit_course_clusters(self, courses):
        """Fit clusters for courses based on their required and provided skills.
        
        This method performs the following steps:
        1. Extract required and provided skills from courses
        2. Calculate 5 features for each course:
           - Coverage: Overall skill coverage ratio
           - Required Entropy: Diversity of required skills
           - Provided Entropy: Diversity of provided skills
           - Avg Level Gap: Average difference between required and provided levels
           - Max Level Gap: Maximum difference between required and provided levels
        3. Scale features to have zero mean and unit variance
        4. Find optimal number of clusters if auto_clusters is enabled
        5. Perform K-means clustering on all courses
        6. Visualize results using PCA and feature pairs
        
        Args:
            courses: Array of courses, each containing required and provided skills
        """
        print("\nStarting course clustering...")
        
        # Extract required and provided skills from courses
        # courses[:, 0] contains required skills for all courses
        # courses[:, 1] contains provided skills for all courses
        required_skills = courses[:, 0]  # Shape: (n_courses, n_skills)
        provided_skills = courses[:, 1]  # Shape: (n_courses, n_skills)
        
        n_skills = required_skills.shape[1]  # Total number of skills
        max_level = 3  # Maximum skill level in the system
        
        # 1. Coverage features: Calculate how much of the total possible skill levels are covered
        # For each course, calculate coverage as ratio of actual levels to maximum possible levels
        required_coverage = np.sum(required_skills, axis=1) / (n_skills * max_level)  # Shape: (n_courses,)
        provided_coverage = np.sum(provided_skills, axis=1) / (n_skills * max_level)  # Shape: (n_courses,)
        coverage = (required_coverage + provided_coverage) / 2  # Average coverage
        
        # 2. Entropy features: Measure the diversity of skill levels
        # Higher entropy means more diverse/balanced distribution of skill levels
        # For required skills
        required_distribution = required_skills / (np.sum(required_skills, axis=1, keepdims=True) + 1e-10)
        required_entropy = -np.sum(required_distribution * np.log2(required_distribution + 1e-10), axis=1)
        
        # For provided skills
        provided_distribution = provided_skills / (np.sum(provided_skills, axis=1, keepdims=True) + 1e-10)
        provided_entropy = -np.sum(provided_distribution * np.log2(provided_distribution + 1e-10), axis=1)
        
        # 3. Level gap features: Measure the difference between required and provided skill levels
        # For each skill in each course, calculate absolute difference between required and provided levels
        level_gap = np.abs(provided_skills - required_skills)  # Shape: (n_courses, n_skills)
        avg_level_gap = np.mean(level_gap, axis=1)  # Average gap across all skills for each course
        max_level_gap = np.max(level_gap, axis=1)  # Maximum gap across all skills for each course
        
        # Combine all features into a single matrix
        # Each row represents a course, each column represents a feature
        self.features = np.column_stack([
            coverage,                    # 1D: Overall coverage of skills
            required_entropy,           # 2D: Diversity of required skills
            provided_entropy,           # 3D: Diversity of provided skills
            avg_level_gap,             # 4D: Average gap between required and provided levels
            max_level_gap              # 5D: Maximum gap between required and provided levels
        ])  # Shape: (n_courses, 5)
        
        # Scale features to have zero mean and unit variance
        # This is important for K-means clustering
        features_scaled = self.scaler.fit_transform(self.features)
        
        # Find optimal number of clusters if auto_clusters is enabled
        if self.auto_clusters:
            self.optimal_k = self.find_optimal_clusters(features_scaled)
            self.n_clusters = self.optimal_k
        
        # Perform K-means clustering on all courses
        # This will assign each course to one of the clusters
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10  # Run 10 times with different initializations and pick the best
        )
        self.course_clusters = kmeans.fit_predict(features_scaled)  # Shape: (n_courses,)
        
        # Store cluster centers and inertia for later use
        self.cluster_centers_ = kmeans.cluster_centers_  # Shape: (n_clusters, 5)
        self.inertia_ = kmeans.inertia_  # Sum of squared distances to closest centroid
        
        # Print cluster information
        print("\nCluster Information:")
        for i in range(self.n_clusters):
            n_formations = np.sum(self.course_clusters == i)
            print(f"Cluster {i}: {n_formations} formations")
        
        # Visualize clusters using PCA
        self.visualize_clusters(features_scaled)
        
        # Visualize feature relationships
        self.visualize_feature_pairs(features_scaled)
        
    def visualize_clusters(self, features_scaled):
        """Visualize the clusters using PCA for dimensionality reduction.
        
        This method:
        1. Reduces the 5D feature space to 2D using PCA
        2. Creates a scatter plot of courses in the reduced space
        3. Shows cluster centers and explained variance
        4. Prints feature contributions to each principal component
        
        The PCA components (PC1, PC2) are linear combinations of the original features.
        The coefficients (feature contributions) show:
        - Positive values: Feature increases when the PC increases
        - Negative values: Feature decreases when the PC increases
        - Magnitude: How strongly the feature influences the PC
        
        For example, if PC1 has:
        - Coverage: 0.562 (positive)
        - Provided Entropy: -0.516 (negative)
        This means courses with high PC1 values tend to have:
        - High coverage
        - Low provided entropy
        """
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        
        # Create figure with larger size and higher DPI
        plt.figure(figsize=(12, 8), dpi=300)
        
        # Plot clusters
        for i in range(self.n_clusters):
            cluster_points = features_2d[self.course_clusters == i]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                label=f'Cluster {i} ({len(cluster_points)} courses)',
                alpha=0.7,
                s=100
            )
        
        # Plot cluster centers
        centers_2d = pca.transform(self.cluster_centers_)
        plt.scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            c='black',
            marker='x',
            s=200,
            linewidths=3,
            label='Cluster Centers'
        )
        
        # Add labels and title with larger font size
        plt.xlabel('First Principal Component', fontsize=14, fontweight='bold')
        plt.ylabel('Second Principal Component', fontsize=14, fontweight='bold')
        plt.title('Course Clusters (PCA Visualization)', fontsize=16, fontweight='bold', pad=20)
        
        # Add explained variance ratio with larger font
        explained_variance = pca.explained_variance_ratio_
        plt.figtext(
            0.73, 0.02,
            f'Explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}',
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5)
        )
        
        # Add legend outside the plot with larger font
        plt.legend(
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            prop={'size': 12, 'weight': 'bold'},
            frameon=True,
            title='Clusters',
            title_fontsize=14
        )
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot with high quality
        plot_path = os.path.join(self.clustering_dir, f'cluster_visualization_pca.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
        # Print feature contributions to principal components
        print("\nFeature Contribution to Principal Components:")
        print("(Values show how much each feature contributes to the principal components)")
        print("(Positive values mean the feature increases with the PC, negative values mean it decreases)")
        print("(The magnitude shows how strongly the feature influences the PC)")
        for i, component in enumerate(pca.components_):
            print(f"\nPC{i+1} (Explained variance: {pca.explained_variance_ratio_[i]:.2%}):")
            for j, feature in enumerate(['Coverage', 'Required Entropy', 'Provided Entropy', 
                                      'Avg Level Gap', 'Max Level Gap']):
                print(f"{feature}: {component[j]:.3f}")
        
    def adjust_reward(self, course_idx, original_reward, prev_reward):
        """Adjust reward based on cluster membership and reward change.
        
        This method implements the reward adjustment rules based on whether the
        course is in the same cluster as the previous course and whether the
        reward has increased or decreased.
        
        For first recommendation in any sequence (k=1,2,3), no reward adjustment is applied.
        
        For subsequent recommendations (k>1), the reward is adjusted based on:
        - Whether the course is in the same cluster as the previous course
        - Whether the reward has increased or decreased
        
        Reward adjustment rules:
        1. Same cluster & reward increase: x{same_cluster_increase}
        2. Same cluster & reward decrease: x{same_cluster_decrease}
        3. Different cluster & reward increase: x{diff_cluster_increase}
        4. Different cluster & reward decrease: x{diff_cluster_decrease}
        
        Args:
            course_idx (int): Index of the current course
            original_reward (float): Original reward value from the environment
            prev_reward (float): Reward value from the previous step
            
        Returns:
            float: Adjusted reward value based on clustering rules
        """
        if self.course_clusters is None:
            return original_reward
            
        # Get current course's cluster
        current_cluster = self.course_clusters[course_idx]
        
        # For first recommendation in any sequence (k=1,2,3)
        # Check if this is the first recommendation (prev_reward is None or 0)
        if prev_reward is None or prev_reward == 0:
            # Store current cluster for next comparison
            self.prev_cluster = current_cluster
            # Return original reward without adjustment
            return original_reward
            
        # For subsequent recommendations in sequence (k>1)
        # Calculate reward change
        reward_change = original_reward - prev_reward
        
        # Store current cluster for next comparison
        self.prev_cluster = current_cluster
        
        # Apply reward adjustment rules using multipliers from config
        if reward_change > 0:  # Reward increased
            if current_cluster == self.prev_cluster:  # Same cluster
                return original_reward * self.reward_multipliers['same_cluster_increase']
            else:  # Different cluster
                return original_reward * self.reward_multipliers['diff_cluster_increase']
        else:  # Reward decreased
            if current_cluster == self.prev_cluster:  # Same cluster
                return original_reward * self.reward_multipliers['same_cluster_decrease']
            else:  # Different cluster
                return original_reward * self.reward_multipliers['diff_cluster_decrease'] 
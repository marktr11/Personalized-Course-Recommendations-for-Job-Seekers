import os
import random

from time import process_time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

import matchings
from clustering import CourseClusterer


class CourseRecEnv(gym.Env):
    """Course Recommendation Environment for Reinforcement Learning.
    
    This class implements a Gymnasium environment for course recommendations using
    reinforcement learning with mastery levels and optional clustering-based reward adjustment.
    
    The environment uses the number of applicable jobs as the reward signal to train
    the RL agent. The reward can be optionally adjusted based on course clustering
    to encourage more stable learning.
    
    Observation Space:
        - Vector of length nb_skills representing learner's current skill levels
        - Each element is an integer in [0, 3] where:
            * 0: No skill
            * 1: Basic mastery
            * 2: Intermediate mastery
            * 3: Advanced mastery
        - Shape: (nb_skills,)
    
    Action Space:
        - Discrete space of size nb_courses
        - Each action represents recommending a specific course
        - Range: [0, nb_courses-1]
    
    Attributes:
        dataset: Dataset object containing learners, jobs, and courses data
        nb_skills (int): Number of unique skills in the system
        mastery_levels (list): List of possible mastery levels [1,2,3]
        max_level (int): Maximum mastery level (3)
        nb_courses (int): Number of available courses
        min_skills (int): Minimum number of skills a learner can have
        max_skills (int): Maximum number of skills a learner can have
        threshold (float): Minimum matching score required for job applicability
        k (int): Maximum number of course recommendations per learner
        use_clustering (bool): Whether to use clustering for reward adjustment
    """
    
    def __init__(self, dataset, threshold=0.5, k=1, is_training=False):
        """Initialize the course recommendation environment.
        
        Args:
            dataset: Dataset object containing learners, jobs, and courses
            threshold (float): Minimum matching score for job applicability
            k (int): Maximum number of course recommendations per learner
            is_training (bool): Whether this is a training environment
        """
        print(f"\nInitializing CourseRecEnv:")
        print(f"use_clustering: {hasattr(dataset, 'config') and dataset.config.get('use_clustering', False)}")
        print(f"is_training: {is_training}")
    
        
        self.dataset = dataset
        self.threshold = threshold
        self.k = k
        self.is_training = is_training
        
        # Initialize basic attributes
        self.nb_skills = len(dataset.skills)  # 46 skills
        self.mastery_levels = [1, 2, 3]
        self.max_level = 3
        self.nb_courses = len(dataset.courses)  # 100 courses
        self.min_skills = min(np.count_nonzero(self.dataset.learners, axis=1))  # 1
        self.max_skills = max(np.count_nonzero(self.dataset.learners, axis=1))  # 15
        
        # Initialize observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=self.max_level, shape=(self.nb_skills,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(self.nb_courses)
        
        # Initialize clustering only in training environment
        self.use_clustering = False
        self.clusterer = None
        self.prev_reward = None
        
        if self.is_training and hasattr(dataset, 'config') and dataset.config.get("use_clustering", False):
            self.use_clustering = True
            self.clusterer = CourseClusterer(
                n_clusters=dataset.config.get("n_clusters", 5),
                random_state=dataset.config.get("seed", 42),
                auto_clusters=dataset.config.get("auto_clusters", False),
                max_clusters=dataset.config.get("max_clusters", 10),
                config=dataset.config.get("clustering", {})
            )
            # Fit clusters in training environment
            if self.clusterer.course_clusters is None:
                self.clusterer.fit_course_clusters(dataset.courses)
        
        # Initialize environment state
        self.reset()

    def _get_obs(self):
        """Get the current observation of the environment.
        
        Returns:
            np.ndarray: Current learner's skill vector representing the state
        """
        return self._agent_skills

    def _get_info(self):
        """Get additional information about the current state.
        
        Returns:
            dict: Dictionary containing the number of applicable jobs for the current state
        """
        return {
            "nb_applicable_jobs": self.dataset.get_nb_applicable_jobs(
                self._agent_skills, threshold=self.threshold
            )
        }

    def get_random_learner(self):
        """Generate a random learner profile for environment initialization.
        
        Creates a learner with:
        - Random number of skills between min_skills and max_skills
        - Random mastery levels for each skill
        
        Returns:
            np.array: the initial observation of the environment, that is the learner's initial skills
        """
        # Randomly choose the number of skills the agent has randomly
        n_skills = random.randint(self.min_skills, self.max_skills)

        # Initialize the skills array with zeros
        initial_skills = np.zeros(self.nb_skills, dtype=np.int32)

        # Choose unique skill indices without replacement
        skill_indices = np.random.choice(self.nb_skills, size=n_skills, replace=False)

        # Assign random mastery levels to these skills, levels can repeat
        initial_skills[skill_indices] = np.random.choice(
            self.mastery_levels, size=n_skills, replace=True
        )
        return initial_skills

    def reset(self, seed=None, learner=None):
        """Method required by the gym environment. It resets the environment to its initial state.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            learner (np.ndarray, optional): Initial learner profile. If None, generates random profile. Defaults to None.
            
        Returns:
            tuple: (observation, info) where:
                - observation: Initial learner's skill vector
                - info: Dictionary containing initial state information
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if learner is not None:
            self._agent_skills = learner
        else:
            self._agent_skills = self.get_random_learner()
        self.nb_recommendations = 0
        
        # Reset clustering-related attributes
        self.prev_reward = None
        if self.use_clustering and self.clusterer is not None:
            self.clusterer.prev_cluster = None
            
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """Execute one step in the environment.
        
        This method:
        1. Recommends a course based on the action
        2. Updates the learner's skills if the course is valid
        3. Calculates the reward based on number of applicable jobs
        4. Adjusts reward using clustering if enabled (only in training)
        5. Checks if the episode should terminate
        
        Args:
            action (int): Index of the course to recommend
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info) where:
                - observation: Updated learner's skill vector
                - reward: Number of applicable jobs
                - terminated: Whether the episode is done
                - truncated: Whether the episode was truncated
                - info: Additional information about the step
        """
        course = self.dataset.courses[action]
        learner = self._agent_skills

        # Skip if learner already has all skills provided by the course
        provided_matching = matchings.learner_course_provided_matching(learner, course)
        required_matching = matchings.learner_course_required_matching(learner, course)
        if required_matching < self.threshold or provided_matching >= 1.0:
            observation = self._get_obs()
            reward = -1
            terminated = True
            info = self._get_info()
            return observation, reward, terminated, False, info
        
        # Update learner's skills
        self._agent_skills = np.maximum(self._agent_skills, course[1])
        observation = self._get_obs()
        info = self._get_info()
        
        # Set reward as number of applicable jobs
        reward = info["nb_applicable_jobs"]

        # Adjust reward using clustering only in training environment
        if self.use_clustering and self.clusterer is not None and self.is_training:
            reward = self.clusterer.adjust_reward(
                course_idx=action,
                original_reward=reward,
                prev_reward=self.prev_reward
            )
            self.prev_reward = reward

        self.nb_recommendations += 1
        terminated = self.nb_recommendations == self.k

        return observation, reward, terminated, False, info


class EvaluateCallback(BaseCallback):
    """Callback for evaluating the RL model during training.
    
    This callback evaluates the model's performance at regular intervals during training.
    It calculates the average number of applicable jobs across all learners and logs
    the results to a file.
    
    The evaluation process:
    1. Runs for each learner in the evaluation dataset
    2. Makes k course recommendations using the current policy
    3. Tracks the number of applicable jobs after each recommendation
    4. Calculates average performance across all learners
    
    Attributes:
        eval_env: Environment used for evaluation
        eval_freq (int): Frequency of evaluation in training steps
        all_results_filename (str): Path to save evaluation results
        mode (str): File opening mode ('w' for first write, 'a' for append)
    """
    
    def __init__(self, eval_env, eval_freq, all_results_filename, verbose=1):
        """Initialize the evaluation callback.
        
        Args:
            eval_env: Environment to use for evaluation
            eval_freq (int): Frequency of evaluation in training steps
            all_results_filename (str): Path to save evaluation results
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        super(EvaluateCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.all_results_filename = all_results_filename
        self.mode = "w"

    def _on_step(self):
        """Evaluate the model at regular intervals during training.
        
        This method:
        1. Evaluates the model every eval_freq steps
        2. Calculates average number of applicable jobs
        3. Logs results to file
        4. Prints progress information
        
        Returns:
            bool: True to continue training
        """
        # Only evaluate every 'eval_freq' training steps
        if self.n_calls % self.eval_freq == 0:
            time_start = process_time()  # Start timing the evaluation
            avg_jobs = 0  # Accumulator for average jobs across learners

            # Loop through each learner in the evaluation dataset
            for learner in self.eval_env.dataset.learners:
                self.eval_env.reset(learner=learner)  # Reset environment with current learner
                done = False  # Flag to control evaluation episode
                tmp_avg_jobs = self.eval_env._get_info()["nb_applicable_jobs"]  # Initial jobs applicable without any recommendations

                # Run one full evaluation episode for the learner
                while not done:
                    obs = self.eval_env._get_obs()  # Get current observation (learner's skills)
                    action, _state = self.model.predict(obs, deterministic=True)  # Predict action using current policy
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)  # Step in environment
                    done = terminated or truncated  # Properly compute done flag

                    # Only update if the recommendation was valid and use nb_applicable_jobs
                    if reward != -1:
                        tmp_avg_jobs = info["nb_applicable_jobs"]

                avg_jobs += tmp_avg_jobs  # Add learner's result to total

            time_end = process_time()  # End timing the evaluation

            # Log the result to the console
            print(
                f"Iteration {self.n_calls}. "
                f"Average jobs: {avg_jobs / len(self.eval_env.dataset.learners)} "
                f"Time: {time_end - time_start}"
            )

            # Write evaluation result to file
            branch_dir = os.path.join(self.eval_env.dataset.config["results_path"], self.eval_env.dataset.config["branch_name"])
            data_dir = os.path.join(branch_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            with open(
                os.path.join(
                    data_dir,
                    self.all_results_filename,
                ),
                self.mode,  # 'w' for first time, 'a' for append afterward
            ) as f:
                f.write(
                    f"{self.n_calls} "
                    f"{avg_jobs / len(self.eval_env.dataset.learners)} "
                    f"{time_end - time_start}\n"
                )

            # After first write, switch mode to append for future evaluations
            if self.mode == "w":
                self.mode = "a"

        return True  # Returning True continues training
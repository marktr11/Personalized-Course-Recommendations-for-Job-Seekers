import os
import random

from time import process_time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback

import matchings


class CourseRecEnv(gym.Env):
    """Course Recommendation Environment for Reinforcement Learning.
    
    This class implements a Gymnasium environment for course recommendations using
    reinforcement learning. The environment simulates the process of recommending
    courses to learners to help them acquire skills needed for jobs.
    
    The environment operates in two modes:
    1. Baseline: Uses number of applicable jobs as reward
    2. Skip-expertise: Uses a utility function that considers both skill acquisition
       and job applicability
    
    Observation Space:
        - Vector of length nb_skills representing learner's current skill levels
        - Each element is an integer in [0, max_level]
        - Shape: (nb_skills,)
    
    Action Space:
        - Discrete space of size nb_courses
        - Each action represents recommending a specific course
        - Range: [0, nb_courses-1]
    
    Attributes:
        dataset: Dataset object containing learners, jobs, and courses data
        nb_skills (int): Number of unique skills in the system
        mastery_levels (list): List of possible mastery levels for skills
        max_level (int): Maximum mastery level possible
        nb_courses (int): Number of available courses
        min_skills (int): Minimum number of skills a learner can have
        max_skills (int): Maximum number of skills a learner can have
        threshold (float): Minimum matching score required for job applicability
        k (int): Maximum number of course recommendations per learner
        baseline (bool): Whether to use baseline reward (True) or utility-based reward (False)
    """
    
    def __init__(self, dataset, threshold=0.8, k=3, baseline=False, feature="Usefulness-as-Rwd"):
        """Initialize the course recommendation environment.
        
        Args:
            dataset: Dataset object containing the recommendation system data
            threshold (float, optional): Minimum matching score for job applicability. Defaults to 0.8.
            k (int, optional): Maximum number of course recommendations. Defaults to 3.
            baseline (bool, optional): Whether to use baseline reward. Defaults to False.
            feature (str, optional): Feature to use for reward. Defaults to "Usefulness-as-Rwd".
        """
        self.feature = feature
        self.baseline = baseline
        self.dataset = dataset 
        self.nb_skills = len(dataset.skills) # 46 skills
        self.mastery_levels = [
            elem for elem in list(dataset.mastery_levels.values()) if elem > 0 # mastery level: [1,2,3,-1]
        ]
        self.max_level = max(self.mastery_levels)
        self.nb_courses = len(dataset.courses) #100 courses
        # get the minimum and maximum number of skills of the learners using np.nonzero
        self.min_skills = min(np.count_nonzero(self.dataset.learners, axis=1)) # 1
        self.max_skills = max(np.count_nonzero(self.dataset.learners, axis=1)) # 15
        self.threshold = threshold 
        self.k = k
        # The observation space is a vector of length nb_skills that represents the learner's skills.
        # The vector contains skill levels, where the minimum level is 0 and the maximum level is max_level (e.g., 3).
        # We cannot set the lower bound to -1 because negative values are not allowed in this Box space.
        self.observation_space = gym.spaces.Box(
            low=0, high=self.max_level, shape=(self.nb_skills,), dtype=np.int32)

        # Define the action space for the environment.
        # This is a discrete space where each action corresponds to recommending a specific course.
        # The total number of possible actions is equal to the number of available courses (nb_courses = 100).
        # The agent will select an integer in [0, nb_courses - 1], representing the index of the recommended course.
        self.action_space = gym.spaces.Discrete(self.nb_courses)

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
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def calculate_course_metrics(self, learner, course):
        """Calculate N1, N2, N3 metrics for a course recommendation.
        
        These metrics evaluate the effectiveness of a course recommendation:
        - N1: Number of missing skills resolved by the course
        - N2: Number of remaining missing skills after taking the course
        - N3: Number of skills provided by the course that are not in missing skills
        
        Args:
            learner (np.ndarray): Current learner's skill vector
            course (np.ndarray): Course's skills array [required, provided]
            
        Returns:
            tuple: (N1, N2, N3) metrics
        """
        # Get missing skills before updating
        missing_skills_before = self.dataset.get_learner_missing_skills(learner)
        
        # Get skills provided by the course
        course_provided_skills = set(np.nonzero(course[1])[0])
        
        # Calculate skills after learning the course, update state
        updated_skills = np.maximum(learner, course[1])
        
        # Get missing skills after updating
        missing_skills_after = self.dataset.get_learner_missing_skills(updated_skills)
        
        # Calculate N1: number of missing skills resolved by this course
        N1 = len(missing_skills_before - missing_skills_after)
        
        # Calculate N2: remaining missing skills after learning this course
        N2 = len(missing_skills_after)
        
        # Calculate N3: number of skills provided by the course that are not in missing skills
        N3 = len(course_provided_skills - missing_skills_before)
        
        return N1, N2, N3

    def calculate_achievable_goals(self, learner, course):
        """Calculate the set of goals (jobs) that become achievable after taking a course.
        
        Args:
            learner (np.ndarray): Current learner's skill vector
            course (np.ndarray): Course's skills array [required, provided]
            
        Returns:
            tuple: (initial_goals, new_goals) where:
                - initial_goals: Number of jobs applicable with current skills
                - new_goals: Number of jobs that become applicable after taking the course
        """
        # Calculate initial goals (jobs applicable with current skills)
        initial_goals = self.dataset.get_nb_applicable_jobs(learner, threshold=self.threshold)
        
        # Calculate skills after learning the course
        updated_skills = np.maximum(learner, course[1])
        
        # Calculate new goals (jobs applicable after learning the course)
        new_goals = self.dataset.get_nb_applicable_jobs(updated_skills, threshold=self.threshold)
        
        return initial_goals, new_goals

    def calculate_utility(self, learner, course):
        """Calculate the utility of a course recommendation.
        
        The utility function is defined as:
        U(φ) = 1/(|G|+1) * [|E(φ)| + N1(φ)/(N1(φ)+N2(φ)+(N3(φ)/(N3(φ)+1)))]
        
        where:
        - |G|: Number of jobs not applicable with initial skills
        - |E(φ)|: Number of new jobs that become applicable
        - N1: Number of missing skills resolved
        - N2: Number of remaining missing skills
        - N3: Number of additional skills provided
        
        Args:
            learner (np.ndarray): Current learner's skill vector
            course (np.ndarray): Course's skills array [required, provided]
            
        Returns:
            float: Utility value of the course recommendation
        """
        # Calculate N1, N2, N3 metrics
        N1, N2, N3 = self.calculate_course_metrics(learner, course)
        
        # Calculate achievable goals
        initial_goals, new_goals = self.calculate_achievable_goals(learner, course)
        
        # Calculate |G|: number of jobs not applicable with initial skills
        total_jobs = len(self.dataset.jobs)
        Ga = total_jobs - initial_goals
        
        # Calculate |E(φ)|: number of new jobs that become applicable
        E_phi = new_goals - initial_goals
        
        # Calculate denominator for N1 fraction
        denominator = N1 + N2 + (N3/(N3+1))
        if denominator == 0:  # Avoid division by zero
            N1_fraction = 0
        else:
            N1_fraction = N1 / denominator
        
        # Calculate U(φ)
        utility = (1 / (Ga + 1)) * (E_phi + N1_fraction)
        
        return utility

    def step(self, action):
        """Execute one step in the environment.
        
        This method:
        1. Recommends a course based on the action
        2. Updates the learner's skills if the course is valid
        3. Calculates the reward based on the selected mode:
           - Baseline: Number of applicable jobs
           - Usefulness-of-info-as-Rwd: Utility function value
           - Weighted-Usefulness-of-info-as-Rwd: Number of applicable jobs + Utility
        4. Checks if the episode should terminate
        
        Args:
            action (int): Index of the course to recommend
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info) where:
                - observation: Updated learner's skill vector
                - reward: Reward value based on the selected mode
                - terminated: Whether the episode is done
                - truncated: Whether the episode was truncated
                - info: Additional information about the step
        """
        course = self.dataset.courses[action]
        learner = self._agent_skills

        # Skip-expertise case: use new metrics and utility
        provided_matching = matchings.learner_course_provided_matching(learner, course)
        required_matching = matchings.learner_course_required_matching(learner, course)
        
        if required_matching < self.threshold or provided_matching >= 1.0:
            observation = self._get_obs()
            reward = -1
            terminated = True
            info = self._get_info()
            return observation, reward, terminated, False, info
        
        if self.baseline : #baseline model
            self._agent_skills = np.maximum(self._agent_skills, course[1])
            observation = self._get_obs()
            info = self._get_info()
            reward = info["nb_applicable_jobs"]
        else: # No-Mastery-Levels Models
            # Calculate Usefulness-of-info-as-Rwd
            utility = self.calculate_utility(learner, course)
            
            self._agent_skills = np.maximum(self._agent_skills, course[1])
            observation = self._get_obs()
            info = self._get_info()
            info["utility"] = utility
            
            if self.feature == "Usefulness-as-Rwd":
                reward = info["utility"]  # Use utility as reward
            elif self.feature == "Weighted-Usefulness-as-Rwd":
                reward = info["nb_applicable_jobs"] + info["utility"]  # Combine both metrics
            else:
                raise ValueError(f"Unknown feature type: {self.feature}")

        self.nb_recommendations += 1
        terminated = self.nb_recommendations == self.k

        return observation, reward, terminated, False, info


class EvaluateCallback(BaseCallback):
    """Callback for evaluating the RL model during training.
    
    This callback evaluates the model's performance at regular intervals during training.
    It calculates the average number of applicable jobs across all learners and logs
    the results to a file.
    
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
            with open(
                os.path.join(
                    self.eval_env.dataset.config["results_path"],
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
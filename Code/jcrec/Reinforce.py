import os
import json
import numpy as np
from time import process_time
from stable_baselines3 import DQN, A2C, PPO

from CourseRecEnv import CourseRecEnv, EvaluateCallback


class Reinforce:
    """Reinforcement Learning-based Course Recommendation System.
    
    This class implements a reinforcement learning approach for course recommendations
    using various RL algorithms from stable-baselines3 with mastery levels and clustering.
    
    The system trains an RL agent to recommend courses to learners with the goal of
    maximizing their job opportunities. The agent learns a policy that maps learner
    skill profiles to course recommendations, considering mastery levels and optional
    clustering-based reward adjustment.
    
    Features:
    - Support for multiple RL algorithms (DQN, A2C, PPO)
    - Mastery levels (1-3) for skills
    - Optional clustering-based reward adjustment
    - Comprehensive evaluation metrics
    
    Attributes:
        dataset: Dataset object containing learners, jobs, and courses data
        model_name (str): Name of the RL algorithm to use ('dqn', 'a2c', or 'ppo')
        k (int): Maximum number of course recommendations per learner
        threshold (float): Minimum matching score required for job applicability
        run (int): Run identifier for experiment tracking
        total_steps (int): Total number of training steps
        eval_freq (int): Frequency of model evaluation during training
    """
    
    def __init__(
        self, dataset, model, k, threshold, run, total_steps=1000, eval_freq=100
    ):  
        """Initialize the reinforcement learning recommendation system.
        
        Args:
            dataset: Dataset object containing the recommendation system data
            model (str): Name of the RL algorithm ('dqn', 'a2c', or 'ppo')
            k (int): Maximum number of course recommendations per learner
            threshold (float): Minimum matching score for job applicability
            run (int): Run identifier for experiment tracking
            total_steps (int, optional): Total training steps. Defaults to 1000.
            eval_freq (int, optional): Evaluation frequency. Defaults to 100.
        """
        self.dataset = dataset
        self.model_name = model
        self.k = k
        self.threshold = threshold
        self.run = run
        self.total_steps = total_steps
        self.eval_freq = eval_freq
        
        # Create the training and evaluation environments
        self.train_env = CourseRecEnv(dataset, threshold=self.threshold, k=self.k, is_training=True)
        self.eval_env = CourseRecEnv(dataset, threshold=self.threshold, k=self.k, is_training=False)
        self.get_model()
        
        # Check if model uses clustering based on config
        if self.train_env.use_clustering:  # Only use clustering if explicitly enabled
            self.all_results_filename = (
                f"all_{self.model_name}_k_{self.k}_total_steps_{self.total_steps}_clusters_auto_run_{run}.txt"
            )
            self.final_results_filename = (
                f"final_{self.model_name}_k_{self.k}_total_steps_{self.total_steps}_clusters_auto_run_{run}.json"
            )
        else:  # model without clustering
            self.all_results_filename = (
                f"all_{self.model_name}_k_{self.k}_total_steps_{self.total_steps}_run_{run}.txt"
            )
            self.final_results_filename = (
                f"final_{self.model_name}_k_{self.k}_total_steps_{self.total_steps}_run_{run}.json"
            )

        self.eval_callback = EvaluateCallback(
            self.eval_env,
            eval_freq=self.eval_freq,
            all_results_filename=self.all_results_filename,
        )

    def get_model(self):
        """Initialize the reinforcement learning model.
        
        Sets up the specified RL algorithm (DQN, A2C, or PPO) with default parameters.
        The model is configured to use a Multi-Layer Perceptron (MLP) policy.
        
        Supported algorithms:
        - DQN: Deep Q-Network for discrete action spaces
        - A2C: Advantage Actor-Critic for continuous action spaces
        - PPO: Proximal Policy Optimization for both discrete and continuous spaces
        """
        # on training env
        if self.model_name == "dqn":
            self.model = DQN(env=self.train_env, verbose=0, policy="MlpPolicy")
        elif self.model_name == "a2c":
            self.model = A2C(env=self.train_env, verbose=0, policy="MlpPolicy", device="cpu")
        elif self.model_name == "ppo":
            self.model = PPO(env=self.train_env, verbose=0, policy="MlpPolicy")

    def update_learner_profile(self, learner, course):
        """Updates the learner's profile with the skills and levels of the course.

        This method updates the learner's skill vector by taking the maximum of
        current skills and course-provided skills. This ensures that:
        1. Skills are never downgraded
        2. New skills are added
        3. Existing skills are maintained at their highest level

        Args:
            learner (np.ndarray): Current learner's skill vector
            course (np.ndarray): Course's skills array [required, provided]
            
        Returns:
            np.ndarray: Updated learner's skill vector
        """
        learner = np.maximum(learner, course[1])
        return learner

    def reinforce_recommendation(self):
        """Train and evaluate the RL model for course recommendations.
        
        This method:
        1. Calculates initial metrics:
           - Average learner attractiveness
           - Average number of applicable jobs
        2. Trains the RL model using the training environment
        3. Evaluates the model on all learners:
           - Generates course recommendations
           - Updates learner profiles
           - Tracks recommendation time
        4. Calculates final metrics:
           - New average learner attractiveness
           - New average number of applicable jobs
        5. Saves results:
           - Intermediate evaluation results to text file
           - Final metrics and recommendations to JSON file
        
        The results are saved in two files:
        - A text file with intermediate evaluation results during training
        - A JSON file with final metrics and recommendations for each learner
        """
        results = dict()

        avg_l_attrac_debut = self.dataset.get_avg_learner_attractiveness() #debut
        print(f"The average attractiveness of the learners is {avg_l_attrac_debut:.2f}")
        results["original_attractiveness"] = avg_l_attrac_debut

        avg_app_j_debut = self.dataset.get_avg_applicable_jobs(self.threshold) #debut
        print(f"The average nb of applicable jobs per learner is {avg_app_j_debut:.2f}")
        results["original_applicable_jobs"] = avg_app_j_debut

        # Train the model using train env
        self.model.learn(total_timesteps=self.total_steps, callback=self.eval_callback)# find the policy

        # Evaluate the model using eval env
        time_start = process_time()
        recommendations = dict()
        for i, learner in enumerate(self.dataset.learners):#run by row
            self.eval_env.reset(learner=learner) #initialize _agent_skills = learner if not NONE
            done = False
            index = self.dataset.learners_index[i]
            recommendation_sequence = []
            while not done:
                obs = self.eval_env._get_obs() #return _agent_skills which is current state
                # The self model was trained on historical data and has already learned a policy.
                action, _state = self.model.predict(obs, deterministic=True) #deterministic != transition probab in env
                # action is Recommended course index [0,99]action_space
                obs, reward, done, _, info = self.eval_env.step(action)
                if reward != -1:
                    recommendation_sequence.append(action.item())
            for course in recommendation_sequence:
                self.dataset.learners[i] = self.update_learner_profile(
                    learner, self.dataset.courses[course]
                )

            recommendations[index] = [
                self.dataset.courses_index[course_id]
                for course_id in recommendation_sequence
            ]

        time_end = process_time()
        avg_recommendation_time = (time_end - time_start) / len(self.dataset.learners)

        print(f"Average Recommendation Time: {avg_recommendation_time:.4f} seconds")
        results["avg_recommendation_time"] = avg_recommendation_time

        avg_l_attrac_fin = self.dataset.get_avg_learner_attractiveness() #fin
        print(f"The new average attractiveness of the learners is {avg_l_attrac_fin:.2f}")
        results["new_attractiveness"] = avg_l_attrac_fin

        avg_app_j_fin = self.dataset.get_avg_applicable_jobs(self.threshold)
        print(f"The new average nb of applicable jobs per learner is {avg_app_j_fin:.2f}")
        results["new_applicable_jobs"] = avg_app_j_fin

        results["recommendations"] = recommendations

        # Create branch directory if it doesn't exist
        branch_dir = os.path.join(self.dataset.config["results_path"], self.dataset.config["branch_name"])
        os.makedirs(branch_dir, exist_ok=True)
        
        # Create data directory for this branch
        data_dir = os.path.join(branch_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        json.dump(
            results,
            open(
                os.path.join(
                    data_dir,
                    self.final_results_filename,
                ),
                "w",
            ),
            indent=4,
        )

# Course Recommendation System with Mastery Levels

A reinforcement learning-based course recommendation system that helps learners acquire skills needed for jobs, taking into account skill mastery levels.

## Overview

This system recommends courses to learners based on their current skills and the skills required by available jobs. It uses reinforcement learning to optimize course recommendations, considering both skill acquisition and job applicability.

### Key Features

- **Mastery Level Integration**: Considers skill mastery levels (1-3) instead of binary skill possession
- **Multiple RL Algorithms**: Supports DQN, A2C, and PPO algorithms
- **Two Reward Models**:
  - Baseline: Uses number of applicable jobs as reward
  - Enhanced: Uses utility function considering both skill acquisition and job applicability
    - Usefulness-as-Rwd: Uses utility function as reward
    - Weighted-Usefulness-as-Rwd: Combines number of applicable jobs with utility
- **MLflow Integration**: Tracks experiments, metrics, and results

## System Components

### 1. Dataset Management (`Dataset.py`)
- Handles data loading and processing from JSON and CSV files
- Manages learner profiles, job requirements, and course information
- Processes skill mastery levels (1-3)
- Provides methods for:
  - Skill matching and gap analysis
  - Missing skills identification (both completely and partially missing)
  - Learner attractiveness calculation
  - Job applicability checking

### 2. Environment (`CourseRecEnv.py`)
- Implements Gymnasium environment for RL training
- Manages state representation (learner's skills with mastery levels)
- Handles course recommendations and skill updates
- Calculates rewards based on selected model:
  - Baseline: Number of applicable jobs
  - Usefulness-as-Rwd: Utility function value
  - Weighted-Usefulness-as-Rwd: Number of applicable jobs + Utility
- Implements course metrics (N1, N2, N3) for utility calculation

### 3. Matching Functions (`matchings.py`)
- Implements various matching algorithms:
  - Binary matching (skills exist/not exist)
  - Mastery level matching (considers skill levels)
  - Learner-job matching (considers mastery levels)
  - Learner-course matching:
    - Required skills matching
    - Provided skills matching
    - Overall course matching

### 4. Reinforcement Learning (`Reinforce.py`)
- Implements RL-based recommendation system
- Supports multiple algorithms (DQN, A2C, PPO)
- Handles model training and evaluation
- Tracks performance metrics:
  - Learner attractiveness
  - Number of applicable jobs
  - Recommendation time
  - Training progress

## Configuration

The system is configured through `run.yaml`:

```yaml
# Key Parameters
threshold: 0.8        # Minimum matching threshold for job eligibility
k: 3                 # Number of course recommendations
model: dqn           # RL algorithm (dqn/a2c/ppo)
baseline: false      # Use baseline or enhanced model
feature: "Weighted-Usefulness-as-Rwd"  # Reward feature type
level_3: true        # Use level 3 taxonomy
max_cv_skills: 15    # Maximum skills per learner
nb_courses: 100      # Number of courses in dataset
nb_jobs: 100         # Number of jobs in dataset
```

## Usage

1. Configure the system in `run.yaml`
2. Run the pipeline:
```bash
python Code/jcrec/pipeline.py --config Code/config/run.yaml
```

## Metrics

The system tracks several metrics:
- Original and new learner attractiveness
- Number of applicable jobs before and after recommendations
- Average recommendation time
- Training progress and evaluation results
- Course recommendation utility (for enhanced models)

## Results

Results are saved in two formats:
1. Text file (`all_*.txt`): Contains intermediate evaluation results during training
2. JSON file (`final_*.json`): Contains:
   - Final metrics
   - Course recommendations for each learner
   - Original and new attractiveness
   - Original and new applicable jobs

## Dependencies

- Python 3.x
- NumPy
- Gymnasium
- Stable-Baselines3
- MLflow
- PyYAML

## Project Structure

```
Code/
├── config/
│   └── run.yaml
├── jcrec/
│   ├── Dataset.py
│   ├── CourseRecEnv.py
│   ├── matchings.py
│   ├── Reinforce.py
│   └── pipeline.py
└── results/
    └── [output files]
```

## Notes

- The system uses level 3 taxonomy by default (configurable)
- Maximum 15 skills per learner profile
- Supports up to 100 courses and jobs in the dataset
- Uses MLflow for experiment tracking (http://127.0.0.1:8080)
- Course recommendations are limited to k courses per learner
- Skills are represented with mastery levels (1-3) instead of binary values
- Missing skills can be either completely missing or partially missing (lower mastery level)
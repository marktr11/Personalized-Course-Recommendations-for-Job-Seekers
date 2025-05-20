import json
import random

import pandas as pd
import numpy as np

from collections import defaultdict

import matchings


class Dataset: #modified class
    """Dataset class for the course recommendation system.
    
    This class handles data loading, processing, and analysis for the recommendation system.
    It manages three main types of data:
    - Learner profiles and their skills with mastery levels
    - Job requirements and their required skills with mastery levels
    - Course information including required and provided skills with mastery levels
    
    The class implements the Mastery-Levels approach, where skills are represented
    with mastery levels (1-3) instead of binary values.
    """
    # The Dataset class is used to load and store the data of the recommendation problem
    def __init__(self, config):
        self.config = config
        self.load_data()
        self.get_jobs_inverted_index()

    def __str__(self):
        # override the __str__ method to print the dataset
        return (
            f"Dataset with {len(self.learners)} learners, "
            f"{len(self.jobs)} jobs, "
            f"{len(self.courses)} courses and "
            f"{len(self.skills)} skills."
        )

    def load_data(self):
        """Load the data from the files specified in the config and store it in the class attributes.
        
        This method initializes the dataset by:
        1. Loading skills and their taxonomy
        2. Loading mastery level mappings
        3. Loading learner profiles with their skill mastery levels
        4. Loading job requirements with required skill mastery levels
        5. Loading courses with their required and provided skill mastery levels
        6. Creating subsamples if specified in config
        7. Making courses consistent by handling skill requirements
        """
        self.rng = random.Random(self.config["seed"])
        self.load_skills() 
        self.load_mastery_levels()
        self.load_learners() 
        self.load_jobs() 
        self.load_courses() 
        self.get_subsample()
        self.make_course_consistent()
        


    def load_skills(self):
        """Loads skills from a taxonomy file and creates mappings between skill IDs and indices.
        
        The method processes skills based on configuration:
        - If level_3 is True: Uses Type Level 3 taxonomy for broader skill categories
        - If level_3 is False: Uses unique_id for individual skills
        
        Attributes Modified:
            self.skills (set): Set of unique skill identifiers or level 3 types
            self.skills2int (dict): Dictionary mapping skill identifiers to integer indices
        """
        # load the skills from the taxonomy file
        self.skills = pd.read_csv(self.config["taxonomy_path"])

        # if level_3 is true, we only use the level 3 of the skill taxonomy, then we need to get the unique values in column Type Level 3
        ## Note: A single taxonomy skill may be shared across multiple skills. Using Level 3 taxonomy is preferred
        # as it maintains effective skill categorization. Levels 1 or 2 are too broad, resulting in overly general domains.
        if self.config["level_3"]:
            # get all the unique values in column Type Level 3
            level2int = {
                level: i for i, level in enumerate(self.skills["Type Level 3"].unique())
            }

            # make a dict from column unique_id to column Type Level 3
            skills_dict = dict(
                zip(self.skills["unique_id"], self.skills["Type Level 3"])
            )

            # map skills_dict values to level2int
            self.skills2int = {
                key: level2int[value] for key, value in skills_dict.items()
            }
            self.skills = set(self.skills2int.values())
            #print(level2int) #output : software and applications development and analysis : 0
            #print(skills_dict) #output : 1000: software and applications development and analysis
            #print(skills2int) #output : 1000: 0
        # if level_3 is false, we use the unique_id column as the skills
        else:
            self.skills = set(self.skills["unique_id"])
            self.skills2int = {skill: i for i, skill in enumerate(self.skills)}

    def load_mastery_levels(self):
        """Load the mastery level mappings from the configuration file.
        
        The mastery levels define how string-based skill levels (e.g., 'beginner', 'intermediate')
        are mapped to numerical values (1-3).
        
        Attributes Modified:
            self.mastery_levels (dict): Dictionary mapping string mastery levels to numerical values
        """
        self.mastery_levels = json.load(open(self.config["mastery_levels_path"]))

    def get_avg_skills(self, skill_list, replace_unk):
        """Calculate average mastery levels for a list of skills.
        
        Args:
            skill_list (list): List of (skill_id, mastery_level) tuples
            replace_unk (int): Value to replace unknown mastery levels (-1)
            
        Returns:
            dict: Dictionary mapping skill indices to their average mastery levels
                 Only includes skills with valid mastery levels
                 Mastery levels are rounded to nearest integer
        """
        avg_skills = defaultdict(list)
        for skill, mastery_level in skill_list:
            # if the mastery level is a string and is in the mastery levels, we replace it with the corresponding value, otherwise we do nothing and continue to the next skill
            if isinstance(mastery_level, str) and mastery_level in self.mastery_levels:
                mastery_level = self.mastery_levels[mastery_level]
                if mastery_level == -1:
                    mastery_level = replace_unk
                skill = self.skills2int[skill]  
                avg_skills[skill].append(mastery_level)
        # we take the average of the mastery levels for each skill because on our dataset we can have multiple mastery levels for the same skill
        for skill in avg_skills.keys():
            avg_skills[skill] = sum(avg_skills[skill]) / len(avg_skills[skill])
            avg_skills[skill] = round(avg_skills[skill])

        return avg_skills

    def get_base_skills(self,skill_list): #new feature
        """
        Convert a learner's list of type-4 skills to a unique set of type-3 base skills.

        Args:
            skill_list (list of tuples): Each tuple contains (skill_id, mastery_level),
                                        e.g., (1024, 'beginner').

        Returns:
            set: A set of base skill IDs (type-3) derived from the input skill list.
                The number of base skills may be less than or equal to the original list,
                due to mapping multiple type-4 skills to the same base skill.
        """
        base_skills = set()
        for skill, mastery_level in skill_list:
            # if the mastery level is a string and is in the mastery levels, we replace it with the corresponding value, otherwise we do nothing and continue to the next skill
            # we keep it to maintain consistency with the original version, which uses this condition.
            if isinstance(mastery_level, str) and mastery_level in self.mastery_levels:
                #eg. skill = 1024 , mastery_level = 'beginner'
                # mapping to an integer which is the id of taxonomy level
                # Mapping skills type 4 of learners to type 3, 
                # so the number of skills may be less than or equal to the original number of skills
                try:
                    base_skills.add(self.skills2int[skill])
                except KeyError:
                    continue
    

        return base_skills
    

    def load_learners(self,replace_unk=1): #### Function modified
        """Load learner profiles and their skill mastery levels.
        
        Args:
            replace_unk (int, optional): Value to replace unknown mastery levels. Defaults to 1.
            
        The method:
        1. Loads learner profiles from JSON
        2. Calculates average mastery levels for each skill
        3. Filters out learners with too many skills
        4. Creates a numpy array of learner skill mastery levels
        5. Maintains mappings between learner IDs and array indices
        """
        learners = json.load(open(self.config["cv_path"]))
        self.max_learner_skills = self.config["max_cv_skills"]
        self.learners_index = dict()

        # numpy array to store the learners skill proficiency levels with default value 0
        self.learners = np.zeros((len(learners), len(self.skills)), dtype=int)
        index = 0

        # fill the numpy array with the learners skill proficiency levels from the json file
        for learner_id, learner in learners.items():
            # Get average mastery levels for each skill
            learner_skills = self.get_avg_skills(learner, replace_unk)

            # if the number of skills is greater than the max_learner_skills, we skip the learner
            if len(learner_skills) > self.max_learner_skills:
                continue

            # we fill the numpy array with the averaged mastery levels
            for skill, level in learner_skills.items():
                self.learners[index][skill] = level

            self.learners_index[index] = learner_id
            self.learners_index[learner_id] = index

            index += 1

        # we update the learners numpy array with the correct number of rows
        self.learners = self.learners[:index]


    def load_jobs(self,replace_unk=3):
        """Load job requirements and their required skill mastery levels.
        
        Args:
            replace_unk (int, optional): Value to replace unknown mastery levels. Defaults to 3.
            
        The method:
        1. Loads job requirements from JSON
        2. Calculates average mastery levels for required skills
        3. Creates a numpy array of job skill requirements
        4. Maintains mappings between job IDs and array indices
        """
        jobs = json.load(open(self.config["job_path"]))
        self.jobs = np.zeros((len(jobs), len(self.skills)), dtype=int)
        self.jobs_index = dict()
        index = 0
        for job_id, job in jobs.items():
            self.jobs_index[index] = job_id
            self.jobs_index[job_id] = index

            # Get average mastery levels for each skill
            job_skills = self.get_avg_skills(job, replace_unk)

            for skill, level in job_skills.items():
                self.jobs[index][skill] = level
            index += 1

       

    def load_courses(self,replace_unk=2):
        """Load course information including required and provided skill mastery levels.
        
        Args:
            replace_unk (int, optional): Value to replace unknown mastery levels. Defaults to 2.
            
        The method:
        1. Loads course information from JSON
        2. Skips courses with no provided skills
        3. Calculates average mastery levels for:
           - Required skills (if any)
           - Provided skills
        4. Creates a numpy array of course skill requirements and provisions
        5. Maintains mappings between course IDs and array indices
        """
        courses = json.load(open(self.config["course_path"]))
        self.courses = np.zeros((len(courses), 2, len(self.skills)), dtype=int)
        self.courses_index = dict()
        index = 0
        for course_id, course in courses.items():
            # Skip courses with no provided skills
            if "to_acquire" not in course:
                continue
            
            self.courses_index[course_id] = index
            self.courses_index[index] = course_id

            # Get average mastery levels for provided skills
            provided_skills = self.get_avg_skills(course["to_acquire"], replace_unk)

            for skill, level in provided_skills.items():
                self.courses[index][1][skill] = level

            # Process required skills if they exist
            if "required" in course:
                required_skills = self.get_avg_skills(course["required"], replace_unk)

                for skill, level in required_skills.items():
                    self.courses[index][0][skill] = level

            index += 1  
        # update the courses numpy array with the correct number of rows
        self.courses = self.courses[:index]


    def get_subsample(self):
        """Create subsamples of the dataset based on configuration parameters.
        
        If specified in config, creates random subsamples of:
        - Learners (nb_cvs)
        - Jobs (nb_jobs)
        - Courses (nb_courses)
        
        Updates the corresponding numpy arrays and index mappings.
        """
        random.seed(self.config["seed"])
        if self.config["nb_cvs"] != -1:
            # get a random sample of self.config["nb_cvs"] of ids from 0 to len(self.learners)
            learners_ids = random.sample(
                range(len(self.learners)), self.config["nb_cvs"]
            )
            # update the learners numpy array and the learners_index dictionary with the sampled ids
            self.learners = self.learners[learners_ids]
            self.learners_index = {
                i: self.learners_index[index] for i, index in enumerate(learners_ids)
            }
            self.learners_index.update({v: k for k, v in self.learners_index.items()})
        if self.config["nb_jobs"] != -1:
            jobs_ids = random.sample(range(len(self.jobs)), self.config["nb_jobs"])
            self.jobs = self.jobs[jobs_ids]
            self.jobs_index = {
                i: self.jobs_index[index] for i, index in enumerate(jobs_ids)
            }
            self.jobs_index.update({v: k for k, v in self.jobs_index.items()})
        if self.config["nb_courses"] != -1:
            courses_ids = random.sample(
                range(len(self.courses)), self.config["nb_courses"]
            )
            self.courses = self.courses[courses_ids]
            self.courses_index = {
                i: self.courses_index[index] for i, index in enumerate(courses_ids)
            }
            self.courses_index.update({v: k for k, v in self.courses_index.items()})

    def make_course_consistent(self):
        """Make courses consistent by handling skill requirements and provisions.
        
        For each course:
        1. If a skill is both required and provided:
           - If provided level is 1, remove requirement
           - Otherwise, set requirement to (provided level - 1)
        2. Remove requirements for skills not provided by the course
        
        This ensures courses are logically consistent in their skill requirements.
        """
        for course in self.courses:
            for skill_id in range(len(self.skills)):
                required_level = course[0][skill_id]
                provided_level = course[1][skill_id]

                if provided_level != 0 and provided_level <= required_level:
                    if provided_level == 1:
                        course[0][skill_id] = 0
                    else:
                        course[0][skill_id] = provided_level - 1

                

    def get_jobs_inverted_index(self):
        """Create an inverted index mapping skills to jobs that require them.
        
        Creates a dictionary where:
        - Keys are skill indices
        - Values are sets of job indices that require that skill
        
        This index is used to efficiently find jobs requiring specific skills.
        """
        self.jobs_inverted_index = defaultdict(set)
        for i, job in enumerate(self.jobs):
            for skill, level in enumerate(job):
                if level > 0:
                    self.jobs_inverted_index[skill].add(i)

    def get_nb_applicable_jobs(self, learner, threshold):
        """Calculate number of jobs a learner is eligible for based on skill mastery levels.
        
        Args:
            learner (np.ndarray): Learner's skill vector with mastery levels
            threshold (float): Minimum matching score required for job eligibility
            
        Returns:
            int: Number of jobs where the learner's skills meet or exceed requirements
                 at the specified threshold
        """
        nb_applicable_jobs = 0
        jobs_subset = set()

        # get the index of the non zero elements in the learner array
        skills = np.nonzero(learner)[0]

        for skill in skills:
            if skill in self.jobs_inverted_index:
                jobs_subset.update(self.jobs_inverted_index[skill])
        for job_id in jobs_subset:
            matching = matchings.learner_job_matching(learner, self.jobs[job_id])
            if matching >= threshold:
                nb_applicable_jobs += 1
        return nb_applicable_jobs

    def get_avg_applicable_jobs(self, threshold):
        """Calculate average number of applicable jobs across all learners.
        
        Args:
            threshold (float): Minimum matching score required for job eligibility
            
        Returns:
            float: Average number of jobs that learners are eligible for
                 based on their skill mastery levels
        """
        avg_applicable_jobs = 0
        for learner in self.learners:
            avg_applicable_jobs += self.get_nb_applicable_jobs(learner, threshold)
        avg_applicable_jobs /= len(self.learners)
        return avg_applicable_jobs

    def get_all_enrollable_courses(self, learner, threshold):
        """Find all courses a learner can enroll in based on their skill mastery levels.
        
        Args:
            learner (np.ndarray): Learner's skill vector with mastery levels
            threshold (float): Minimum matching score required for course enrollment
            
        Returns:
            dict: Dictionary mapping course indices to their skill requirements and provisions
                 Only includes courses where:
                 - Learner meets required skill mastery levels
                 - Course provides at least one new skill
        """
        enrollable_courses = {}
        for i, course in enumerate(self.courses):
            required_matching = matchings.learner_course_required_matching(
                learner, course
            )
            provided_matching = matchings.learner_course_provided_matching(
                learner, course
            )
            if required_matching >= threshold and provided_matching < 1.0:
                enrollable_courses[i] = course
        return enrollable_courses

    def get_learner_acquired_skills(self, learner):
        """Get the skills that a learner currently possesses with their mastery levels.
        
        Args:
            learner (np.ndarray): Learner's skill vector where each value indicates
                                the mastery level of that skill (0 means no skill).
            
        Returns:
            dict: Dictionary mapping skill indices to their mastery levels.
                 Only includes skills with mastery level > 0.
        """
        # Get all non-zero skills and their mastery levels
        skills = np.nonzero(learner)[0]
        return {skill: learner[skill] for skill in skills}

    def get_learner_missing_skills(self, learner):
        """Identify skills that a learner needs to acquire or improve to be eligible for jobs.
        
        This function analyzes the gap between a learner's current skills and
        the skills required by available jobs. It considers two types of missing skills:
        1. Completely missing skills (not in learner's skill set)
        2. Partially missing skills (in learner's skill set but with lower mastery level)
        
        The function examines ALL jobs in the dataset to:
        - Collect all skills required by any job
        - For each skill, track the highest mastery level required
        - For partially missing skills, track the largest gap between required and current level
        
        Args:
            learner (np.ndarray): Learner's skill vector where each value indicates
                                the mastery level of that skill (0 means no skill).
            
        Returns:
            dict: Dictionary mapping skill indices to their required mastery levels.
                 For completely missing skills, the value is the required mastery level.
                 For partially missing skills, the value is the required mastery level
                 (which is higher than the current level).
                 
        Example:
            If learner has skill 1 with level 1 and skill 3 with level 1:
            - Job 1 requires skill 1 (level 3) and skill 3 (level 2)
            - Job 2 requires skill 1 (level 2) and skill 2 (level 1)
            - Job 3 requires skill 0 (level 2) and skill 3 (level 3)
            
            Result will be:
            {
                0: 2,  # completely missing skill
                1: 2,  # partially missing (current: 1, required: 3, gap: 2)
                2: 1,  # completely missing skill
                3: 3   # partially missing (current: 1, required: 3, gap: 2)
            }
        """
        # Get learner's current skills and their mastery levels
        learner_skills = self.get_learner_acquired_skills(learner)
        
        # Dictionary to store missing skills and their required mastery levels
        # Key: skill index
        # Value: highest required mastery level from all jobs
        missing_skills = {}
        
        # Check each job's requirements
        for job in self.jobs:
            # Get skills required by this job and their mastery levels
            # Only include skills with mastery level > 0
            job_skills = {skill: level for skill, level in enumerate(job) if level > 0}
            
            # Check each required skill in this job
            for skill, required_level in job_skills.items():
                if skill not in learner_skills:
                    # Case 1: Completely missing skill
                    # Add to missing_skills if:
                    # - Skill not yet in missing_skills, OR
                    # - This job requires a higher level than previously recorded
                    if skill not in missing_skills or required_level > missing_skills[skill]:
                        missing_skills[skill] = required_level
                elif learner_skills[skill] < required_level:
                    # Case 2: Partially missing skill (learner has it but level too low)
                    # Calculate the gap between required and current level
                    gap = required_level - learner_skills[skill]
                    
                    # Update missing_skills if:
                    # - Skill not yet in missing_skills, OR
                    # - This job has a larger gap than previously recorded
                    if skill not in missing_skills or gap > (missing_skills[skill] - learner_skills[skill]):
                        missing_skills[skill] = required_level
        
        return missing_skills

    def get_learner_missing_skills_with_frequency(self, learner):
        """Analyze the frequency of missing skills in job requirements.
        
        This function extends get_learner_missing_skills by adding frequency analysis.
        It helps prioritize which missing skills are most in demand in the job market.
        
        Args:
            learner (np.ndarray): Learner's skill vector where 1 indicates
                                possession of a skill and 0 indicates absence.
            
        Returns:
            dict: Dictionary mapping skill indices to their frequency in job requirements.
                 Higher frequency indicates higher demand for that skill in the job market.
        """
        # Get learner's current skills
        learner_skills = self.get_learner_acquired_skills(learner)
        
        # Count frequency of each skill in job requirements
        skill_frequency = defaultdict(int)
        for job in self.jobs:
            job_skills = set(np.nonzero(job)[0])
            for skill in job_skills:
                if skill not in learner_skills:
                    skill_frequency[skill] += 1
        
        return dict(skill_frequency)

    def get_learner_attractiveness(self, learner):
        """Calculate a learner's attractiveness in the job market.
        
        This function measures how many jobs require at least one of the
        learner's current skills. It provides a basic measure of the learner's
        marketability based on their current skill set.
        
        Args:
            learner (np.ndarray): Learner's skill vector where 1 indicates
                                possession of a skill and 0 indicates absence.
            
        Returns:
            int: Number of jobs that require at least one of the learner's skills.
        """
        attractiveness = 0
        skills = np.nonzero(learner)[0]
        
        for skill in skills:
            if skill in self.jobs_inverted_index:
                attractiveness += len(self.jobs_inverted_index[skill])
        return attractiveness

    def get_avg_learner_attractiveness(self):
        """Calculate the average attractiveness across all learners.
        
        This function provides a measure of the overall marketability of
        the learner population based on their current skill sets.
        
        Returns:
            float: The average number of jobs that require at least one
                  of each learner's skills.
        """
        attractiveness = 0
        for learner in self.learners:
            attractiveness += self.get_learner_attractiveness(learner)
        attractiveness /= len(self.learners)
        return attractiveness

    # def get_learner_base_skills(self, learner):
    #     """Get the base skills (indices) that a learner has.
        
    #     Args:
    #         learner (np.ndarray): Learner's skill vector
            
    #     Returns:
    #         set: Set of skill indices that the learner has (value = 1)
    #     """
    #     return set(np.nonzero(learner)[0])

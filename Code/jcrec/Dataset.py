import json
import random

import pandas as pd
import numpy as np

from collections import defaultdict

import matchings


class Dataset: #modified class : skip expertise
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
        """Load the data from the files specified in the config and store it in the class attributes"""
        self.rng = random.Random(self.config["seed"])
        self.load_skills() 
        self.load_mastery_levels()
        self.load_learners() #modify
        self.load_jobs() #modify
        self.load_courses() #modify
        self.get_subsample()
        if self.config["exp_feature"] == "courses-consistency":
           self.make_course_consistent()
        


    def load_skills(self):
        """
        Loads skills from a taxonomy file into the instance, processes them based on configuration,
        and creates a mapping of skills to integer indices.

        The method reads a CSV file specified in the configuration and processes the skills
        either by extracting unique values from the 'Type Level 3' column (if level_3 is True)
        or using the 'unique_id' column (if level_3 is False). It populates `self.skills` with
        a set of skills and `self.skills2int` with a dictionary mapping skills to integer indices.

        Attributes Modified:
            self.skills (set): A set of unique skill identifiers or level 3 types.
            self.skills2int (dict): A dictionary mapping skill identifiers to integer indices.

        Raises:
            FileNotFoundError: If the taxonomy file path in `self.config["taxonomy_path"]` is invalid.
            KeyError: If required columns ('unique_id' or 'Type Level 3') are missing in the CSV file.
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
        """Load the mastery levels from the file specified in the config and store it in the class attribute"""
        self.mastery_levels = json.load(open(self.config["mastery_levels_path"]))

    def get_avg_skills(self, skill_list, replace_unk):
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
        """Load the learners from the file specified in the config and store it in the class attribute

        Args:
            replace_unk (int, optional): The value to replace the unknown mastery levels. Defaults to 1.
        """
        learners = json.load(open(self.config["cv_path"]))
        self.max_learner_skills = self.config["max_cv_skills"]
        self.learners_index = dict()

        # numpy array to store the learners skill proficiency levels with default value 0
        self.learners = np.zeros((len(learners), len(self.skills)), dtype=int)
        index = 0

        # fill the numpy array with the learners skill proficiency levels from the json file
        for learner_id, learner in learners.items():

            if self.config["feature"] == "original":
                learner_skills = self.get_avg_skills(learner, replace_unk)
            else :
                learner_base_skills = self.get_base_skills(learner) #remove expertise
                learner_skills = {skill: 1 for skill in learner_base_skills}
            

            # if the number of skills is greater than the max_learner_skills, we skip the learner
            if len(learner_skills) > self.max_learner_skills:
                continue

            # we fill the numpy array with the averaged mastery levels
            for skill, level in learner_skills.items():
                self.learners[index][skill] = level

            self.learners_index[index] = learner_id
            self.learners_index[learner_id] = index #????? why 

            index += 1

        # we update the learners numpy array with the correct number of rows
        self.learners = self.learners[:index]

    def load_jobs(self,replace_unk=3): #### Function modified
        """Load the jobs from the file specified in the config and store it in the class attribute

        Args:
            replace_unk (int, optional): The value to replace the unknown mastery levels. Defaults to 3.
        """
        jobs = json.load(open(self.config["job_path"]))
        self.jobs = np.zeros((len(jobs), len(self.skills)), dtype=int)
        self.jobs_index = dict()
        index = 0
        for job_id, job in jobs.items():
            self.jobs_index[index] = job_id
            self.jobs_index[job_id] = index

            if self.config["feature"] == "original":
                job_skills = self.get_avg_skills(job, replace_unk)
            else :
                job_base_skills = self.get_base_skills(job) #remove expertise
                job_skills = {skill: 1 for skill in job_base_skills}
            

            for skill, level in job_skills.items():
                self.jobs[index][skill] = level
            index += 1

    def load_courses(self,replace_unk=2): #### Function modified
        """Load the courses from the file specified in the config and store it in the class attribute

        Args:
            replace_unk (int, optional): The value to replace the unknown mastery levels. Defaults to 2.
        """
        courses = json.load(open(self.config["course_path"]))
        self.courses = np.zeros((len(courses), 2, len(self.skills)), dtype=int)
        self.courses_index = dict()
        index = 0
        for course_id, course in courses.items():
            # if the course does not provide any skills, we skip it
            #process required skills
            if "to_acquire" not in course:
                continue

            self.courses_index[course_id] = index
            self.courses_index[index] = course_id

            if self.config["feature"] == "original":
                provided_skills = self.get_avg_skills(course["to_acquire"], replace_unk)
            else :
                provided_base_skills = self.get_base_skills(course["to_acquire"]) #remove expertise
                provided_skills = {skill: 1 for skill in provided_base_skills}

            for skill,level in provided_skills.items():
                self.courses[index][1][skill] = level
            
            #process required skills
            if "required" in course:
                if self.config["feature"] == "original":
                   required_skills = self.get_avg_skills(course["to_acquire"], replace_unk)
                else :
                   required_base_skills = self.get_base_skills(course["required"]) #remove expertise
                   required_skills = {skill: 1 for skill in required_base_skills}
                   
                for skill,level in required_skills.items():
                    self.courses[index][0][skill] = level

            index += 1
        # update the courses numpy array with the correct number of rows
        self.courses = self.courses[:index]

    def get_subsample(self):
        """Get a subsample of the dataset based on the config parameters"""
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
        """Make the courses consistent by removing the skills that are provided and required at the same time.
        In binary case (only care about having/not having skills), if a course both requires and provides a skill,
        we remove the requirement since the learner can learn that skill from the course."""
        for course in self.courses:
            for skill_id in range(len(self.skills)):
                required_level = course[0][skill_id]
                provided_level = course[1][skill_id]
                # If the course both requires and provides the skill (both levels > 0)
                if provided_level > 0 and required_level > 0:
                    # Remove the requirement since the skill can be learned from the course
                    course[0][skill_id] = 0

    def get_jobs_inverted_index(self):
        """Get the inverted index for the jobs. The inverted index is a dictionary that maps the skill to the jobs that require it"""
        self.jobs_inverted_index = defaultdict(set)
        for i, job in enumerate(self.jobs):
            for skill, level in enumerate(job):
                if level > 0:
                    self.jobs_inverted_index[skill].add(i)

    def get_nb_applicable_jobs(self, learner, threshold):
        """Get the number of applicable jobs for a learner

        Args:
            learner (list): list of skills and mastery level of the learner
            threshold (float): the threshold for the matching

        Returns:
            int: the number of applicable jobs
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
        """Get the average number of applicable jobs for all the learners

        Args:
            threshold (float): the threshold for the matching

        Returns:
            float: the average number of applicable jobs
        """
        avg_applicable_jobs = 0
        for learner in self.learners:
            avg_applicable_jobs += self.get_nb_applicable_jobs(learner, threshold)
        avg_applicable_jobs /= len(self.learners)
        return avg_applicable_jobs

    def get_all_enrollable_courses(self, learner, threshold):
        """Get all the enrollable courses for a learner

        Args:
            learner (list): list of skills and mastery level of the learner
            threshold (float): the threshold for the matching

        Returns:
            dict: dictionary of enrollable courses
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

    def get_learner_attractiveness(self, learner):
        """Get the attractiveness of a learner

        Args:
            learner (list): list of skills and mastery level of the learner

        Returns:
            int: number of jobs that require at least one of the learner's skills
        """
        attractiveness = 0

        # get the index of the non zero elements in the learner array
        skills = np.nonzero(learner)[0]

        for skill in skills:
            if skill in self.jobs_inverted_index:
                attractiveness += len(self.jobs_inverted_index[skill])
        return attractiveness

    def get_avg_learner_attractiveness(self):
        """Get the average attractiveness of all the learners

        Returns:
            float: the average attractiveness of the learners
        """
        attractiveness = 0
        for learner in self.learners:
            attractiveness += self.get_learner_attractiveness(learner)
        attractiveness /= len(self.learners)
        return attractiveness

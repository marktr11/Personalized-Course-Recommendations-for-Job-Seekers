import numpy as np
from typing import Tuple, Union, Dict, Set


def matching_binary(level1: np.ndarray, level2: np.ndarray) -> float:
    """
    Compute the matching score between two skill vectors in binary case.
    Only checks if skills exist (non-zero) in both vectors, regardless of their levels.

    Args:
        level1 (np.ndarray): An array of skills for the first entity (e.g., learner).
        level2 (np.ndarray): An array of skills for the second entity (e.g., job or course).

    Returns:
        float: A matching score between 0 and 1, where:
            - 1.0 means all required skills exist in level1 (perfect match)
            - 0.0 means none of the required skills exist in level1
            - Values in between represent the proportion of required skills that exist
            - If level2 has no skills (all zeros), returns -1.0 (invalid case)

    Raises:
        ValueError: If input arrays have different shapes
    """
    if level1.shape != level2.shape:
        raise ValueError("Input arrays must have the same shape")

    # Get indices of non-zero elements in both arrays
    skills1 = set(np.nonzero(level1)[0])
    skills2 = set(np.nonzero(level2)[0])
    
    # If no skills are required/provided, return -1.0 (invalid case)
    if not skills2:
        return -1.0
        
    # Count how many skills exist in level1
    matching_skills = len(skills1.intersection(skills2))
    
    # Return the proportion of required skills that exist
    return matching_skills / len(skills2)

def matching(level1, level2):
    """
    Compute the matching score between two skill vectors in original case.
    Takes into account the mastery levels of skills.

    Args:
        level1 (np.ndarray): An array of skills for the first entity (e.g., learner).
        level2 (np.ndarray): An array of skills for the second entity (e.g., job or course).

    Returns:
        float: A matching score between 0 and 1, where:
            - 1.0 means perfect match (all required skills at required levels)
            - 0.0 means no match
            - Values in between represent partial matches based on skill levels
    """
    # get the minimum of the two arrays
    minimum_skill = np.minimum(level1, level2)

    # get the indices of the non zero elements of the job skill levels
    nonzero_indices = np.nonzero(level2)[0]

    # divide the minimum by the job skill levels on the non zero indices
    matching = minimum_skill[nonzero_indices] / level2[nonzero_indices]

    # sum the result and divide by the number of non zero job skill levels
    matching = np.sum(matching) / np.count_nonzero(level2)

    return matching

def learner_job_matching(learner: np.ndarray, job: np.ndarray) -> float:
    """
    Compute the matching score between a learner and a job.

    Args:
        learner (np.ndarray): Learner's skill vector where 1 indicates
                            possession of a skill and 0 indicates absence.
        job (np.ndarray): Job's required skills vector where 1 indicates
                         required skill and 0 indicates not required.

    Returns:
        float: Matching score between 0 and 1, where:
            - 1.0 means the learner has all required skills for the job
            - 0.0 means the learner has none of the required skills
            - Values in between represent partial matches
    """
    # Check if one of the arrays is empty
    if not (np.any(job) and np.any(learner)):
        return 0.0

    return matching(learner, job)

def learner_course_required_matching(learner, course):
    """
    Compute the matching score between a learner and a course's required skills.
    Always uses original matching as required skills have mastery levels.

    Args:
        learner (np.ndarray): Learner's skill vector
        course (np.ndarray): Course's skills array [required, provided]

    Returns:
        float: Matching score between 0 and 1
    """
    required_course = course[0] #required skills

    # check if the course has no required skills and return 1
    if not np.any(required_course): # not( true if at least one element is not 0 )
        return 1.0

    return matching(learner, required_course)

def learner_course_provided_matching(learner: np.ndarray, course: np.ndarray) -> float:
    """
    Compute the matching score between a learner and a course's provided skills.
    This measures how many of the course's provided skills the learner already has.

    Args:
        learner (np.ndarray): Learner's skill vector where 1 indicates
                            possession of a skill and 0 indicates absence.
        course (np.ndarray): Course's skills array [required, provided] where
                            provided skills are in the second dimension.

    Returns:
        float: Matching score between 0 and 1, where:
            - 1.0 means the learner already has all skills the course provides
            - 0.0 means the learner has none of the skills the course provides
            - Values in between represent partial matches
    """
    provided_course = course[1]  # provided skills
    return matching(learner, provided_course)



def learner_course_matching(learner: np.ndarray, course: np.ndarray) -> float:
    """
    Compute the overall matching score between a learner and a course.
    This is used to measure user-course relevantness.

    The score is calculated as: required_matching * (1 - provided_matching)
    This formula ensures that:
    - Courses that provide new skills (low provided_matching) are preferred
    - Courses that the learner is qualified for (high required_matching) are preferred

    Args:
        learner (np.ndarray): Learner's skill vector where 1 indicates
                            possession of a skill and 0 indicates absence.
        course (np.ndarray): Course's skills array [required, provided] where
                            required skills are in the first dimension and
                            provided skills are in the second dimension.

    Returns:
        float: Overall matching score between 0 and 1, where higher values
               indicate better course recommendations for the learner.
    """
    # Get the required and provided matchings
    required_matching = learner_course_required_matching(learner, course)
    provided_matching = learner_course_provided_matching(learner, course)

    return required_matching * (1 - provided_matching)  # user-course relevantness

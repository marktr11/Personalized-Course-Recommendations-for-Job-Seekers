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
    Compute the matching score between two skill vectors considering mastery levels.
    Calculates how well the mastery levels in level1 match the required levels in level2.

    Args:
        level1 (np.ndarray): An array of skills with mastery levels for the first entity (e.g., learner).
        level2 (np.ndarray): An array of required skills with mastery levels for the second entity (e.g., job or course).

    Returns:
        float: A matching score between 0 and 1, where:
            - 1.0 means perfect match (all required skills at or above required levels)
            - 0.0 means no match
            - Values in between represent partial matches based on skill levels
            - For each skill, the score is min(level1, level2) / level2
            - Final score is average of all skill scores
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
    Compute the matching score between a learner and a job based on mastery levels.

    Args:
        learner (np.ndarray): Learner's skill vector where each value indicates
                            the mastery level of that skill (0 means no skill).
        job (np.ndarray): Job's required skills vector where each value indicates
                         the required mastery level (0 means not required).

    Returns:
        float: Matching score between 0 and 1, where:
            - 1.0 means the learner has all required skills at or above required levels
            - 0.0 means either:
              * The learner has none of the required skills, or
              * Either the learner or job has no skills
            - Values in between represent partial matches based on mastery levels
    """
    # Check if one of the arrays is empty
    if not (np.any(job) and np.any(learner)):
        return 0.0

    return matching(learner, job)

def learner_course_required_matching(learner, course):
    """
    Compute the matching score between a learner and a course's required skills.
    Always uses mastery level matching as required skills have mastery levels.

    Args:
        learner (np.ndarray): Learner's skill vector where each value indicates
                            the mastery level of that skill (0 means no skill).
        course (np.ndarray): Course's skills array [required, provided] where
                            required skills are in the first dimension with mastery levels.

    Returns:
        float: Matching score between 0 and 1, where:
            - 1.0 means either:
              * The course has no required skills, or
              * The learner meets all required skill mastery levels
            - 0.0 means the learner meets none of the required skill levels
            - Values in between represent partial matches based on mastery levels
    """
    required_course = course[0] #required skills

    # check if the course has no required skills and return 1
    if not np.any(required_course): # not( true if at least one element is not 0 )
        return 1.0

    return matching(learner, required_course)

def learner_course_provided_matching(learner: np.ndarray, course: np.ndarray) -> float:
    """
    Compute the matching score between a learner and a course's provided skills.
    This measures how many of the course's provided skills the learner already has
    at the same or higher mastery level.

    Args:
        learner (np.ndarray): Learner's skill vector where each value indicates
                            the mastery level of that skill (0 means no skill).
        course (np.ndarray): Course's skills array [required, provided] where
                            provided skills are in the second dimension with mastery levels.

    Returns:
        float: Matching score between 0 and 1, where:
            - 1.0 means the learner already has all provided skills at or above the course's levels
            - 0.0 means the learner has none of the provided skills at required levels
            - Values in between represent partial matches based on mastery levels
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
        learner (np.ndarray): Learner's skill vector where each value indicates
                            the mastery level of that skill (0 means no skill).
        course (np.ndarray): Course's skills array [required, provided] where:
                            - First dimension contains required skills with mastery levels
                            - Second dimension contains provided skills with mastery levels

    Returns:
        float: Overall matching score between 0 and 1, where:
            - Higher values indicate better course recommendations
            - Score is higher when:
              * Learner meets required skill levels (high required_matching)
              * Course provides new skills (low provided_matching)
    """
    # Get the required and provided matchings
    required_matching = learner_course_required_matching(learner, course)
    provided_matching = learner_course_provided_matching(learner, course)

    return required_matching * (1 - provided_matching)  # user-course relevantness

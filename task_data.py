#
#   task_data.py
#   Defines all synthetic data from the assignment description as dicts
#   and contains the common cost function
#

tasks = {
    1: {'estimated_time': 4, 'difficulty': 3, 'deadline': 8,  'skill': 'A'},
    2: {'estimated_time': 6, 'difficulty': 5, 'deadline': 12, 'skill': 'B'},
    3: {'estimated_time': 2, 'difficulty': 2, 'deadline': 6,  'skill': 'A'},
    4: {'estimated_time': 5, 'difficulty': 4, 'deadline': 10, 'skill': 'C'},
    5: {'estimated_time': 3, 'difficulty': 1, 'deadline': 7,  'skill': 'A'},
    6: {'estimated_time': 8, 'difficulty': 6, 'deadline': 15, 'skill': 'B'},
    7: {'estimated_time': 4, 'difficulty': 3, 'deadline': 9,  'skill': 'C'},
    8: {'estimated_time': 7, 'difficulty': 5, 'deadline': 14, 'skill': 'B'},
    9: {'estimated_time': 2, 'difficulty': 2, 'deadline': 5,  'skill': 'A'},
    10: {'estimated_time': 6, 'difficulty': 4, 'deadline': 11, 'skill': 'C'},
}

employees = {
    1: {'available_hours': 10, 'skill_level': 4, 'skills': {'A', 'C'}},
    2: {'available_hours': 12, 'skill_level': 6, 'skills': {'A', 'B', 'C'}},
    3: {'available_hours': 8, 'skill_level': 3, 'skills': {'A'}},
    4: {'available_hours': 15, 'skill_level': 7, 'skills': {'B', 'C'}},
    5: {'available_hours': 9, 'skill_level': 5, 'skills': {'A', 'C'}},
}
# sets are used for employee skills as they should not have duplicates

# objective function definition
ALPHA = 0.20
BETA = 0.20
DELTA = 0.20
SIGMA = 0.20
GAMMA = 0.20

def calculate_cost_function(overload_penalty,
                            skill_mismatch_penalty,
                            difficulty_violation_penalty,
                            deadline_violation_penalty,
                            unique_assignment_violation_penalty):
    
    return round((ALPHA*overload_penalty
            + BETA*skill_mismatch_penalty
            + DELTA*difficulty_violation_penalty
            + GAMMA*deadline_violation_penalty
            + SIGMA*unique_assignment_violation_penalty), 2)
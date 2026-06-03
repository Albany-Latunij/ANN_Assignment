#
#   genetic_algorithm.py
#   Contains all functions required for applying a genetic algoritm
#   to the task data 
#

import random
import psutil
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from task_data import tasks, employees, calculate_cost_function

np.random.seed(28)
random.seed(28)

# Generate a chromosome through randomly selecting employee numbers
def generate_chromosome(num_tasks, num_employees):    
    # chromosome is an 1D array of 10 valid task-assignments
    # eg if the 2nd element is 5, task 2 from the tasks dict has been assigned to employee 5
    chromosome = np.zeros(num_tasks, dtype=int)
    
    # each task is only assigned to one employee on intial generation
    for task in range(num_tasks):
        # chosen_employee is the gene
        chosen_employee = random.randint(1, num_employees)
        chromosome[task] = chosen_employee
    
    return chromosome

# Run the generate_chromosome function for the desired population size
def generate_population(population_size):
    num_tasks = len(tasks)
    num_employees = len(employees)
    
    #numpy arrays are used instead of lists for efficiency
    population = np.zeros((population_size, num_tasks), dtype=int)
    
    for i in range(population_size):
        chromosome = generate_chromosome(num_tasks, num_employees)
        population[i] = chromosome
    
    return population

# Sort assigned tasks by time for calculating the deadline penalty
def sort_tasks_by_time(assigned_tasks_numbers):
    unsorted_tasks = [tasks[assigned_task] for assigned_task in assigned_tasks_numbers]
    # Lambda usage adapted from https://www.w3schools.com/python/python_lambda.asp
    # sorts the list of tasks based on their estimated time
    sorted_tasks = sorted(unsorted_tasks, key = (lambda task: task["estimated_time"]))
    
    return sorted_tasks

# Find the cost of each chromosome usng the cost_function from task_data.py
def evaluate_chromosome_fitness(chromosome):
    num_employees = len(employees)
    
    chromosome_overload_penalty = 0
    chromosome_skill_mismatch_penalty = 0
    chromosome_difficulty_penalty = 0
    chromosome_deadline_penalty = 0
    chromosome_unique_assignment_penalty = 0
    
    constraint_violations = 0
    
    # sort tasks based on which employee they are assigned to
    employee_tasks = {(i + 1) : [] for i in range(num_employees)}
    already_assigned_tasks = []
    
    # check for duplicate task assignments
    # (impossible in the current chromosome format)
    for task, employee in enumerate(chromosome):
        task_number = task + 1
        employee_tasks[employee].append(task_number)
        
        if task_number in already_assigned_tasks:
            chromosome_unique_assignment_penalty += 1
        already_assigned_tasks.append(task_number)
    
    # loop through the dictionary of employees' assigned tasks to evaluate the chromosome
    for employee_id, assigned_tasks_numbers in employee_tasks.items():
        employee_hours = employees[employee_id]["available_hours"]
        employee_skill_level = employees[employee_id]["skill_level"]
        employee_skills = employees[employee_id]["skills"]
        
        sorted_tasks = sort_tasks_by_time(assigned_tasks_numbers)
        
        total_tasks_time = 0
        
        for task in sorted_tasks:
            # evaluating the skill level constraint
            chromosome_difficulty_penalty += max(0, task["difficulty"] - employee_skill_level)
           
            # evaluating the specialised skill matching constraint
            if task["skill"] not in employee_skills:
                chromosome_skill_mismatch_penalty += 1
            
            # calculating the cumulative task time
            task_time = task["estimated_time"]
            total_tasks_time += task_time
            
            # evaluating the deadline consideration penalty
            chromosome_deadline_penalty += max(0, total_tasks_time - task["deadline"])
        
        # evaluating the capacity constraint
        chromosome_overload_penalty += max(0, total_tasks_time - employee_hours)  
        
    # checks if any constraint violations have occured in the chromosome
    if chromosome_unique_assignment_penalty > 0:
        constraint_violations += 1
    if chromosome_difficulty_penalty > 0:
        constraint_violations += 1
    if chromosome_skill_mismatch_penalty > 0:
        constraint_violations += 1
    if chromosome_deadline_penalty > 0:
        constraint_violations += 1
    if chromosome_overload_penalty > 0:
        constraint_violations += 1
        
    # calculates the cost of the chromosome
    chromosome_cost = calculate_cost_function(
        chromosome_overload_penalty,
        chromosome_skill_mismatch_penalty,
        chromosome_difficulty_penalty,
        chromosome_deadline_penalty,
        chromosome_unique_assignment_penalty
    )
    
    return chromosome_cost, constraint_violations

# Run the evaluate_chromosome_fitness function for every chromosome
def evaluate_population_fitness(population, population_size):
    chromosome_fitnesses = np.zeros(population_size)
    chromosome_violations = np.zeros(population_size)
    
    fittest_index = 0
    lowest_cost = float("inf")
    
    # evaluates the fitness of each chromosome and keeps track of the fittest chromosome
    for i in range(population_size):
        fitness, violations = evaluate_chromosome_fitness(population[i])
        chromosome_fitnesses[i] = fitness
        chromosome_violations[i] = violations
        
        if fitness < lowest_cost:
            fittest_index = i
            lowest_cost = fitness
    
    # Calculate key characteristics of the population fitnesses for later graphing
    population_characteristics = {
        "min": round(float(np.min(chromosome_fitnesses)), 2),
        "average": round(float(np.average(chromosome_fitnesses)), 2),
        "deviation": round(float(np.std(chromosome_fitnesses)), 2),
        # tracks the average number of constraint violations per chromosome
        "average_violations": np.average(chromosome_violations)
    }
    
    return chromosome_fitnesses, population_characteristics, fittest_index

# Returns True or False depending on if a random number is less than the crossover rate
def should_crossover(crossover_percentage):
    return random.random() < crossover_percentage

# Select k_size random chromosomes and choose the best one as the parent for the next generation
def tournament_selection(population, population_size, fitnesses, k_size):
    # randomly select indices for the tournament selection
    random_chromosome_indices = np.random.choice(population_size, k_size, replace=False)
    # returns the index of the selected chromosome with the lowest cost
    best_selected_index = np.argmin(fitnesses[random_chromosome_indices])
    # get the best chromosome index from the best selected indices
    best_chromosome_index = random_chromosome_indices[best_selected_index]
    return population[best_chromosome_index]

# Create the new population for the next generation through reproducing and performing crossovers
def crossover_population(population, population_size, chromosome_fitnesses, crossover_percentage, elitism_percentage, k_size):
    num_tasks = len(tasks)
    
    new_population = np.zeros((population_size, num_tasks), dtype=int)
    
    # Sort the population from best fitness to least (lowest to highest cost function)
    # inspired from https://www.geeksforgeeks.org/python/how-to-get-the-indices-of-the-sorted-array-using-numpy-in-python/
    sorted_fitnesses_indices = np.argsort(chromosome_fitnesses)
    sorted_population = population[sorted_fitnesses_indices]
    
    # inspired from https://algorithmafternoon.com/genetic/elitist_genetic_algorithm/
    # Copy over the best performing chromosomes directly into the new population
    num_elitists = int(elitism_percentage * population_size)
    elistists = population[sorted_fitnesses_indices][:num_elitists]
    
    new_population[:num_elitists] = elistists
    
    # While loop is used to account for odd sized populations
    # Loop through the remaining empty populationa and populate through tournament selection
    i = num_elitists
    while i < population_size:
        # if there is one slot left to fill, copy a parent through tournament selection
        if i == population_size - 1:
            parent = tournament_selection(population, population_size, chromosome_fitnesses, k_size)
            new_population[i] = parent.copy()
        else:
            # choose the parents through tournament selection
            parent_1 = tournament_selection(population, population_size, chromosome_fitnesses, k_size)
            parent_2 = tournament_selection(population, population_size, chromosome_fitnesses, k_size)
            
            # crossover genes
            if should_crossover(crossover_percentage):
                crossover_point = random.randint(1, num_tasks - 1)
                child_1 = np.concatenate((parent_1[:crossover_point], parent_2[crossover_point:]))
                child_2 = np.concatenate((parent_2[:crossover_point], parent_1[crossover_point:]))
            else:
                child_1 = parent_1.copy()
                child_2 = parent_2.copy()
                
            new_population[i] = child_1
            new_population[i+1] = child_2
            
        i += 2
    
    return new_population

# Returns True or False depending on if a random number is less than the mutation rate
def should_mutate(mutation_percentage):
    return random.random() < mutation_percentage

# Iterates through each gene and will mutate them if should_mutate() returns true
def mutate_population(population, population_size, mutation_percentage):
    num_tasks = len(tasks)
    num_employees = len(employees)
    
    # Loop through every gene (g) in every chromosome (c)
    for c in range(population_size):
        for g in range(num_tasks):
            if should_mutate(mutation_percentage):
                old_gene = population[c][g]
                new_gene = random.randint(1, num_employees)
                
                # ensures that the randomly chosen gene is not the same as the first
                while old_gene == new_gene:
                    new_gene = random.randint(1, num_employees)
                
                population[c][g] = new_gene
    
    return population

# Graphs the results from the statistics dict
def graph_GA_results(statistics):
    # First plot contains the population statistics
    plt.figure(figsize=(6, 6))
    plt.plot(statistics["min"], label="Minimum")
    plt.plot(statistics["average"], label="Average")
    plt.plot(statistics["deviation"], label="Deviation")

    plt.xlabel("Generation Number")
    plt.ylabel("Cost")
    plt.title("GA Cost statistics each generation")
    plt.legend()
    
    # Second plot contans resoure usage information
    fig2, axes2 = plt.subplots(2, 1, figsize=(6, 6))
    axes2[0].plot(statistics["memory_usage"])
    axes2[0].set_title("Memory Usage")
    axes2[0].set_xlabel("Generation Number")
    axes2[0].set_ylabel("Memory (MB)")
    axes2[1].plot(statistics["time_taken"])
    axes2[1].set_title("Time Taken")
    axes2[1].set_xlabel("Generation Number")
    axes2[1].set_ylabel("Time (Seconds)")
    plt.tight_layout()
    
    # Third plot contans constraint violation information
    plt.figure(figsize=(6, 6))
    plt.plot(statistics["average_violations"], label="Average Violations")
    plt.xlabel("Generation Number")
    plt.ylabel("Violations")
    plt.title("Average Violations per Chromosome")
    plt.legend()
    
    plt.show()

# Defines all parameters and runs the genetic algorithm
def run_GA(crossover_rate, mutation_rate, elitism_rate, tournament_k):
    # inspired from https://www.geeksforgeeks.org/python/how-to-get-current-cpu-and-ram-usage-in-python/
    # used to track the resource usage of the program
    process = psutil.Process(os.getpid())
    
    
    # parameters of the genetic algorithm    
    POPULATION_SIZE = 100
    CROSSOVER_PERCENTAGE = crossover_rate
    MUTATION_PERCENTAGE = mutation_rate
    ELITISM_PERCENTAGE = elitism_rate
    TOURNAMENT_K_SIZE = tournament_k
    MAX_GENERATIONS = 200
    
    generation_when_solution_found = float("inf")
    
    statistics = {
        "min": [],
        "average": [],
        "deviation": [],
        "average_violations": [],
        "memory_usage": [],
        "time_taken": []
    }
    
    solution = None

    starting_memory = process.memory_info().rss / (1024 ** 2)
    # tracks how long it takes to perform the genetic algorithm
    start_time = time.time()
    
    population = generate_population(POPULATION_SIZE)
    
    for i in range(1, MAX_GENERATIONS + 1):
        fitnesses, population_fitness_characteristics, fittest_index = evaluate_population_fitness(population, POPULATION_SIZE)
        
        for key in population_fitness_characteristics:
            statistics[key].append(population_fitness_characteristics[key])
        
        # Checks if a solution with 0 cost has been found and keeps a copy of it
        if population_fitness_characteristics["min"] == 0 and generation_when_solution_found == float("inf"):
            generation_when_solution_found = i
            solution = population[fittest_index].copy()
        
        crossed_population = crossover_population(population, POPULATION_SIZE, fitnesses, CROSSOVER_PERCENTAGE, ELITISM_PERCENTAGE, TOURNAMENT_K_SIZE)
        mutated_population = mutate_population(crossed_population, POPULATION_SIZE, MUTATION_PERCENTAGE)
        
        population = mutated_population
        
        # Converts the memory used into MB and stores it each iteration
        memory_used = process.memory_info().rss / (1024 ** 2)
        statistics["memory_usage"].append(memory_used - starting_memory)
        
        elapsed_time = time.time() - start_time
        statistics["time_taken"].append(elapsed_time)
        
    return solution, generation_when_solution_found, statistics

if __name__ == "__main__":
    num_trials = 1000
    fails = 0
    
    CROSSOVER_PERCENTAGE = 0.4
    MUTATION_PERCENTAGE = 0.075
    ELITISM_PERCENTAGE = 0.07
    TOURNAMENT_K_SIZE = 7
    
    all_statistics = {
        "min": [],
        "average": [],
        "deviation": [],
        "average_violations": [],
        "memory_usage": [],
        "time_taken": []
    }
    
    for trial in range(num_trials):
        solution, generation_when_solution_found, statistics = run_GA(CROSSOVER_PERCENTAGE, MUTATION_PERCENTAGE, ELITISM_PERCENTAGE, TOURNAMENT_K_SIZE)
        print(f"trial {trial+1}: Solution found in generation {generation_when_solution_found}, Solution: {solution}")
        
        for key in all_statistics:
            all_statistics[key].append(statistics[key])
            
        if solution is None:
            fails += 1
    
    averaged_statistics = {key: np.mean(all_statistics[key], axis=0) for key in all_statistics}
    
    print(f"Failed {fails} out of {num_trials} trials")
    graph_GA_results(averaged_statistics)
    

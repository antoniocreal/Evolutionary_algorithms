import numpy as np

# Define parameters
population_size = 100
solution_length = 10
mutation_rate = 0.25
num_generations = 1000
num_parents = int(population_size*0.8)

# Initialize population
population = np.random.rand(population_size, solution_length)
# print('Population:')
# print(population)


def fitness_function(solution):
    expression_value = (np.sin(solution[0]) ** 2) + (np.cos(solution[1]*solution[4]) ** 2) - (np.cos(solution[2]) * (solution[3] ** 2))
    expression_value += np.sin(solution[5]*solution[2]**3) * np.arctan(solution[6])  # Additional trigonometric terms
    expression_value -= (np.sqrt(np.abs(solution[7])) * np.sqrt(np.abs(solution[8]))) / (solution[9] + 1)  # More complex terms
    
    return expression_value

def mutate(individual, mutation_rate):
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if np.random.rand() < mutation_rate:
            mutated_individual[i] += np.random.uniform(low=-0.1, high=0.1)
    return mutated_individual

best_individual_idx = np.argmax(np.apply_along_axis(fitness_function, 1, population))
best_individual = population[best_individual_idx]
best_fitness = fitness_function(best_individual)

best_fitness_every_generation = []
best_fitness_ever = best_fitness

print("Best solution:", best_individual)
print("Best fitness:", best_fitness)


# Genetic Algorithm
for generation in range(num_generations):

    # Defining previous best solutions
    previous_best_individual_idx = best_individual_idx
    previous_best_individual = best_individual
    previous_best_fitness = best_fitness

    # Evaluate fitness
    fitness_values = np.apply_along_axis(fitness_function, 1, population)
    # print('fitness_values:')
    # print(fitness_values)
    
    # Choosing the best parents
    # Building a dicionary with the population chromosomes and its fitness values
    dict_fitness_values_population ={}
    for i,k in zip(population, fitness_values):
        dict_fitness_values_population[tuple(i)] = k
    
    # print('dict_fitness_values_population')
    # print(dict_fitness_values_population)

    # Sorting the dictionary by the fitness values
    sorted_dict = dict(sorted(dict_fitness_values_population.items(), key=lambda item: item[1], reverse=True))
    # print('sorted_dict:')
    # print(sorted_dict)

    # Choosing the top 'num_parents' solutions to constitue the parents for the offspring
    parents = np.array(list(sorted_dict.keys())[:num_parents])
    # print('parents:', len(parents))
    # print(parents)


    # Reshape parents array to ensure it has 3D structure
    parents = parents.reshape(-1, num_parents, solution_length)
    # print('parents:')
    # print(parents)


    # Initialize empty offspring array
    offspring = np.empty((0, solution_length))
    while len(offspring) < population_size:
        # Perform multiple crossovers until desired number of offspring is reached
        # Perform crossover (single-point crossover)
        # Choosing the point of crossover
        crossover_point = np.random.randint(2, solution_length)
        # print('crossover_point', crossover_point)

        # The chromosome is contituted by the first parent genes from the first gene until the crossover point and by the second parent genes from the crossover point until the last gene
        first_parent = np.random.randint(0, num_parents)
        second_parent = np.random.randint(0, num_parents)

        while first_parent == second_parent:
            second_parent = np.random.randint(0,num_parents)

        # print('first_parent', first_parent)
        # print('second_parent', second_parent)

        offspring_1 = np.concatenate((parents[:, first_parent, :crossover_point], parents[:, second_parent, crossover_point:]), axis=1) 
        offspring_1 = offspring_1.reshape(1, -1)
        offspring_2 = np.concatenate((parents[:, second_parent, :crossover_point], parents[:, first_parent, crossover_point:]), axis=1) 
        offspring_2 = offspring_2.reshape(1, -1)
        # print('offspring_1')
        # print(offspring_1)
        # print('offspring_2')
        # print(offspring_2)

        # Add the offspring from the single point crossover to the array
        offspring_1_2 =np.concatenate((offspring_1,offspring_2), axis=0)
        offspring = np.concatenate((offspring, offspring_1_2), axis=0)
        # print('offspring: size ',len(offspring))
        # print(offspring)

    # print('offspring: size ',len(offspring))
    # print(offspring)
    
    # Truncate excess offspring randomly if more than population size
    # Chosen offspring from the original ones
    selected_rows = np.random.choice(offspring.shape[0], size=population_size, replace=False)
    # print('selected_rows', selected_rows)
    offspring = offspring[selected_rows]

    # print('offspring:')
    # print(offspring)
    
    # Perform mutation
    final_offspring = np.empty((0, solution_length))
    for child in offspring:
        # print('original')
        # print(child)
        child = mutate(child, mutation_rate)
        # print('mutated child')
        # print(child)
        child = child.reshape(1, -1) # Needed for the concatenation
        final_offspring =np.concatenate((final_offspring,child), axis=0)

    # print('final_offspring', final_offspring)
    
    # Replace old population with offspring
    population = final_offspring
    # print('population:')
    # print(population)

    print(f'\nFinish iteration: {generation}')

    best_individual_idx = np.argmax(np.apply_along_axis(fitness_function, 1, population))
    best_individual = population[best_individual_idx]
    best_fitness = fitness_function(best_individual)
    best_fitness_every_generation.append(round(best_fitness,2))
    best_fitness_ever = max(best_fitness_every_generation)

    if best_fitness > previous_best_fitness:
        print('Best fitness ever:', best_fitness_ever)
        


# Find best individual in final population
best_individual_idx = np.argmax(np.apply_along_axis(fitness_function, 1, population))
best_individual = population[best_individual_idx]
best_fitness = fitness_function(best_individual)
best_fitness_every_generation.append(round(best_fitness,2))
best_fitness_ever = max(best_fitness_every_generation)

print("Best solution:", best_individual)
print("Best fitness:", best_fitness)
print('Best fitness ever:', best_fitness_ever)
print('Every best fitness of every generation',best_fitness_every_generation)
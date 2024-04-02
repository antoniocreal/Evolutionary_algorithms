import numpy as np

# Define parameters
population_size = 4
chromosome_length = 100
mutation_rate = 0.25
num_generations = 100
num_parents = 2
num_points = 3 # Has to be an odd number for the multi-point mutation
# num_parents = int(min(population_size, population_size *1))  # Ensure even number of parents and do not exceed population size
# print(num_parents)


# Initialize population
population = np.random.randint(2, size=(population_size, chromosome_length))
# print('Population:')
# print(population)

def fitness_function(chromosome):
    return np.sum(chromosome)

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
    # print('parents:')
    # print(parents)


    # Reshape parents array to ensure it has 3D structure
    parents = parents.reshape(-1, 2, chromosome_length)
    print('parents:')
    print(parents)


    # Initialize empty offspring array
    offspring = np.empty((0, chromosome_length))
    
    # Perform multiple crossovers until desired number of offspring is reached
    while len(offspring) < population_size:
        # Perform crossover (single-point crossover)

        # Choosing the point of crossover
        crossover_point = np.random.randint(2, chromosome_length)
        print('crossover_point:')
        print(crossover_point)

        # The chromosome is contituted by the first parent genes from the first gene until the crossover point and by the second parent genes from the crossover point until the last gene
        offspring_batch_1 = np.concatenate((parents[:, 0, :crossover_point], parents[:, 1, crossover_point:]), axis=1)
        print(f'offspring_batch_1: {len(offspring_batch_1)}')
        print(offspring_batch_1)

        # Joining the offspring from the single point crossover
        # offspring = np.concatenate((offspring, offspring_batch_1), axis=0)
        # print('First offspring')
        # print(offspring)

        # Perform crossover (multi-point crossover)
        crossover_points = np.unique(np.random.randint(1, chromosome_length, size=num_points))
        print('crossover_points')
        print(crossover_points)
        # Check if the number of unique crossover points is less than the desired number (For cases where the crossover points are not all different)
        while len(crossover_points) < num_points:   
            # Generate additional crossover points until the desired number is reached
            additional_points = np.unique(np.random.randint(1, chromosome_length, size=num_points - len(crossover_points)))
            crossover_points = np.concatenate((crossover_points, additional_points))
            crossover_points = np.unique(crossover_points)
            print('crossover_points')
            print(crossover_points)

        crossover_points = sorted(crossover_points)
        print('crossover_points:')
        print(crossover_points)

        offspring_batch_2 = parents[:, 0, :crossover_points[0]][0] # The first part of the chromosome is from the first parent from the first gene until the first crossover point
        print('offspring_batch_2')
        print(offspring_batch_2)

        for i in range(1,len(crossover_points)):
            gene_array = (parents[0][i%2][crossover_points[i-1]:crossover_points[i]]) # i%2 gives the modulus. so the first equals to 1 the second equals to 0m the third equals to 1, and so on 
            offspring_batch_2 = np.concatenate((offspring_batch_2, gene_array), axis=None) #It keeps on concatenating each gene_array
            print(f'offspring_batch_2: {len(offspring_batch_2)}')
            print(offspring_batch_2)

        # The last set of genes come from the second parent
        gene_array = (parents[0][1][crossover_points[i]:])
        print('gene_array')
        print(gene_array)
        offspring_batch_2 = np.concatenate((offspring_batch_2, gene_array), axis=None)
        print(f'offspring_batch: {len(offspring_batch_2)}')
        print(offspring_batch_2)

        offspring_batch_2 = offspring_batch_2.reshape(1, -1)
        
        # Add the offspring from the single point and multi point crossover to the array
        offspring = np.concatenate((offspring, offspring_batch_1, offspring_batch_2), axis=0)
        print('offspring:')
        print(offspring)
    
    # Truncate excess offspring if more than population size
    offspring = offspring[:population_size]
    print('offspring:')
    print(offspring)
    
    # Perform mutation
    mutation_mask = np.random.random(offspring.shape) < mutation_rate
    print('mutation_mask:')
    print(mutation_mask)

    offspring[mutation_mask] = np.random.randint(2, size=np.count_nonzero(mutation_mask))
    print('offspring:')
    print(offspring)
    
    # Replace old population with offspring
    population = offspring
    print('population:')
    print(population)

    print(f'\nFinish iteration: {generation}')

    best_individual_idx = np.argmax(np.apply_along_axis(fitness_function, 1, population))
    best_individual = population[best_individual_idx]
    best_fitness = fitness_function(best_individual)
    best_fitness_every_generation.append(best_fitness)
    best_fitness_ever = max(best_fitness_every_generation)

    if best_fitness > previous_best_fitness:
        print("Best solution:", best_individual)
        print("Best fitness:", best_fitness)
        print('Best fitness ever:', best_fitness_ever)
        


# Find best individual in final population
best_individual_idx = np.argmax(np.apply_along_axis(fitness_function, 1, population))
best_individual = population[best_individual_idx]
best_fitness = fitness_function(best_individual)
best_fitness_every_generation.append(best_fitness)
best_fitness_ever = max(best_fitness_every_generation)

print("Best solution:", best_individual)
print("Best fitness:", best_fitness)
print('Best fitness ever:', best_fitness_ever)
print('Every best fitness of every generation',best_fitness_every_generation)
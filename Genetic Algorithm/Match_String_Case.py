import numpy as np

Characters = ["'",  ' ', 'q', 'é', 'ã', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '<', '>', ',', '!', '.', '"', '#', '$', '%', '&', '/', '(', ')', '¹', '@', '£', '§', '½', '¬', '{', '[', ']', '?', '}', ';', '.', ':', '-', '_', '~', '^', '+', '*']
# Unique characters that occur in the text
chars = sorted(list(set(Characters)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { char:int_ for int_,char in enumerate(chars) }
itos = { int_:char for int_,char in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


Objective_phrase = "This genetic algorithm ain't bad!"
print(len(Objective_phrase))
print(encode(Objective_phrase))

# Define parameters
population_size = 100
vector_length = len(Objective_phrase)
mutation_rate = 0.01
num_generations = 10000
num_parents = int(population_size*0.8)

# Initialize population
population = np.random.randint(0, vocab_size, size=(population_size, vector_length))

def fitness_function(phrase, objective_phrase):
    sum_phrase = 0
    for i,k in zip(phrase, encode(objective_phrase)):
        if i == k:
            sum_phrase += 1

    return sum_phrase

def mutate(individual, mutation_rate):
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if np.random.rand() < mutation_rate:
            mutated_individual[i] = np.random.randint(0, vocab_size)
    return mutated_individual

best_individual_idx = np.argmax(np.apply_along_axis(fitness_function, 1, population, objective_phrase=Objective_phrase))
best_individual = population[best_individual_idx]
best_fitness = fitness_function(best_individual, Objective_phrase)

best_fitness_every_generation = []
written_solutions = []
best_fitness_ever = best_fitness


# Genetic Algorithm
for generation in range(num_generations):

    # Defining previous best solutions
    previous_best_individual_idx = best_individual_idx
    previous_best_individual = best_individual
    previous_best_fitness = best_fitness
    previous_best_fitness_ever = best_fitness_ever

    # Evaluate fitness
    fitness_values = np.apply_along_axis(fitness_function, 1, population, objective_phrase=Objective_phrase)
    # print('fitness_values', fitness_values)
    
    # Choosing the best parents
    # Building a dicionary with the population vectors and its fitness values
    dict_fitness_values_population ={}
    for i,k in zip(population, fitness_values):
        dict_fitness_values_population[tuple(i)] = k

    # Sorting the dictionary by the fitness values
    sorted_dict = dict(sorted(dict_fitness_values_population.items(), key=lambda item: item[1], reverse=True))

    # Choosing the top 'num_parents' solutions to constitue the parents for the offspring
    parents = np.array(list(sorted_dict.keys())[:num_parents])

    # Reshape parents array to ensure it has 3D structure
    parents = parents.reshape(-1, len(parents), vector_length)

    # Initialize empty offspring array
    offspring = np.empty((0, vector_length))
    while len(offspring) < population_size:
        # Perform multiple crossovers until desired number of offspring is reached
        # Perform crossover (single-point crossover)
        # Choosing the point of crossover
        crossover_point = np.random.randint(2, vector_length)

        # The vector is contituted by the first parent genes from the first gene until the crossover point and by the second parent genes from the crossover point until the last gene
        first_parent = np.random.randint(0, num_parents)
        second_parent = np.random.randint(0, num_parents)

        while first_parent == second_parent:
            second_parent = np.random.randint(0,num_parents)

        offspring_1 = np.concatenate((parents[:, first_parent, :crossover_point], parents[:, second_parent, crossover_point:]), axis=1) 
        offspring_1 = offspring_1.reshape(1, -1) # Needed for the concatenation
        offspring_2 = np.concatenate((parents[:, second_parent, :crossover_point], parents[:, first_parent, crossover_point:]), axis=1) 
        offspring_2 = offspring_2.reshape(1, -1) # Needed for the concatenation

        # Add the offspring from the single point crossover to the array
        offspring_1_2 =np.concatenate((offspring_1,offspring_2), axis=0)
        offspring = np.concatenate((offspring, offspring_1_2), axis=0)

    
    # Perform mutation
    final_offspring = np.empty((0, vector_length))
    for child in offspring:
        child = mutate(child, mutation_rate)
        child = child.reshape(1, -1) # Needed for the concatenation
        final_offspring =np.concatenate((final_offspring,child), axis=0)
    
    # Replace old population with offspring
    population = final_offspring

    print(f'\nFinish iteration: {generation}')

    best_individual_idx = np.argmax(np.apply_along_axis(fitness_function, 1, population, objective_phrase=Objective_phrase))
    best_individual = population[best_individual_idx]
    best_fitness = fitness_function(best_individual, Objective_phrase)
    best_fitness_every_generation.append(round(best_fitness,2))
    best_fitness_ever = max(best_fitness_every_generation)

    if best_fitness > previous_best_fitness:
        print(decode(best_individual))
        print('Best fitness ever:', best_fitness_ever)

    if best_fitness > previous_best_fitness_ever:
        written_solutions.append(decode(best_individual))
    
    if best_fitness_ever == vector_length:
        break

        

# Find best individual in final population
best_individual_idx = np.argmax(np.apply_along_axis(fitness_function, 1, population, objective_phrase=Objective_phrase))
best_individual = population[best_individual_idx]
best_fitness = fitness_function(best_individual, Objective_phrase)
best_fitness_every_generation.append(round(best_fitness,2))
best_fitness_ever = max(best_fitness_every_generation)

print("Best solution:", decode((best_individual)))
print("Best fitness:", best_fitness)
print('Best fitness ever:', best_fitness_ever)
print('Every best fitness of every generation',best_fitness_every_generation)
print('Best written solutions: ', written_solutions)
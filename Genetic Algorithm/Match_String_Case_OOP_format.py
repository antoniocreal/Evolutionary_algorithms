import numpy as np

class Chromosome:
    def __init__(self, chromosome_length, vocab_size, objective_phrase):
        self.individual = np.random.randint(0, vocab_size, size=chromosome_length)
        self.objective_phrase = objective_phrase
        self.fitness = 0

    def mutation(chromosome, mutation_rate, chromosome_length, vocab_size):
        for i in range(chromosome_length):
            if np.random.rand() < mutation_rate:
                chromosome[i] = np.random.randint(0, vocab_size) 
        
        return chromosome

    def evaluate_fitness(self, fitness_function):
        self.fitness = fitness_function(self.individual, self.objective_phrase)


class GA:
    def __init__(self, population_size, chromosome_length, vocab_size, mutation_rate, num_generations, num_parents, objective_phrase):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.vocab_size = vocab_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.num_parents = num_parents
        self.objective_phrase = objective_phrase
        self.population = [Chromosome(chromosome_length, vocab_size, objective_phrase) for _ in range(population_size)]
        self.best_individual = None # Best chromosome
        self.best_fitness = 0 #Fitness of the best chromosome of the current population
        self.best_fitness_every_generation = []
        self.global_best_fitness = self.best_fitness
        self.written_solutions = []

    def parents_determination(self, fitness_values):
        dict_fitness_values_population ={}
        for chromosome,vector_value in zip(self.population, fitness_values):
            dict_fitness_values_population[tuple(chromosome.individual)] = vector_value
        
        dict_fitness_values_population = dict(sorted(dict_fitness_values_population.items(), key=lambda item: item[1], reverse=True))

        parents = np.array(list(dict_fitness_values_population.keys())[:self.num_parents])

        # Reshape parents array to ensure it has 3D structure
        parents = parents.reshape(-1, len(parents), self.chromosome_length)
        
        return parents
    
    def offspring_determination(self, parents):
            # Initialize empty offspring array
        offspring = np.empty((0, self.chromosome_length))
        while len(offspring) < self.population_size:
            # Perform multiple crossovers until desired number of offspring is reached
            # Perform crossover (single-point crossover)
            # Choosing the point of crossover
            crossover_point = np.random.randint(2, self.chromosome_length)

            # The vector is contituted by the first parent genes from the first gene until the crossover point and by the second parent genes from the crossover point until the last gene
            first_parent = np.random.randint(0, self.num_parents)
            second_parent = np.random.randint(0, self.num_parents)

            while first_parent == second_parent:
                second_parent = np.random.randint(0,self.num_parents)

            offspring_1 = np.concatenate((parents[:, first_parent, :crossover_point], parents[:, second_parent, crossover_point:]), axis=1) 
            offspring_1 = offspring_1.reshape(1, -1) # Needed for the concatenation
            offspring_2 = np.concatenate((parents[:, second_parent, :crossover_point], parents[:, first_parent, crossover_point:]), axis=1) 
            offspring_2 = offspring_2.reshape(1, -1) # Needed for the concatenation

            # Add the offspring from the single point crossover to the array
            offspring_1_2 =np.concatenate((offspring_1,offspring_2), axis=0)
            offspring = np.concatenate((offspring, offspring_1_2), axis=0)

        return offspring

    def offspring_mutation(self, offspring):
        final_offspring = np.empty((0, self.chromosome_length))
        for child in offspring:
            child = Chromosome.mutation(child, self.mutation_rate, self.chromosome_length, self.vocab_size)
            child = child.reshape(1, -1) # Needed for the concatenation
            final_offspring =np.concatenate((final_offspring,child), axis=0)

        return final_offspring

    def optimize(self, fitness_function, objective_phrase):

        for generation in range(self.num_generations):
            print('Generation: ', generation)

            # Evaluate fitness of the current population
            fitness_values = []
            for chromosome in self.population:
                chromosome.evaluate_fitness(fitness_function)
                fitness_values.append(chromosome.fitness)

            # Determine the parents for the next population
            parents = GA.parents_determination(self, fitness_values)

            # The current population's parents create the offspring
            offspring = GA.offspring_determination(self, parents)

            # The offspring suffer from mutation
            mutated_offspring = GA.offspring_mutation(self,offspring)

            for i in range(len(mutated_offspring)):
                self.population[i].individual = mutated_offspring[i]

            for chromosome in self.population:
                chromosome.evaluate_fitness(fitness_function)
                
                if chromosome.fitness > self.best_fitness:
                    # Updates
                    self.best_fitness = chromosome.fitness # Fitness of the best chromosome of the current population
                    self.best_individual = chromosome.individual # Best chromosome of the current population

            # Updates 
            self.best_fitness_every_generation.append(round(self.best_fitness,2))

            if self.best_fitness > self.global_best_fitness:
                self.global_best_fitness = self.best_fitness
                print(decode(self.best_individual))
                print('Best fitness ever:', self.global_best_fitness)
                self.written_solutions.append(decode(self.best_individual))
            
            if self.global_best_fitness == self.chromosome_length:
                break
        
        return self.best_individual, self.best_fitness_every_generation, self.written_solutions, self.global_best_fitness



Characters = ["'",  ' ', 'q', 'é', 'ã', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm', 'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '<', '>', ',', '!', '.', '"', '#', '$', '%', '&', '/', '(', ')', '¹', '@', '£', '§', '½', '¬', '{', '[', ']', '?', '}', ';', '.', ':', '-', '_', '~', '^', '+', '*']
# Unique characters that occur in the text
chars = sorted(list(set(Characters)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { char:int_ for int_,char in enumerate(chars) }
itos = { int_:char for int_,char in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

def fitness_function(chromosome, objective_phrase):
    sum_phrase = 0
    for i,k in zip(chromosome, encode(objective_phrase)):
        if i == k:
            sum_phrase += 1

    return sum_phrase

Objective_phrase = "This genetic algorithm ain't bad!"
population_size = 100
chromosome_length = len(Objective_phrase)
mutation_rate = 0.01
num_generations = 10000
num_parents = int(population_size*0.8)

print('len(Objective_phrase)', len(Objective_phrase))
print('encode(Objective_phrase)', encode(Objective_phrase))

ga = GA(population_size, chromosome_length, vocab_size, mutation_rate, num_generations, num_parents, Objective_phrase)
best_individual, best_fitness_every_generation, written_solutions, global_best_fitness = ga.optimize(fitness_function, Objective_phrase)

print("Best individual:", best_individual)
print("Best fitness every generation:", best_fitness_every_generation)
print("Written solutions:", written_solutions)
print("Best Global fitness:", global_best_fitness)

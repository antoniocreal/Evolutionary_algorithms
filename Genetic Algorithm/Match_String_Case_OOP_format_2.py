import numpy as np

class Population:
    def __init__(self,chromosome_length, population_size, num_parents, objective_phrase, mutation_rate, vocab_size):
        self.chromosome_length = chromosome_length
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.vocab_size = vocab_size
        self.chromosomes = np.random.randint(0, self.vocab_size, size=(population_size,chromosome_length))
        self.num_parents = num_parents
        self.objective_phrase = objective_phrase
        self.best_individual = None # Population's best chromosome
        self.best_fitness = 0   # Best population's chromosome fitness

    def evaluate_fitness(self, fitness_function):
        # Evaluatig all the chromosomes' fitness in the population
        fitness_values = np.apply_along_axis(fitness_function, 1, self.chromosomes, objective_phrase=self.objective_phrase)
        max_fitness_index = np.argmax(fitness_values)

        if fitness_values[max_fitness_index] > self.best_fitness:
            self.best_fitness = fitness_values[max_fitness_index]
            self.best_individual = self.chromosomes[max_fitness_index]
        

    def parents_determination(self, fitness_function):
        fitness_values = np.apply_along_axis(fitness_function, 1, self.chromosomes, objective_phrase=Objective_phrase)

        dict_fitness_values_population ={}
        for chromosome,fitness_value in zip(self.chromosomes, fitness_values):
            dict_fitness_values_population[tuple(chromosome)] = fitness_value
        
        dict_fitness_values_population = dict(sorted(dict_fitness_values_population.items(), key=lambda item: item[1], reverse=True))

        parents = np.array(list(dict_fitness_values_population.keys())[:self.num_parents])

        # Reshape parents array to ensure it has 3D structure
        parents = parents.reshape(-1, len(parents), self.chromosome_length)
        
        return parents

    def mutation(self, chromosome):
        mutated_chromosome = chromosome.copy()
        mask = np.random.rand(*chromosome.shape) < self.mutation_rate # Gives an array with the shape of the chromosome of true's and false's
        mutated_values = np.random.randint(0, self.vocab_size, size=chromosome.shape) # Generates an array with the shape of the chromosome with random values 
        mutated_chromosome[mask] = mutated_values[mask] # Where the mask is true, the chromosome values are substituted with the generated array
        
        return mutated_chromosome

    def crossover(self, parents):
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

            mutated_offspring = self.mutation(offspring)

        return mutated_offspring
        

class GA:
    def __init__(self, population_size, chromosome_length, vocab_size, mutation_rate, num_generations, num_parents, objective_phrase):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.vocab_size = vocab_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.num_parents = num_parents
        self.objective_phrase = objective_phrase
        self.population = Population(self.chromosome_length,self.population_size,self.num_parents,self.objective_phrase, self.mutation_rate, self.vocab_size)
        self.best_fitness_every_generation = []
        self.global_best_individual = self.population.best_individual
        self.global_best_fitness = self.population.best_fitness
        self.written_solutions = []

    def optimize(self, fitness_function):

        for generation in range(self.num_generations):
            print('Generation: ', generation)

            # Determine the parents for the next population
            parents = Population.parents_determination(self.population, fitness_function)

            # Determine current population's offspring 
            offspring = Population.crossover(self.population, parents)

            # Updating the population
            self.population.chromosomes = offspring

            # Updates 
            Population.evaluate_fitness(self.population, fitness_function)
            self.best_fitness_every_generation.append(round(self.population.best_fitness,2))

            if self.population.best_fitness > self.global_best_fitness:
                self.global_best_fitness = self.population.best_fitness
                self.global_best_individual = self.population.best_individual
                print(decode(self.population.best_individual))
                print('Best fitness ever:', self.global_best_fitness)
                self.written_solutions.append(decode(self.population.best_individual))
            
            if self.global_best_fitness == self.chromosome_length:
                break
        
        return self.global_best_individual, self.best_fitness_every_generation, self.written_solutions, self.global_best_fitness



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
best_individual, best_fitness_every_generation, written_solutions, global_best_fitness = ga.optimize(fitness_function)

print("Best individual:", best_individual)
print("Best fitness every generation:", best_fitness_every_generation)
print("Written solutions:", written_solutions)
print("Best Global fitness:", global_best_fitness)

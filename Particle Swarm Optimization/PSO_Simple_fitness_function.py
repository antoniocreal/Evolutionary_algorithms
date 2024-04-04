import numpy as np

class Particle:
  # Each particle has a position and a velocity. For its iterations each particle has a best position, which is calculated with the fitness_function through the evaluate_fitness function.
    def __init__(self, num_dimensions):
        self.position = np.random.uniform(-5, 5, num_dimensions)  # Initialize random position
        self.velocity = np.random.uniform(-1, 1, num_dimensions)  # Initialize random velocity
        self.best_position = self.position.copy()  # Initialize best position as current position
        self.best_fitness = float('inf')  # Initialize best fitness as infinity

    def update_position(self):
        self.position += self.velocity  # Update position based on velocity

    def update_velocity(self, global_best_position, inertia_weight, cognitive_coeff, social_coeff):
        self.velocity = inertia_weight * self.velocity \
                        + cognitive_coeff * np.random.rand() * (self.best_position - self.position) \
                        + social_coeff * np.random.rand() * (global_best_position - self.position)

    def evaluate_fitness(self, fitness_function):
        self.fitness = fitness_function(self.position)  # Evaluate fitness
        if self.fitness < self.best_fitness:  # Update best position if current fitness is better
            self.best_position = self.position.copy()
            self.best_fitness = self.fitness

class PSO:
    def __init__(self, num_particles, num_dimensions, num_iterations, inertia_weight, cognitive_coeff, social_coeff):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.num_iterations = num_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.particles = [Particle(num_dimensions) for _ in range(num_particles)]

    def optimize(self, fitness_function):
        for particle in self.particles:
          print(particle.position, particle.velocity)
        for iteration in range(self.num_iterations):
            print('iteration', iteration)
            for particle in self.particles:
                particle.evaluate_fitness(fitness_function)
                if particle.best_fitness < self.global_best_fitness:  # Update global best if current particle's best fitness is better
                    print('Best fitness: ', particle.best_fitness)
                    print('Best position: ', particle.best_position)
                    self.global_best_position = particle.best_position.copy()
                    self.global_best_fitness = particle.best_fitness
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.inertia_weight, self.cognitive_coeff, self.social_coeff)
                particle.update_position()
        return self.global_best_position, self.global_best_fitness

# Example usage:
def fitness_function(x):
    return np.sum(x ** 2) + np.cos(x[1]) - np.cos(x[2]*1000)*np.sin(x[3]*3) - np.cos(x[4]**4)*np.sin(x[5]/1000) # Example fitness function (sphere function)

num_particles = 20
num_dimensions = 6 # Dimensions of each particle
num_iterations = 1000
inertia_weight = 0.5
cognitive_coeff = 1.5
social_coeff = 1.5

pso = PSO(num_particles, num_dimensions, num_iterations, inertia_weight, cognitive_coeff, social_coeff)
best_position, best_fitness = pso.optimize(fitness_function)

print("Best solution:", best_position)
print("Best fitness:", best_fitness)
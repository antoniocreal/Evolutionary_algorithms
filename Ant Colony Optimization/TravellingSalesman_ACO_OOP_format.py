import numpy as np

class Ant:
    def __init__(self, n_cities, pheromone_matrix, alpha, distance_matrix, beta):
        self.n_cities = n_cities
        self.pheromone_matrix = pheromone_matrix
        self.alpha = alpha
        self.distance_matrix = distance_matrix
        self.beta = beta
    
    def tour_construction(self):
        tour = []
        unvisited_cities = list(range(self.n_cities)) 
        current_city = np.random.randint(0,self.n_cities) # Start from a random city
        tour.append(current_city) 
        unvisited_cities.remove(current_city)

        # Build tour
        while len(tour) < self.n_cities:

            # Calculate probabilities for the next city
            probabilities = (self.pheromone_matrix[current_city][unvisited_cities] ** self.alpha) * ((1 / (self.distance_matrix[current_city][unvisited_cities]) + 1) ** self.beta)
            # pheromone_matrix[current_city] represents the pheromone levels on the edges departing from the current city
            # alpha is a parameter that controls the relative importance of pheromone information compared to heuristic information.
            # distance_matrix[current_city] represents the distances from the current city to the candidate cities
            # beta is a parameter that controls the relative importance of heuristic information compared to pheromone informatio

            probabilities /= np.sum(probabilities) # The sum of the probabilities matrix must be 1 
            
            # Choose the next city based on the probabilities
            next_city = np.random.choice(unvisited_cities, p=probabilities)

            # Move to the next city
            tour.append(next_city)
            current_city = next_city
            unvisited_cities.remove(next_city)
        
        return tour
    

class ACO:
    def __init__(self,n_cities, alpha, beta, num_ants, num_iterations, evaporation_rate, max_distance, min_distance):
        self.n_cities = n_cities
        self.alpha = alpha
        self.beta = beta
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.evaporation_rate = evaporation_rate
        self.max_distance = max_distance
        self.min_distance = min_distance
        
        self.All_best_iteration_tours = []
        self.All_best_iteration_tour_lengths = []
        self.best_tour_length = float('inf')
        self.best_tour = None

        # Create a symmetric matrix as a distance matrix    
        # Generate random values for the upper triangle of the matrix
        np.random.seed(0) # Having a seed is good to measure how better the performance increases from the changes one makes
        upper_triangle = np.random.randint(self.min_distance, self.max_distance, size=(self.n_cities, self.n_cities))

        # Make the matrix symmetric by copying the upper triangle to the lower triangle
        self.distance_matrix = np.triu(upper_triangle) + np.triu(upper_triangle, 1).T

        # Initialize pheromone trails
        self.pheromone_matrix = np.ones_like(self.distance_matrix) / np.mean(self.distance_matrix)

        self.colony = [Ant(self.n_cities, self.pheromone_matrix, self.alpha, self.distance_matrix, self.beta) for _ in range(self.num_ants)]

    def optimize(self):

        for iteration in range(self.num_iterations):
            ant_tours = []
            for ant in self.colony:
                tour = Ant.tour_construction(self)
                ant_tours.append(tour)
            
            # Update pheromone trails
            self.pheromone_matrix *= (1 - self.evaporation_rate)  # Evaporation
            for ant_tour in ant_tours:
                for i in range(len(ant_tour) - 1):
                    self.pheromone_matrix[ant_tour[i], ant_tour[i + 1]] += 1 / self.distance_matrix[ant_tour[i], ant_tour[i + 1]]
            
            # Apply pheromone evaporation
            self.pheromone_matrix = np.maximum(self.pheromone_matrix, np.ones_like(self.pheromone_matrix) * 1e-10)  # Avoid zero division
            
            # Print the best tour found in this iteration
            for ant_tour in ant_tours:
                tour_length = sum(self.distance_matrix[ant_tour[i], ant_tour[i + 1]] for i in range(len(ant_tour) - 1))
                if tour_length < self.best_tour_length:
                    self.best_tour_length = tour_length
                    self.best_tour = ant_tour
                    self.All_best_iteration_tours.append(self.best_tour)
                    self.All_best_iteration_tour_lengths.append(self.best_tour_length)
                    print("Iteration", iteration, "- Best tour:", self.best_tour, "- Length:", self.best_tour_length)

        return self.All_best_iteration_tours, self.All_best_iteration_tour_lengths, self.best_tour_length, self.best_tour
    

# Define parameters
num_ants = 5
num_iterations = 1000
evaporation_rate = 0.5
alpha = 1.0  # Pheromone importance
beta = 2.0   # Heuristic information importance
max_distance = 1000 # Lowest distance must be >0 otherwise, the distance between two cities will be none
min_distance = 1
n_cities = 50


aco = ACO(n_cities, alpha, beta, num_ants, num_iterations, evaporation_rate, max_distance, min_distance)
All_best_iteration_tours, All_best_iteration_tours_lengths, best_tour_length, best_tour = aco.optimize()

print("All_best_iteration_tours:", All_best_iteration_tours)
print("All_best_iteration_tours_lengths:", All_best_iteration_tours_lengths)
print("best_tour_length:", best_tour_length)
print("best_tour:", best_tour)




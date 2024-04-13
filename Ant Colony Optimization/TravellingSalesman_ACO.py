import numpy as np

# Define parameters
num_ants = 50
num_iterations = 1000
evaporation_rate = 0.5
alpha = 1.0  # Pheromone importance
beta = 2.0   # Heuristic information importance
max_distance = 1000 # Lowest distance must be >0 otherwise, the distance between two cities will be none
min_distance = 1
n_cities = 50

# Create a symmetric matrix as a distance matrix    
# Generate random values for the upper triangle of the matrix
np.random.seed(0) # Having a seed is good to measure how better the performance increases from the changes one makes
upper_triangle = np.random.randint(min_distance, max_distance, size=(n_cities, n_cities))

# Make the matrix symmetric by copying the upper triangle to the lower triangle
distance_matrix = np.triu(upper_triangle) + np.triu(upper_triangle, 1).T

# Initialize pheromone trails
pheromone_matrix = np.ones_like(distance_matrix) / np.mean(distance_matrix)

# Ant Colony Optimization algorithm
All_best_iteration_tours = []
All_best_iteration_tour_lengths = []
best_tour_length = float('inf')
best_tour = None

for iteration in range(num_iterations):
    ant_tours = []
    for ant in range(num_ants):
        # Initialize ant's tour
        tour = []
        unvisited_cities = list(range(n_cities)) 
        current_city = np.random.randint(0,n_cities) # Start from a random city
        tour.append(current_city) 
        unvisited_cities.remove(current_city)
        
        # Build tour
        while len(tour) < n_cities:

            # Calculate probabilities for the next city
            probabilities = (pheromone_matrix[current_city][unvisited_cities] ** alpha) * ((1 / (distance_matrix[current_city][unvisited_cities]) + 1) ** beta)
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

        # Add the tour to the list of ant tours
        ant_tours.append(tour)
    
    # Update pheromone trails
    pheromone_matrix *= (1 - evaporation_rate)  # Evaporation
    for ant_tour in ant_tours:
        for i in range(len(ant_tour) - 1):
            pheromone_matrix[ant_tour[i], ant_tour[i + 1]] += 1 / distance_matrix[ant_tour[i], ant_tour[i + 1]]
    
    # Apply pheromone evaporation
    pheromone_matrix = np.maximum(pheromone_matrix, np.ones_like(pheromone_matrix) * 1e-10)  # Avoid zero division
    
    # Print the best tour found in this iteration
    for ant_tour in ant_tours:
        tour_length = sum(distance_matrix[ant_tour[i], ant_tour[i + 1]] for i in range(len(ant_tour) - 1))
        if tour_length < best_tour_length:
            best_tour_length = tour_length
            best_tour = ant_tour
            All_best_iteration_tours.append(best_tour)
            All_best_iteration_tour_lengths.append(best_tour_length)
            print("Iteration", iteration, "- Best tour:", best_tour, "- Length:", best_tour_length)

# print('All_best_iteration_tours: ', All_best_iteration_tours)
print('All_best_iteration_tour_lengths: ', All_best_iteration_tour_lengths)
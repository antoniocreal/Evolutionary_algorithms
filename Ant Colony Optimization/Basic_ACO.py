import numpy as np
# Define the distance matrix (example TSP instance)


# Define parameters
num_ants = 2
num_iterations = 200
evaporation_rate = 0.5
alpha = 1.0  # Pheromone importance
beta = 2.0   # Heuristic information importance
max_distance= 1000
n_cities = 50

# Create a symmetric matrix as a distance matrix    
# Generate random values for the upper triangle of the matrix
upper_triangle = np.random.randint(0, max_distance, size=(n_cities, n_cities))

# Make the matrix symmetric by copying the upper triangle to the lower triangle
distance_matrix = np.triu(upper_triangle) + np.triu(upper_triangle, 1).T
# print("Distance Matrix (symmetric):")
# print(distance_matrix)

# Initialize pheromone trails
pheromone_matrix = np.ones_like(distance_matrix) / np.mean(distance_matrix)
print('pheromone_matrix', pheromone_matrix)
print('pheromone_matrix', np.sum(pheromone_matrix), 'np.mean(distance_matrix)', np.mean(distance_matrix) )
# print(np.ones_like(distance_matrix))
a
# Ant Colony Optimization algorithm
for iteration in range(num_iterations):
    ant_tours = []
    for ant in range(num_ants):
        # Initialize ant's tour
        tour = []
        tour.append(np.random.randint(0,n_cities))  # Start from a random city
        print('tour',tour)
        current_city = 0
        
        # Build tour
        while len(tour) < len(distance_matrix):
            print('len(tour)', len(tour))
            print('len(distance_matrix)',len(distance_matrix))
            # Calculate probabilities for the next city
            probabilities = (pheromone_matrix[current_city] ** alpha) * ((1 / (distance_matrix[current_city] + 1e-10)) ** beta)
            # pheromone_matrix[current_city] represents the pheromone levels on the edges departing from the current city
            # alpha is a parameter that controls the relative importance of pheromone information compared to heuristic information.
            # distance_matrix[current_city] represents the distances from the current city to the candidate cities
            # beta is a parameter that controls the relative importance of heuristic information compared to pheromone information
            print('probabilities', probabilities)
            print('sum(probabilities)', sum(probabilities))

            a

            probabilities /= np.sum(probabilities)
            print('probabilities', probabilities)
            
            # Choose the next city based on the probabilities
            next_city = np.random.choice(range(len(distance_matrix)), p=probabilities)
            print('next_city', next_city)
            
            # Move to the next city
            tour.append(next_city)
            current_city = next_city
        
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
    best_tour_length = float('inf')
    best_tour = None
    for ant_tour in ant_tours:
        tour_length = sum(distance_matrix[ant_tour[i], ant_tour[i + 1]] for i in range(len(ant_tour) - 1))
        if tour_length < best_tour_length:
            best_tour_length = tour_length
            best_tour = ant_tour
    print("Iteration", iteration + 1, "- Best tour:", best_tour, "- Length:", best_tour_length)
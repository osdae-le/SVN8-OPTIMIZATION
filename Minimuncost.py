import random
import numpy as np
import time
num_routes = 50
num_locations = 4
num_transports = 4
locations = ['HaNoi', 'HaiPhong', 'VungTau', 'GiaLai']
transport_modes = ['airport', 'train_station', 'seaport', 'warehouse']
transport_costs = np.random.rand(num_locations, num_locations, num_transports) * 100
transport_times = np.random.rand(num_locations, num_locations, num_transports) * 10 

population_size = 100
mutation_rate = 0.1
num_generations = 100
num_iterations = 1000
tabu_size = 50
initial_temperature = 1000.0
cooling_rate = 0.95

input_configuration = [
    ('HaNoi', 'HaiPhong', 0),
    ('HaiPhong', 'VungTau', 1),
    ('GiaLai', 'HaiPhong', 2),
    ('VungTau', 'HaNoi', 3),
]

print(transport_costs)
print(transport_times)

def fitness(individual):
    total_cost = 0
    for route in individual:
        origin, destination, transport = route
        total_cost += transport_costs[locations.index(origin)][locations.index(destination)][transport]
    return -total_cost

def generate_individual():
    return input_configuration

def select_parents(population, num_parents):
    parents = []
    for _ in range(num_parents):
        tournament_size = 5
        tournament_contestants = random.sample(population, tournament_size)
        winner = max(tournament_contestants, key=fitness)
        parents.append(winner)
    return parents

def crossover(parent1, parent2):
    crossover_point = random.randint(1, num_routes - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = (random.choice(locations), random.choice(locations), random.randint(0, num_transports - 1))

def genetic_algorithm(population, fitness_function, crossover_function, mutate_function, num_generations, mutation_rate):
    best_solution = None
    best_fitness = float('-inf')
    for generation in range(num_generations):
        population = sorted(population, key=fitness_function)
        if fitness_function(population[0]) > best_fitness:
            best_solution = population[0]
            best_fitness = fitness_function(population[0])
        new_population = population[:2]
        for _ in range(population_size // 2 - 1):
            parent1, parent2 = select_parents(population, 2)
            offspring = crossover_function(parent1, parent2)
            mutate_function(offspring, mutation_rate)
            new_population.append(offspring)
        population = new_population
    return best_solution

def calculate_cost(solution):
    total_cost = 0
    for route in solution:
        origin, destination, transport = route
        total_cost += transport_costs[locations.index(origin)][locations.index(destination)][transport]
    return total_cost

def calculate_time_cost(solution):
    total_time = 0
    for route in solution:
        origin, destination, transport = route
        total_time += transport_times[locations.index(origin)][locations.index(destination)][transport]
    return total_time

def time_to_money_cost(time_cost, rate):
    return time_cost * rate

def get_neighborhood(solution):
    neighborhood = []
    for i in range(len(solution)):
        for j in range(len(locations)):
            for k in range(num_transports):
                neighbor = list(solution)  # Convert tuple to list
                neighbor[i] = (neighbor[i][0], locations[j], k)
                neighborhood.append(tuple(neighbor))  # Convert back to tuple
    return neighborhood

def tabu_search(initial_solution, num_iterations, tabu_size):
    current_solution = initial_solution
    best_solution = current_solution
    tabu_list = []
    for _ in range(num_iterations):
        neighborhood = get_neighborhood(current_solution)
        neighborhood = [tuple(route) for route in neighborhood]
        neighborhood = list(set(neighborhood) - set(tabu_list))
        if not neighborhood:
            break
        next_solution = min(neighborhood, key=calculate_cost)
        current_cost = calculate_cost(current_solution)
        next_cost = calculate_cost(next_solution)
        if next_cost < current_cost:
            current_solution = next_solution
            if next_cost < calculate_cost(best_solution):
                best_solution = next_solution
        tabu_list.append(next_solution)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)
    return best_solution

def get_neighbor(solution):
    neighbor = solution[:]
    index = random.randint(0, len(neighbor) - 1)
    neighbor[index] = (neighbor[index][0], random.choice(locations), random.randint(0, num_transports - 1))
    return neighbor

def acceptance_probability(current_cost, new_cost, temperature):
    if new_cost < current_cost:
        return 1.0
    return np.exp((current_cost - new_cost) / temperature)

def simulated_annealing(initial_solution, num_iterations, initial_temperature, cooling_rate):
    current_solution = initial_solution
    best_solution = current_solution
    current_cost = calculate_cost(current_solution)
    best_cost = current_cost
    temperature = initial_temperature
    for _ in range(num_iterations):
        neighbor = get_neighbor(current_solution)
        new_cost = calculate_cost(neighbor)
        if acceptance_probability(current_cost, new_cost, temperature) > random.random():
            current_solution = neighbor
            current_cost = new_cost
            if new_cost < best_cost:
                best_solution = current_solution
                best_cost = new_cost
        temperature *= cooling_rate
    return best_solution

def greedy_algorithm(locations, transport_costs):
    num_locations = len(locations)
    num_transports = len(transport_costs[0][0])

    visited = [False] * num_locations
    solution = []

    current_location = 0
    visited[current_location] = True

    while len(solution) < num_locations - 1:
        next_location = None
        min_cost = float('inf')

        for i in range(num_locations):
            if not visited[i]:
                for transport in range(num_transports):
                    cost = transport_costs[current_location][i][transport]
                    if cost < min_cost:
                        min_cost = cost
                        next_location = i

        solution.append((locations[current_location], locations[next_location], np.argmin(transport_costs[current_location][next_location])))
        visited[next_location] = True
        current_location = next_location

    solution.append((locations[current_location], locations[0], np.argmin(transport_costs[current_location][0])))

    return solution

class Ant:
    def __init__(self, num_locations):
        self.num_locations = num_locations
        self.visited = [False] * num_locations
        self.tour = []

    def select_next_location(self, pheromone_matrix, distance_matrix, alpha, beta):
        current_location = self.tour[-1] if self.tour else 0

        unvisited_locations = [i for i in range(self.num_locations) if not self.visited[i]]
        probabilities = [((pheromone_matrix[current_location][next_location] ** alpha) *
                          (1.0 / distance_matrix[current_location][next_location] ** beta))
                         for next_location in unvisited_locations]

        total_probability = sum(probabilities)
        probabilities = [prob / total_probability for prob in probabilities]

        next_location_index = np.random.choice(len(unvisited_locations), p=probabilities)
        next_location = unvisited_locations[next_location_index]

        self.visited[next_location] = True
        self.tour.append(next_location)

        return next_location

def ant_colony_optimization(num_ants, num_iterations, alpha, beta, evaporation_rate):
    pheromone_matrix = np.ones((num_locations, num_locations))  # Initialize pheromone matrix
    best_solution = None
    best_cost = float('inf')

    for _ in range(num_iterations):
        ants = [Ant(num_locations) for _ in range(num_ants)]

        for ant in ants:
            while len(ant.tour) < num_locations:
                ant.select_next_location(pheromone_matrix, transport_costs[:,:,0], alpha, beta)  # Assuming transport mode 0 for simplicity

        for i in range(num_locations):
            for j in range(num_locations):
                if i != j:
                    pheromone_matrix[i][j] *= (1 - evaporation_rate)  # Evaporation
                    for ant in ants:
                        if j in ant.tour and i in ant.tour:
                            pheromone_matrix[i][j] += (1.0 / calculate_cost([(locations[i], locations[j], 0)]))

        for ant in ants:
            tour_cost = calculate_cost([(locations[ant.tour[i-1]], locations[ant.tour[i]], 0) for i in range(1, len(ant.tour))])
            if tour_cost < best_cost:
                best_solution = [(locations[ant.tour[i-1]], locations[ant.tour[i]], 0) for i in range(1, len(ant.tour))]
                best_cost = tour_cost

    return best_solution

class Particle:
    def __init__(self, num_locations):
        self.position = [random.randint(0, num_locations - 1) for _ in range(num_locations)]
        self.velocity = [random.uniform(-1, 1) for _ in range(num_locations)]
        self.best_position = self.position.copy()

    def update_velocity(self, global_best_position, inertia_weight, cognitive_weight, social_weight):
        for i in range(len(self.velocity)):
            cognitive_component = cognitive_weight * random.random() * (self.best_position[i] - self.position[i])
            social_component = social_weight * random.random() * (global_best_position[i] - self.position[i])
            self.velocity[i] = inertia_weight * self.velocity[i] + cognitive_component + social_component

    def update_position(self, num_locations):
        for i in range(len(self.position)):
            self.position[i] = max(0, min(num_locations - 1, int(round(self.position[i] + self.velocity[i]))))

def particle_swarm_optimization(num_particles, num_iterations, inertia_weight, cognitive_weight, social_weight):
    particles = [Particle(num_locations) for _ in range(num_particles)]
    global_best_position = min(particles, key=lambda p: calculate_cost([(locations[p.position[i-1]], locations[p.position[i]], 0) for i in range(1, len(p.position))])).position
    global_best_cost = calculate_cost([(locations[global_best_position[i-1]], locations[global_best_position[i]], 0) for i in range(1, len(global_best_position))])

    for _ in range(num_iterations):
        for particle in particles:
            particle.update_velocity(global_best_position, inertia_weight, cognitive_weight, social_weight)
            particle.update_position(num_locations)

            particle_cost = calculate_cost([(locations[particle.position[i-1]], locations[particle.position[i]], 0) for i in range(1, len(particle.position))])
            best_particle_cost = calculate_cost([(locations[particle.best_position[i-1]], locations[particle.best_position[i]], 0) for i in range(1, len(particle.best_position))])

            if particle_cost < best_particle_cost:
                particle.best_position = particle.position.copy()

            if particle_cost < global_best_cost:
                global_best_position = particle.position.copy()
                global_best_cost = particle_cost

    return [(locations[global_best_position[i-1]], locations[global_best_position[i]], 0) for i in range(1, len(global_best_position))]

time_cost_rate = 50.0 

start_time_pso = time.time()
solution_pso = particle_swarm_optimization(num_particles=20, num_iterations=100, inertia_weight=0.9, cognitive_weight=0.5, social_weight=0.5)
cost_pso = calculate_cost(solution_pso)
time_cost_pso = calculate_time_cost(solution_pso)
money_cost_pso = time_to_money_cost(time_cost_pso, time_cost_rate)
end_time_pso = time.time()

print("Particle Swarm Optimization - Optimal Solution:", solution_pso)
print("Particle Swarm Optimization - Total Cost:", cost_pso)
print("Particle Swarm Optimization - Time Cost:", time_cost_pso)
print("Particle Swarm Optimization - Money Cost:", money_cost_pso)
print("Time taken by Particle Swarm Optimization:", end_time_pso - start_time_pso)

start_time_aco = time.time()
solution_aco = ant_colony_optimization(num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.1)
cost_aco = calculate_cost(solution_aco)
time_cost_aco = calculate_time_cost(solution_aco)
money_cost_aco = time_to_money_cost(time_cost_aco, time_cost_rate)
end_time_aco = time.time()

print("Ant Colony Optimization - Optimal Solution:", solution_aco)
print("Ant Colony Optimization - Total Cost:", cost_aco)
print("Ant Colony Optimization - Time Cost:", time_cost_aco)
print("Ant Colony Optimization - Money Cost:", money_cost_aco)
print("Time taken by Ant Colony Optimization:", end_time_aco - start_time_aco)

start_time_greedy = time.time()
solution_greedy = greedy_algorithm(locations, transport_costs)
cost_greedy = calculate_cost(solution_greedy)
time_cost_greedy = calculate_time_cost(solution_greedy)
money_cost_greedy = time_to_money_cost(time_cost_greedy, time_cost_rate)
end_time_greedy = time.time()

print("Greedy Algorithm - Optimal Solution:", solution_greedy)
print("Greedy Algorithm - Total Cost:", cost_greedy)
print("Greedy Algorithm - Time Cost:", time_cost_greedy)
print("Greedy Algorithm - Money Cost:", money_cost_greedy)
print("Time taken by Greedy Algorithm:", end_time_greedy - start_time_greedy)

start_time_genetic = time.time()
population_genetic = [generate_individual() for _ in range(population_size)]
solution_genetic = genetic_algorithm(population_genetic, fitness, crossover, mutate, num_generations, mutation_rate)
cost_genetic = -fitness(solution_genetic)
time_cost_genetic = calculate_time_cost(solution_genetic)
money_cost_genetic = time_to_money_cost(time_cost_genetic, time_cost_rate)
end_time_genetic = time.time()

print("Genetic Algorithm - Optimal Solution:", solution_genetic)
print("Genetic Algorithm - Total Cost:", cost_genetic)
print("Genetic Algorithm - Time Cost:", time_cost_genetic)
print("Genetic Algorithm - Money Cost:", money_cost_genetic)
print("Time taken by Genetic Algorithm:", end_time_genetic - start_time_genetic)

start_time_tabu = time.time()
initial_solution_tabu = input_configuration
solution_tabu = tabu_search(initial_solution_tabu, num_iterations, tabu_size)
cost_tabu = calculate_cost(solution_tabu)
time_cost_tabu = calculate_time_cost(solution_tabu)
money_cost_tabu = time_to_money_cost(time_cost_tabu, time_cost_rate)
end_time_tabu = time.time()

print("Tabu Search - Optimal Solution:", solution_tabu)
print("Tabu Search - Total Cost:", cost_tabu)
print("Tabu Search - Time Cost:", time_cost_tabu)
print("Tabu Search - Money Cost:", money_cost_tabu)
print("Time taken by Tabu Search:", end_time_tabu - start_time_tabu)

start_time_annealing = time.time()
initial_solution_annealing = input_configuration
solution_annealing = simulated_annealing(initial_solution_annealing, num_iterations, initial_temperature, cooling_rate)
cost_annealing = calculate_cost(solution_annealing)
time_cost_annealing = calculate_time_cost(solution_annealing)
money_cost_annealing = time_to_money_cost(time_cost_annealing, time_cost_rate)
end_time_annealing = time.time()

print("Simulated Annealing - Optimal Solution:", solution_annealing)
print("Simulated Annealing - Total Cost:", cost_annealing)
print("Simulated Annealing - Time Cost:", time_cost_annealing)
print("Simulated Annealing - Money Cost:", money_cost_annealing)
print("Time taken by Simulated Annealing:", end_time_annealing - start_time_annealing)

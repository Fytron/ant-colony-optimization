import numpy as np
import random
import matplotlib.pyplot as plt

# ACO Hyperparameters
ALPHA = 1  # Pheromone importance
BETA = 2   # Distance importance
EVAPORATION_RATE = 0.5
PHEROMONE_DEPOSIT = 1.0
DEBUG = False

def parse_evrp_data(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    coordinates, demands, station_coords, depots = [], [], [], []
    params = {
        "DIMENSION": (int, "dimension"),
        "CAPACITY": (int, "cargo_capacity"),
        "ENERGY_CAPACITY": (int, "battery_capacity"),
        "ENERGY_CONSUMPTION": (float, "energy_consumption"),
        "STATIONS": (int, "num_stations"),
        "VEHICLES": (int, "num_vehicles")
    }
    data = {param: 0 for param in params.values()}

    reading_section = None
    for line in content:
        line = line.strip()
        if ":" in line:
            key, value = line.split(":")
            if key in params:
                parse_type, param_name = params[key]
                data[param_name] = parse_type(value)
        elif "EOF" in line:
            break
        elif "SECTION" in line:
            reading_section = line.split()[0]
        else:
            if reading_section == "NODE_COORD_SECTION":
                node_id, x, y = map(float, line.split())
                coordinates.append((x, y))
            elif reading_section == "DEMAND_SECTION":
                node_id, demand = map(int, line.split())
                demands.append((node_id - 1, demand))
            elif reading_section == "STATIONS_COORD_SECTION":
                station_id = int(line) - 1
                if 0 <= station_id < data["dimension"]:
                    station_coords.append(station_id)
            elif reading_section == "DEPOT_SECTION" and int(line) != -1:
                depots.append(int(line))

    data.update({
        "coordinates": coordinates,
        "demands": demands,
        "station_coords": station_coords,
        "depots": depots
    })

    return data

def euclidean_distance(point1, point2):
    return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

def create_distance_matrix(coordinates):
    num_nodes = len(coordinates)
    dist_mat = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist_mat[i][j] = euclidean_distance(coordinates[i], coordinates[j])
            else:
                dist_mat[i][j] = float('inf')

    return dist_mat

def evrp_obj(x_ij, d_ij):
    total_distance = 0
    for i in range(len(x_ij)):
        for j in range(len(x_ij)):
            if x_ij[i][j] == 1:
                total_distance += d_ij[i][j]
    return total_distance

class EV_Ant:
    def __init__(self, dimensions, dist_mat, battery_capacity, cargo_capacity, charging_stations, demands, depots):
        self.dimensions = dimensions
        self.dist_mat = dist_mat
        self.battery_capacity = battery_capacity
        self.cargo_capacity = cargo_capacity
        self.charging_stations = charging_stations
        self.demands = demands
        self.depots = depots
        
        self.pheromone_trails = np.ones((dimensions, dimensions)) * 0.1
        self.reset()

    def reset(self):
        self.position = np.random.permutation(self.dimensions)
        self.x_ij = np.zeros((self.dimensions, self.dimensions))
        self.current_battery = self.battery_capacity
        self.current_cargo = self.cargo_capacity
        self.quality = float('inf')

    def probabilistic_node_selection(self, current_node, unvisited_nodes, global_pheromones):
        probabilities = []
        for node in unvisited_nodes:
            # Probabilistic selection considering pheromones and distance
            pheromone = global_pheromones[current_node][node]
            visibility = 1 / (self.dist_mat[current_node][node] + 1e-10)
            probabilities.append((node, pheromone ** ALPHA * visibility ** BETA))
        
        total = sum(prob for _, prob in probabilities)
        probabilities = [(node, prob/total) for node, prob in probabilities]
        return random.choices(probabilities, weights=[p for _, p in probabilities])[0][0]

    def calculate_battery_consumption(self, from_node, to_node):
        return self.dist_mat[from_node][to_node]

    def can_visit_next_node(self, current_node, next_node):
        if next_node < 0 or next_node >= len(self.demands):
            return False

        battery_needed = self.calculate_battery_consumption(current_node, next_node)
        return (self.current_battery >= battery_needed and 
                self.current_cargo >= self.demands[next_node][1])

    def route_construction(self, global_pheromones):
        self.reset()
        
        # Start from the depot
        current_node = self.depots[0]
        route = [current_node]
        visited_nodes = set([current_node])
        
        nodes_to_visit = set(
            node for node in range(self.dimensions) 
            if node not in self.depots and node not in self.charging_stations
        )
        
        while nodes_to_visit:
            # Potential next nodes are unvisited customer nodes
            potential_next_nodes = [
                node for node in nodes_to_visit 
                if self.can_visit_next_node(current_node, node)
            ]
            
            # If no direct route is possible, use a charging station
            if not potential_next_nodes:
                # Find the nearest charging station
                potential_stations = [
                    station for station in self.charging_stations 
                    if station not in visited_nodes
                ]
                
                if not potential_stations:
                    break
                
                # Select charging station probabilistically
                next_station = self.probabilistic_node_selection(
                    current_node, 
                    potential_stations, 
                    global_pheromones
                )
                
                # Recharge and update route
                route.append(next_station)
                visited_nodes.add(next_station)
                self.current_battery = self.battery_capacity
                current_node = next_station
                continue
            
            # Select next node probabilistically
            next_node = self.probabilistic_node_selection(
                current_node, 
                potential_next_nodes, 
                global_pheromones
            )
            
            # Update route details
            battery_consumption = self.calculate_battery_consumption(current_node, next_node)
            self.current_battery -= battery_consumption
            self.current_cargo -= self.demands[next_node][1]
            
            # Update tracking information
            route.append(next_node)
            visited_nodes.add(next_node)
            nodes_to_visit.remove(next_node)
            self.x_ij[current_node][next_node] = 1
            current_node = next_node
        
        # Return to depot
        if current_node != self.depots[0]:
            self.x_ij[current_node][self.depots[0]] = 1
            route.append(self.depots[0])
        
        self.position = route
        self.quality = evrp_obj(self.x_ij, self.dist_mat)
        return route

def ant_colony_optimization_evrp(file_path, num_generations=20, population_size=10, plot=True):
    # Parse problem data
    parsed_data = parse_evrp_data(file_path)
    
    coordinates = parsed_data['coordinates']
    demands = parsed_data['demands']
    stations = parsed_data['station_coords']
    depots = parsed_data['depots']
    dimensions = parsed_data['dimension']
    battery_capacity = parsed_data['battery_capacity']
    cargo_capacity = parsed_data['cargo_capacity']
    
    # Create distance matrix
    dist_mat = create_distance_matrix(coordinates)
    
    # Initialize global pheromone trails
    global_pheromones = np.ones((dimensions, dimensions)) * 0.1
    
    # Initialize population
    population = [
        EV_Ant(dimensions, dist_mat, battery_capacity, cargo_capacity, stations, demands, depots) 
        for _ in range(population_size)
    ]
    
    # Best solution tracking
    gbest = None
    
    # Optimization loop
    for generation in range(num_generations):
        # Construct routes for each ant
        for ant in population:
            ant.route_construction(global_pheromones)
        
        # Update best solution
        current_best = min(population, key=lambda ant: ant.quality)
        if gbest is None or current_best.quality < gbest.quality:
            gbest = current_best
        
        # Pheromone update
        global_pheromones *= (1 - EVAPORATION_RATE)
        for ant in population:
            route_pheromone = PHEROMONE_DEPOSIT / ant.quality
            for i in range(len(ant.position) - 1):
                global_pheromones[ant.position[i]][ant.position[i+1]] += route_pheromone
    
    # Visualization of routes and pheromones
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.imshow(global_pheromones, cmap='YlGnBu')
    plt.colorbar(label='Pheromone Intensity')
    plt.title('Pheromone Trails')
    
    plt.subplot(122)
    coordinates_array = np.array(coordinates)
    plt.scatter(coordinates_array[:, 0], coordinates_array[:, 1], c='gray', alpha=0.5)
    
    best_route_coords = [coordinates[node] for node in gbest.position]
    best_route_coords = np.array(best_route_coords)
    
    plt.plot(best_route_coords[:, 0], best_route_coords[:, 1], 'r-', linewidth=2)
    
    # Highlight stations and depot
    for station in stations:
        plt.scatter(coordinates[station][0], coordinates[station][1], c='blue', marker='s')
    
    for depot in depots:
        plt.scatter(coordinates[depot][0], coordinates[depot][1], c='green', marker='*', s=200)
    
    plt.title('Best Route')
    plt.tight_layout()
    plt.show()
    
    return gbest.position, gbest.quality

# Solve the problem
file_path = 'e-cvrp_benchmark_instances-master/E-n35-k3-s5.evrp'
best_route, best_quality = ant_colony_optimization_evrp(file_path, num_generations=100, population_size=100, plot=True)
print(f"Best Route: {best_route}")
print(f"Route Quality: {best_quality}")
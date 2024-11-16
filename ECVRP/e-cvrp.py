import numpy as np
import re
# from scipy.spatial.distance import pdist, squareform

# Set to true to print traversal
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
                dist_mat[i][j] = float('inf')  # Avoid zero distance to the same node

    return dist_mat

# Objective function for the E-CVRP
def evrp_obj(x_ij, d_ij):
    total_distance = 0
    for i in range(len(x_ij)):
        for j in range(len(x_ij)):
            if x_ij[i][j] == 1:
                total_distance += d_ij[i][j]
    return total_distance

# Class defining candidate solutions a.k.a. agents
class EV_Ant:
    def __init__(self, number_of_nodes, dist_mat, battery_capacity, cargo_capacity, charging_stations, demands, depots):
        self.position = np.random.permutation(number_of_nodes)
        self.x_ij = np.zeros((number_of_nodes, number_of_nodes))
        self.battery_capacity = battery_capacity
        self.cargo_capacity = cargo_capacity
        self.charging_stations = charging_stations
        self.current_battery = battery_capacity
        self.current_cargo = cargo_capacity
        self.demands = demands
        self.depots = depots

        # Initial route construction
        self.route_construction(number_of_nodes, dist_mat)

    # def calculate_battery_consumption(self, from_node, to_node, dist_mat):
    #     # Calculate battery consumption between two nodes
    #     return dist_mat[from_node][to_node]

    def calculate_battery_consumption(self, from_node, to_node, dist_mat):
        distance = dist_mat[from_node][to_node]
        if DEBUG:
            print(f"Distance from node {from_node} to node {to_node}: {distance}")
        return distance

    def can_visit_next_node(self, current_node, next_node, dist_mat):
        # next_node = next_node - 1
        if next_node < 0 or next_node >= len(self.demands):
            return False

        battery_needed = self.calculate_battery_consumption(current_node, next_node, dist_mat)
        if DEBUG:
            print(f"Battery need to visit {next_node} is {battery_needed}")
        return self.current_battery >= battery_needed and self.current_cargo >= self.demands[next_node][1]

    def visit_node(self, current_node, next_node, dist_mat):
        # Debugging
        if DEBUG:
            print(f"Visiting from node: {current_node}, to node: {next_node}")
            print(f"Battery level: {self.current_battery}")
            print(f"Cargo: {self.current_cargo}")

        # Deduct the battery based on distance
        self.current_battery -= self.calculate_battery_consumption(current_node, next_node, dist_mat)

        # Deduct cargo only for customer nodes
        if next_node not in self.charging_stations and next_node not in self.depots:
            self.current_cargo -= self.demands[next_node][1]

        # Mark the path as visited
        self.x_ij[current_node][next_node] = 1

    def recharge_battery(self):
        # Recharge the battery to full capacity
        self.current_battery = self.battery_capacity

    def route_construction(self, number_of_nodes, dist_mat):
        # Resetting battery and cargo for the start of the route construction
        self.current_battery = self.battery_capacity
        self.current_cargo = self.cargo_capacity
        # Keep track of visited customer nodes
        visited_nodes = set()  

        # Start from the first node in the position array
        current_node = self.position[0]

        for i in range(1, number_of_nodes):
            # Next node to visit
            next_node = self.position[i]

            # If the next node is a depot or charging station, it's okay to revisit
            if next_node in self.depots or next_node in self.charging_stations:
                pass
            elif next_node in visited_nodes:
                # Skip already visited customer nodes
                continue

            # Check if the next node can be visited
            if self.can_visit_next_node(current_node, next_node, dist_mat):
                self.visit_node(current_node, next_node, dist_mat)
                if next_node not in self.depots and next_node not in self.charging_stations:
                    visited_nodes.add(next_node)  # Mark customer nodes as visited

            # Update the current node to the one just visited
            current_node = next_node

        # Returning to the starting depot at the end of the route
        start_depot = self.depots[0]
        self.visit_node(current_node, start_depot, dist_mat)

        # Calculate the quality of the constructed route
        self.quality = evrp_obj(self.x_ij, dist_mat)

# Ant Colony Optimization function for E-CVRP
def ant_colony_optimization_evrp(file_path):
    # Parse the data
    parsed_data = parse_evrp_data(file_path)

    # Extract data from the parsed data
    coordinates = parsed_data['coordinates']
    demands = parsed_data['demands']
    station_coords = parsed_data['station_coords']
    depots = parsed_data['depots']
    dimensions = parsed_data['dimension']
    battery_capacity = parsed_data['battery_capacity']
    cargo_capacity = parsed_data['cargo_capacity']
    stations = parsed_data['station_coords']
    depots = parsed_data['depots']

    
    # points = np.array([coords[1:] for coords in coordinates])
    # dist_mat = squareform(pdist(points, 'euclidean'))
    # Generate distance matrix
    dist_mat = create_distance_matrix(coordinates)

    # Population initialization
    gen = 10
    pop = 20

    print(f"Starting run with...\nGen: {gen}\tPop: {pop}\tDemands: {len(demands)}")

    population = [EV_Ant(dimensions, dist_mat, battery_capacity, cargo_capacity, stations, demands, depots) for _ in range(pop)]
    gbest = min(population, key=lambda ant: ant.quality)

    # Optimization loop
    for j in range(gen):
        for ant in population:
            ant.route_construction(dimensions, dist_mat)

        # Update the best solution
        iter_best = min(population, key=lambda ant: ant.quality)
        if iter_best.quality < gbest.quality:
            gbest = iter_best

    print(f"The best solution found: {gbest.position} with quality: {gbest.quality}")

# Set the path for E-CVRP data
# file_path = 'e-cvrp_benchmark_instances-master/E-n112-k8-s11.evrp'
file_path = 'e-cvrp_benchmark_instances-master/X-n469-k26-s10.evrp'
# Start the ant colony
ant_colony_optimization_evrp(file_path)
import numpy as np
import random
from scipy.spatial import Voronoi

# Constants
MAP_HEIGHT = 40
MAP_WIDTH = 50
TILE_TYPES = [0, 1, 2, 3, 4]  # 0: Empty, 1: Water, 2: Grass, 3: Rocks, 4: Mountains

# Update the terrain distribution constants
TERRAIN_DISTRIBUTION = {
    0: 0.05,  # Empty (5%)
    1: 0.15,  # Water (15%)
    2: 0.50,  # Grass (50%)
    3: 0.20,  # Rocks (20%)
    4: 0.10   # Mountains (10%)
}

def terrain_transition_score(tile1, tile2):
    """Score the naturalness of terrain transitions."""
    # Natural transitions table (higher score = more natural)
    transitions = {
        (1, 2): 5,  # Grass next to water is natural
        (2, 1): 5,
        (1, 3): 3,  # Grass to rocks is okay
        (3, 1): 3,
        (3, 4): 4,  # Rocks to mountains is natural
        (4, 3): 4,
        (2, 4): -5,  # Water next to mountains is unnatural
        (4, 2): -5
    }
    return transitions.get((tile1, tile2), 0)

def check_river_flow(map_grid, row, col):
    """Check if river tile has proper flow (should connect to at least one other river)."""
    neighbors = get_neighbors(map_grid, row, col)
    river_count = sum(1 for n in neighbors if n == 2)
    return river_count >= 1

# Fitness function
def fitness_function(map_grid):
    """Enhanced fitness function with stronger distribution penalties."""
    score = 0
    
    # Distribution scoring (increased weight)
    unique, counts = np.unique(map_grid, return_counts=True)
    distribution = dict(zip(unique, counts))
    total_tiles = MAP_HEIGHT * MAP_WIDTH
    
    distribution_score = 0
    for tile_type, desired_ratio in TERRAIN_DISTRIBUTION.items():
        current_ratio = distribution.get(tile_type, 0) / total_tiles
        # Increased penalty for distribution mismatch
        distribution_score -= abs(current_ratio - desired_ratio) * 500
    score += distribution_score
    
    # Water connectivity scoring
    water_connectivity_score = check_water_connectivity(map_grid)
    score += water_connectivity_score * 2
    
    # Add existing scoring components
    # 1. River continuity and flow
    river_score = 0
    for row in range(MAP_HEIGHT):
        for col in range(MAP_WIDTH):
            if map_grid[row][col] == 2:  # Water tile
                if check_river_flow(map_grid, row, col):
                    river_score += 5
                else:
                    river_score -= 20
    score += river_score

    # 2. Terrain transitions
    transition_score = 0
    for row in range(MAP_HEIGHT):
        for col in range(MAP_WIDTH):
            current = map_grid[row][col]
            # Check horizontal and vertical neighbors
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                if 0 <= row+dr < MAP_HEIGHT and 0 <= col+dc < MAP_WIDTH:
                    neighbor = map_grid[row+dr][col+dc]
                    transition_score += terrain_transition_score(current, neighbor)
    score += transition_score

    # 3. Biome clustering
    cluster_score = 0
    cluster_score += count_clusters(map_grid, 1) * 10  # Grass clusters
    cluster_score += count_clusters(map_grid, 4) * 15  # Mountain clusters
    score += cluster_score

    # 4. Border constraints
    border_score = 0
    for i in range(MAP_HEIGHT):
        if map_grid[i][0] == 2 or map_grid[i][-1] == 2:  # Rivers shouldn't start at borders
            border_score -= 10
    score += border_score

    return score

def get_neighbors(grid, row, col):
    """Get the neighbors of a cell in the grid."""
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = row + dr, col + dc
        if 0 <= r < MAP_HEIGHT and 0 <= c < MAP_WIDTH:
            neighbors.append(grid[r][c])
    return neighbors

def count_clusters(grid, tile_type):
    """Count clusters of a specific tile type using an iterative approach."""
    visited = np.zeros_like(grid, dtype=bool)
    clusters = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def get_cluster_size(start_row, start_col):
        if visited[start_row][start_col] or grid[start_row][start_col] != tile_type:
            return 0
            
        # Use a stack for iterative flood fill
        stack = [(start_row, start_col)]
        size = 0
        
        while stack:
            r, c = stack.pop()
            if visited[r][c] or grid[r][c] != tile_type:
                continue
                
            visited[r][c] = True
            size += 1
            
            # Add unvisited neighbors
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                if (0 <= new_r < MAP_HEIGHT and 
                    0 <= new_c < MAP_WIDTH and 
                    not visited[new_r][new_c] and 
                    grid[new_r][new_c] == tile_type):
                    stack.append((new_r, new_c))
        
        return size

    # Find all clusters
    for row in range(MAP_HEIGHT):
        for col in range(MAP_WIDTH):
            if not visited[row][col] and grid[row][col] == tile_type:
                cluster_size = get_cluster_size(row, col)
                if cluster_size > 0:
                    clusters += 1

    return clusters

def create_water_body(map_grid, start_row, start_col, size):
    """Create a connected water body (river or lake)."""
    stack = [(start_row, start_col)]
    water_tiles = set()
    
    while stack and len(water_tiles) < size:
        row, col = stack.pop(0)
        if (row, col) in water_tiles:
            continue
            
        map_grid[row][col] = 1  # Water
        water_tiles.add((row, col))
        
        # Add neighboring cells with probability
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < MAP_HEIGHT and 
                0 <= new_col < MAP_WIDTH and 
                (new_row, new_col) not in water_tiles):
                if random.random() < 0.7:  # High probability for connected water
                    stack.append((new_row, new_col))

def check_water_connectivity(map_grid):
    """Check if water bodies are well-connected and return a score."""
    visited = np.zeros_like(map_grid, dtype=bool)
    water_groups = []
    
    def flood_fill(row, col):
        if (row < 0 or row >= MAP_HEIGHT or col < 0 or col >= MAP_WIDTH or
            visited[row][col] or map_grid[row][col] != 1):
            return 0
        
        size = 1
        visited[row][col] = True
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            size += flood_fill(row + dr, col + dc)
        return size
    
    for row in range(MAP_HEIGHT):
        for col in range(MAP_WIDTH):
            if not visited[row][col] and map_grid[row][col] == 1:
                group_size = flood_fill(row, col)
                water_groups.append(group_size)
    
    if not water_groups:
        return 0
    
    # Penalize many small water bodies
    return -len(water_groups) * 10 + max(water_groups)

def create_river_system(map_grid, start_row, start_col):
    """Create a natural river system with tributaries."""
    queue = [(start_row, start_col, 1.0)]  # Include flow strength
    river_tiles = set()
    
    while queue and len(river_tiles) < MAP_HEIGHT * 2:  # Limit river length
        row, col, strength = queue.pop(0)
        if (row, col) in river_tiles or strength < 0.2:
            continue
            
        map_grid[row][col] = 1  # Water
        river_tiles.add((row, col))
        
        # River flow directions with weights (prefer moving down/sideways)
        directions = [
            (1, 0, 0.8),    # Down
            (1, -1, 0.6),   # Down-left
            (1, 1, 0.6),    # Down-right
            (0, -1, 0.4),   # Left
            (0, 1, 0.4),    # Right
            (-1, 0, 0.2)    # Up (rare)
        ]
        
        # Add branches with decreasing strength
        for dr, dc, flow_mod in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < MAP_HEIGHT and 
                0 <= new_col < MAP_WIDTH and 
                (new_row, new_col) not in river_tiles):
                if random.random() < strength * flow_mod:
                    queue.append((new_row, new_col, strength * 0.9))
                    # Add lake-like features occasionally
                    if random.random() < 0.1:
                        for lake_dr, lake_dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            lake_r, lake_c = new_row + lake_dr, new_col + lake_dc
                            if (0 <= lake_r < MAP_HEIGHT and 
                                0 <= lake_c < MAP_WIDTH and 
                                (lake_r, lake_c) not in river_tiles):
                                map_grid[lake_r][lake_c] = 1

def create_mountain_range(map_grid, start_row, start_col, size):
    """Create a natural mountain range with surrounding terrain."""
    stack = [(start_row, start_col, 1.0)]  # Include height factor
    mountain_tiles = set()
    
    while stack and len(mountain_tiles) < size:
        row, col, height = stack.pop(0)
        if (row, col) in mountain_tiles:
            continue
            
        # Place mountains and surrounding rocks based on height
        if height > 0.7:
            map_grid[row][col] = 4  # Mountain
            mountain_tiles.add((row, col))
        elif height > 0.4:
            map_grid[row][col] = 3  # Rocks
        
        # Spread in random directions with decreasing height
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        random.shuffle(directions)
        
        for dr, dc in directions[:4]:  # Limit spread directions
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < MAP_HEIGHT and 
                0 <= new_col < MAP_WIDTH and 
                (new_row, new_col) not in mountain_tiles):
                new_height = height * random.uniform(0.8, 0.95)
                if new_height > 0.3:  # Minimum height threshold
                    stack.append((new_row, new_col, new_height))

class VoronoiTerrainGenerator:
    def __init__(self, width, height, num_points=50):
        self.width = width
        self.height = height
        self.num_points = num_points
        
    def generate_points(self):
        """Generate random points for Voronoi regions"""
        points = []
        # Add points with padding to avoid edge effects
        padding = 10
        for _ in range(self.num_points):
            x = random.uniform(-padding, self.width + padding)
            y = random.uniform(-padding, self.height + padding)
            points.append([x, y])
        return np.array(points)
    
    def assign_terrain_types(self, vor, elevation_noise):
        """Assign terrain types to Voronoi regions based on elevation"""
        region_types = {}
        for idx, region in enumerate(vor.regions):
            if -1 not in region and len(region) > 0:
                # Get center point of region
                center = np.mean([vor.vertices[i] for i in region], axis=0)
                # Combine Voronoi with noise for more natural boundaries
                elev = elevation_noise[int(center[1]) % self.height,
                                    int(center[0]) % self.width]
                
                # Assign terrain type based on elevation
                if elev < 0.2:
                    region_types[idx] = 1  # Water
                elif elev < 0.4:
                    region_types[idx] = 2  # Grass
                elif elev < 0.7:
                    region_types[idx] = 3  # Rocks
                else:
                    region_types[idx] = 4  # Mountains
                    
        return region_types

    def generate_terrain(self):
        """Generate terrain using Voronoi diagram"""
        points = self.generate_points()
        vor = Voronoi(points)
        
        # Generate fractal noise for elevation
        elevation = self.generate_fractal_noise()
        
        # Create terrain map
        terrain = np.zeros((self.height, self.width), dtype=int)
        region_types = self.assign_terrain_types(vor, elevation)
        
        # Fill the map
        for y in range(self.height):
            for x in range(self.width):
                point = [x, y]
                region = self.find_region(vor, point)
                if region in region_types:
                    terrain[y, x] = region_types[region]
                else:
                    terrain[y, x] = 2  # Default to grass
                    
        return terrain
    
    def generate_fractal_noise(self, octaves=6):
        """Generate fractal noise for more natural terrain"""
        noise = np.zeros((self.height, self.width))
        frequency = 1
        amplitude = 1
        persistence = 0.5
        
        for _ in range(octaves):
            for y in range(self.height):
                for x in range(self.width):
                    noise[y,x] += amplitude * random.uniform(0, 1)
            frequency *= 2
            amplitude *= persistence
            
        # Normalize
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        return noise
    
    def find_region(self, vor, point):
        """Find which Voronoi region a point belongs to"""
        distances = []
        for idx, vertex in enumerate(vor.points):
            dist = np.sqrt((point[0]-vertex[0])**2 + (point[1]-vertex[1])**2)
            distances.append((dist, idx))
        return vor.point_region[min(distances, key=lambda x: x[0])[1]]

def generate_population(pop_size):
    """Generate population using Voronoi-based terrain generation"""
    population = []
    terrain_gen = VoronoiTerrainGenerator(MAP_WIDTH, MAP_HEIGHT)
    
    for _ in range(pop_size):
        # Generate base terrain using Voronoi
        terrain = terrain_gen.generate_terrain()
        
        # Smooth transitions
        smoothed = terrain.copy()
        for row in range(1, MAP_HEIGHT-1):
            for col in range(1, MAP_WIDTH-1):
                # Get surrounding tiles
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        neighbors.append(terrain[row+dr, col+dc])
                
                # Smooth based on neighbors
                current = terrain[row, col]
                if current == 2:  # Grass
                    if neighbors.count(3) >= 5:
                        smoothed[row, col] = 3
                elif current == 1:  # Water
                    if neighbors.count(1) <= 2:
                        smoothed[row, col] = 2
        
        population.append(smoothed)
    
    return population

def mutate(map_grid):
    """Smarter mutation that considers terrain rules."""
    mutated = map_grid.copy()
    num_mutations = random.randint(1, 5)
    
    for _ in range(num_mutations):
        row = random.randint(0, MAP_HEIGHT - 1)
        col = random.randint(0, MAP_WIDTH - 1)
        current = mutated[row][col]
        
        # Get neighboring tiles
        neighbors = get_neighbors(mutated, row, col)
        
        # Smart tile selection based on neighbors
        if 2 in neighbors:  # Next to water
            new_tile = random.choice([1, 2])  # Prefer grass or water
        elif 4 in neighbors:  # Next to mountains
            new_tile = random.choice([3, 4])  # Prefer rocks or mountains
        else:
            new_tile = random.choice(TILE_TYPES)
        
        mutated[row][col] = new_tile
    
    return mutated

def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    crossover_point = random.randint(0, MAP_HEIGHT - 1)
    child = np.vstack((parent1[:crossover_point, :], parent2[crossover_point:, :]))
    return child

def evolve_population(population, generations, mutation_rate=0.2):
    """Evolve the population over generations."""
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [(map_grid, fitness_function(map_grid)) for map_grid in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness
        population = [x[0] for x in fitness_scores[:len(population) // 2]]  # Select top 50%

        # Create next generation
        next_generation = []
        while len(next_generation) < len(population) * 2:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            next_generation.append(child)

        population = next_generation
        best_map = population[0]
        print(f"Generation {generation + 1}, Best Fitness: {fitness_function(best_map)}")

    return best_map

# Save the generated landscape to a file
def save_landscape_to_file(landscape, filename="generated_landscape.txt"):
    with open(filename, "w") as f:
        for row in landscape:
            f.write("".join(map(str, row)) + "\n")
    print(f"Landscape saved to {filename}")

for i in range(10):
    population = generate_population(pop_size=5*(i+1)+5)
    best_landscape = evolve_population(population, generations=50)
    save_landscape_to_file(best_landscape, filename=f"generated_landscape_{i}.txt")
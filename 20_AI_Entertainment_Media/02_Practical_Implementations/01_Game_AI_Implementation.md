# Game AI Implementation: Practical Guide

## 🎮 Game AI: From Theory to Practice

Game AI encompasses a wide range of techniques from simple rule-based systems to complex neural networks and reinforcement learning. This implementation guide provides hands-on examples for building intelligent game AI systems.

## 🛠️ Setup and Installation

### **Required Libraries**
```bash
# Game development libraries
pip install pygame
pip install pymunk
pip install panda3d

# AI and ML libraries
pip install torch
pip install tensorflow
pip install gymnasium
pip install stable-baselines3

# Computer vision for games
pip install opencv-python
pip install mediapipe

# Data processing
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
```

## 📊 Implementation Examples

### **1. NPC Behavior System with State Machines**

```python
import pygame
import numpy as np
from enum import Enum
from typing import List, Tuple, Dict
import math

# NPC States
class NPCState(Enum):
    IDLE = "idle"
    PATROL = "patrol"
    CHASE = "chase"
    ATTACK = "attack"
    FLEE = "flee"
    SEARCH = "search"

# NPC Behavior System
class NPCBehavior:
    def __init__(self, npc, game_world):
        self.npc = npc
        self.game_world = game_world
        self.current_state = NPCState.IDLE
        self.state_machine = StateMachine()
        self.perception = NPCPerception(npc, game_world)
        self.memory = NPCMemory()
        self.personality = NPCPersonality()

        # Behavior parameters
        self.detection_range = 150
        self.attack_range = 50
        self.flee_threshold = 0.3
        self.patrol_points = []
        self.current_patrol_index = 0

        # Initialize state machine
        self.setup_state_machine()

    def setup_state_machine(self):
        """Setup NPC state machine"""
        # Idle state
        self.state_machine.add_state(
            NPCState.IDLE,
            on_enter=self.on_idle_enter,
            on_update=self.on_idle_update,
            on_exit=self.on_idle_exit
        )

        # Patrol state
        self.state_machine.add_state(
            NPCState.PATROL,
            on_enter=self.on_patrol_enter,
            on_update=self.on_patrol_update,
            on_exit=self.on_patrol_exit
        )

        # Chase state
        self.state_machine.add_state(
            NPCState.CHASE,
            on_enter=self.on_chase_enter,
            on_update=self.on_chase_update,
            on_exit=self.on_chase_exit
        )

        # Attack state
        self.state_machine.add_state(
            NPCState.ATTACK,
            on_enter=self.on_attack_enter,
            on_update=self.on_attack_update,
            on_exit=self.on_attack_exit
        )

        # Flee state
        self.state_machine.add_state(
            NPCState.FLEE,
            on_enter=self.on_flee_enter,
            on_update=self.on_flee_update,
            on_exit=self.on_flee_exit
        )

        # Search state
        self.state_machine.add_state(
            NPCState.SEARCH,
            on_enter=self.on_search_enter,
            on_update=self.on_search_update,
            on_exit=self.on_search_exit
        )

        # Set initial state
        self.state_machine.set_state(NPCState.IDLE)

    def update(self, dt):
        """Update NPC behavior"""
        # Perception update
        self.perception.update()

        # Decision making
        self.make_decision()

        # State machine update
        self.state_machine.update(dt)

        # Memory update
        self.memory.update()

    def make_decision(self):
        """Make behavioral decisions"""
        # Get perceived information
        perceived_threats = self.perception.get_threats()
        perceived_targets = self.perception.get_targets()
        health_ratio = self.npc.health / self.npc.max_health

        # Decision logic based on personality and perception
        if health_ratio < self.flee_threshold:
            self.state_machine.set_state(NPCState.FLEE)
        elif perceived_threats:
            self.state_machine.set_state(NPCState.FLEE)
        elif perceived_targets:
            target = perceived_targets[0]
            distance = self.calculate_distance(self.npc.position, target.position)

            if distance < self.attack_range:
                self.state_machine.set_state(NPCState.ATTACK)
            else:
                self.state_machine.set_state(NPCState.CHASE)
        elif self.memory.has_memory_of("last_seen_target"):
            self.state_machine.set_state(NPCState.SEARCH)
        elif self.patrol_points:
            self.state_machine.set_state(NPCState.PATROL)
        else:
            self.state_machine.set_state(NPCState.IDLE)

    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two points"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    # State handlers
    def on_idle_enter(self):
        """Called when entering idle state"""
        self.npc.velocity = [0, 0]
        self.idle_timer = 0

    def on_idle_update(self, dt):
        """Called during idle state update"""
        self.idle_timer += dt
        if self.idle_timer > 3.0:  # Idle for 3 seconds
            if self.patrol_points:
                self.state_machine.set_state(NPCState.PATROL)

    def on_idle_exit(self):
        """Called when exiting idle state"""
        pass

    def on_patrol_enter(self):
        """Called when entering patrol state"""
        if self.patrol_points:
            self.target_patrol_point = self.patrol_points[self.current_patrol_index]

    def on_patrol_update(self, dt):
        """Called during patrol state update"""
        if self.target_patrol_point:
            # Move towards patrol point
            direction = [
                self.target_patrol_point[0] - self.npc.position[0],
                self.target_patrol_point[1] - self.npc.position[1]
            ]

            distance = math.sqrt(direction[0]**2 + direction[1]**2)

            if distance > 5:
                # Normalize direction
                direction = [direction[0] / distance, direction[1] / distance]
                self.npc.velocity = [direction[0] * self.npc.speed, direction[1] * self.npc.speed]
            else:
                # Reached patrol point, move to next
                self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_points)
                self.target_patrol_point = self.patrol_points[self.current_patrol_index]

    def on_patrol_exit(self):
        """Called when exiting patrol state"""
        pass

    def on_chase_enter(self):
        """Called when entering chase state"""
        self.chase_target = self.perception.get_targets()[0]

    def on_chase_update(self, dt):
        """Called during chase state update"""
        if self.chase_target:
            # Calculate direction to target
            direction = [
                self.chase_target.position[0] - self.npc.position[0],
                self.chase_target.position[1] - self.npc.position[1]
            ]

            distance = math.sqrt(direction[0]**2 + direction[1]**2)

            if distance > self.attack_range:
                # Move towards target
                direction = [direction[0] / distance, direction[1] / distance]
                self.npc.velocity = [direction[0] * self.npc.speed * 1.5, direction[1] * self.npc.speed * 1.5]
            else:
                # Close enough to attack
                self.state_machine.set_state(NPCState.ATTACK)

    def on_chase_exit(self):
        """Called when exiting chase state"""
        self.chase_target = None

    def on_attack_enter(self):
        """Called when entering attack state"""
        self.attack_target = self.perception.get_targets()[0]
        self.attack_cooldown = 0

    def on_attack_update(self, dt):
        """Called during attack state update"""
        self.attack_cooldown += dt

        if self.attack_cooldown > 1.0:  # Attack every 1 second
            if self.attack_target:
                # Perform attack
                damage = self.npc.attack_power
                self.attack_target.take_damage(damage)
                self.attack_cooldown = 0

                # Check if target is still in range
                distance = self.calculate_distance(self.npc.position, self.attack_target.position)
                if distance > self.attack_range * 1.5:
                    self.state_machine.set_state(NPCState.CHASE)

    def on_attack_exit(self):
        """Called when exiting attack state"""
        self.attack_target = None

    def on_flee_enter(self):
        """Called when entering flee state"""
        self.flee_direction = self.calculate_flee_direction()
        self.flee_timer = 0

    def on_flee_update(self, dt):
        """Called during flee state update"""
        self.flee_timer += dt

        # Flee in calculated direction
        self.npc.velocity = [self.flee_direction[0] * self.npc.speed * 2,
                            self.flee_direction[1] * self.npc.speed * 2]

        if self.flee_timer > 5.0 or self.npc.health / self.npc.max_health > 0.6:
            self.state_machine.set_state(NPCState.IDLE)

    def on_flee_exit(self):
        """Called when exiting flee state"""
        pass

    def on_search_enter(self):
        """Called when entering search state"""
        self.search_center = self.memory.get_memory("last_seen_target_position")
        self.search_radius = 100
        self.search_timer = 0

    def on_search_update(self, dt):
        """Called during search state update"""
        self.search_timer += dt

        # Search in expanding circles
        search_angle = self.search_timer * 2  # 2 radians per second
        search_distance = min(self.search_timer * 20, self.search_radius)

        target_x = self.search_center[0] + math.cos(search_angle) * search_distance
        target_y = self.search_center[1] + math.sin(search_angle) * search_distance

        direction = [target_x - self.npc.position[0], target_y - self.npc.position[1]]
        distance = math.sqrt(direction[0]**2 + direction[1]**2)

        if distance > 5:
            direction = [direction[0] / distance, direction[1] / distance]
            self.npc.velocity = [direction[0] * self.npc.speed, direction[1] * self.npc.speed]

        if self.search_timer > 10.0:  # Search for 10 seconds
            self.state_machine.set_state(NPCState.IDLE)

    def on_search_exit(self):
        """Called when exiting search state"""
        pass

    def calculate_flee_direction(self):
        """Calculate optimal flee direction"""
        threats = self.perception.get_threats()
        if not threats:
            return [1, 0]  # Default direction

        # Calculate center of threats
        center_x = sum(threat.position[0] for threat in threats) / len(threats)
        center_y = sum(threat.position[1] for threat in threats) / len(threats)

        # Flee direction is opposite to threat center
        direction = [self.npc.position[0] - center_x, self.npc.position[1] - center_y]
        distance = math.sqrt(direction[0]**2 + direction[1]**2)

        if distance > 0:
            return [direction[0] / distance, direction[1] / distance]
        else:
            return [1, 0]

# State Machine Implementation
class StateMachine:
    def __init__(self):
        self.states = {}
        self.current_state = None
        self.state_data = {}

    def add_state(self, state_name, on_enter=None, on_update=None, on_exit=None):
        """Add state to state machine"""
        self.states[state_name] = {
            'on_enter': on_enter,
            'on_update': on_update,
            'on_exit': on_exit
        }

    def set_state(self, state_name):
        """Set current state"""
        if state_name in self.states:
            # Exit current state
            if self.current_state and self.current_state != state_name:
                current_state_data = self.states[self.current_state]
                if current_state_data['on_exit']:
                    current_state_data['on_exit']()

            # Enter new state
            self.current_state = state_name
            new_state_data = self.states[state_name]
            if new_state_data['on_enter']:
                new_state_data['on_enter']()

    def update(self, dt):
        """Update current state"""
        if self.current_state and self.current_state in self.states:
            state_data = self.states[self.current_state]
            if state_data['on_update']:
                state_data['on_update'](dt)

# NPC Perception System
class NPCPerception:
    def __init__(self, npc, game_world):
        self.npc = npc
        self.game_world = game_world
        self.perceived_entities = {}
        self.last_perception_update = 0
        self.perception_update_interval = 0.1  # Update every 0.1 seconds

    def update(self):
        """Update perception system"""
        current_time = pygame.time.get_ticks() / 1000.0
        if current_time - self.last_perception_update > self.perception_update_interval:
            self.update_perceived_entities()
            self.last_perception_update = current_time

    def update_perceived_entities(self):
        """Update list of perceived entities"""
        self.perceived_entities.clear()

        for entity in self.game_world.entities:
            if entity != self.npc:
                distance = self.calculate_distance(self.npc.position, entity.position)

                # Check if entity is within perception range
                if distance <= self.npc.behavior.detection_range:
                    # Line of sight check
                    if self.has_line_of_sight(self.npc.position, entity.position):
                        self.perceived_entities[entity.id] = {
                            'entity': entity,
                            'distance': distance,
                            'last_seen': current_time
                        }

    def has_line_of_sight(self, pos1, pos2):
        """Check if there's line of sight between two positions"""
        # Simplified line of sight - in practice, you'd check for obstacles
        return True

    def get_threats(self):
        """Get list of perceived threats"""
        threats = []
        for entity_id, perception_data in self.perceived_entities.items():
            entity = perception_data['entity']
            if hasattr(entity, 'faction') and entity.faction != self.npc.faction:
                threats.append(entity)
        return threats

    def get_targets(self):
        """Get list of perceived targets"""
        targets = []
        for entity_id, perception_data in self.perceived_entities.items():
            entity = perception_data['entity']
            if hasattr(entity, 'faction') and entity.faction != self.npc.faction:
                targets.append(entity)
        return targets

    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two points"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

# NPC Memory System
class NPCMemory:
    def __init__(self):
        self.memories = {}
        self.memory_decay_rate = 0.01  # Memories decay over time

    def update(self):
        """Update memory system"""
        # Decay memories
        memories_to_remove = []
        for memory_id, memory_data in self.memories.items():
            memory_data['strength'] -= self.memory_decay_rate
            if memory_data['strength'] <= 0:
                memories_to_remove.append(memory_id)

        for memory_id in memories_to_remove:
            del self.memories[memory_id]

    def add_memory(self, memory_type, data, strength=1.0):
        """Add new memory"""
        memory_id = f"{memory_type}_{pygame.time.get_ticks()}"
        self.memories[memory_id] = {
            'type': memory_type,
            'data': data,
            'strength': strength,
            'timestamp': pygame.time.get_ticks() / 1000.0
        }

    def has_memory_of(self, memory_type):
        """Check if NPC has memory of specific type"""
        return any(memory['type'] == memory_type for memory in self.memories.values())

    def get_memory(self, memory_type):
        """Get memory of specific type"""
        for memory_data in self.memories.values():
            if memory_data['type'] == memory_type:
                return memory_data['data']
        return None

# NPC Personality System
class NPCPersonality:
    def __init__(self):
        self.traits = {
            'aggression': 0.5,      # How aggressive the NPC is
            'caution': 0.5,        # How cautious the NPC is
            'curiosity': 0.5,      # How curious the NPC is
            'loyalty': 0.5,         # How loyal to allies
            'fear': 0.5            # How easily scared
        }

    def set_trait(self, trait_name, value):
        """Set personality trait value"""
        if trait_name in self.traits:
            self.traits[trait_name] = max(0.0, min(1.0, value))

    def get_trait(self, trait_name):
        """Get personality trait value"""
        return self.traits.get(trait_name, 0.5)

# Example NPC Class
class NPC:
    def __init__(self, x, y, name="NPC"):
        self.position = [x, y]
        self.velocity = [0, 0]
        self.name = name
        self.health = 100
        self.max_health = 100
        self.speed = 50
        self.attack_power = 10
        self.faction = "neutral"
        self.behavior = NPCBehavior(self, None)  # Will be set by game world

    def update(self, dt):
        """Update NPC"""
        # Update behavior
        if self.behavior:
            self.behavior.update(dt)

        # Update position based on velocity
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt

    def take_damage(self, damage):
        """NPC takes damage"""
        self.health -= damage
        self.health = max(0, self.health)

        if self.health <= 0:
            self.on_death()

    def on_death(self):
        """NPC death logic"""
        print(f"{self.name} has died!")
        # Remove from game world, play death animation, etc.

# Simple Game World
class GameWorld:
    def __init__(self):
        self.entities = []
        self.npcs = []

    def add_entity(self, entity):
        """Add entity to game world"""
        self.entities.append(entity)
        if isinstance(entity, NPC):
            entity.behavior.game_world = self
            self.npcs.append(entity)

    def update(self, dt):
        """Update game world"""
        for entity in self.entities:
            if hasattr(entity, 'update'):
                entity.update(dt)

# Example Usage
def create_game_example():
    """Create example game with NPCs"""
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Game AI Example")
    clock = pygame.time.Clock()

    # Create game world
    world = GameWorld()

    # Create NPCs
    npc1 = NPC(100, 100, "Guard")
    npc1.behavior.patrol_points = [(200, 100), (200, 200), (100, 200)]
    npc1.faction = "guards"

    npc2 = NPC(400, 300, "Bandit")
    npc2.faction = "bandits"
    npc2.behavior.detection_range = 200
    npc2.behavior.attack_range = 30

    world.add_entity(npc1)
    world.add_entity(npc2)

    # Game loop
    running = True
    dt = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update
        world.update(dt)

        # Render
        screen.fill((0, 0, 0))  # Black background

        # Draw NPCs
        for npc in world.npcs:
            pygame.draw.circle(screen, (255, 0, 0), (int(npc.position[0]), int(npc.position[1])), 10)

        pygame.display.flip()
        dt = clock.tick(60) / 1000.0  # 60 FPS

    pygame.quit()

if __name__ == "__main__":
    create_game_example()
```

### **2. Procedural Content Generation for Games**

```python
import numpy as np
import random
from typing import List, Tuple, Dict
import math
from enum import Enum

class TerrainType(Enum):
    GRASS = "grass"
    WATER = "water"
    MOUNTAIN = "mountain"
    FOREST = "forest"
    DESERT = "desert"
    SNOW = "snow"

# Procedural Terrain Generation
class TerrainGenerator:
    def __init__(self, width, height, seed=None):
        self.width = width
        self.height = height
        self.seed = seed
        if seed:
            random.seed(seed)
            np.random.seed(seed)

    def generate_terrain(self, terrain_type="mixed"):
        """Generate terrain based on type"""
        if terrain_type == "island":
            return self.generate_island()
        elif terrain_type == "mountains":
            return self.generate_mountains()
        elif terrain_type == "forest":
            return self.generate_forest()
        elif terrain_type == "desert":
            return self.generate_desert()
        else:
            return self.generate_mixed_terrain()

    def generate_mixed_terrain(self):
        """Generate mixed terrain using Perlin noise"""
        # Create height map using Perlin noise
        height_map = self.generate_perlin_noise(self.width, self.height, scale=0.1)

        # Create temperature map
        temperature_map = self.generate_perlin_noise(self.width, self.height, scale=0.05)

        # Create moisture map
        moisture_map = self.generate_perlin_noise(self.width, self.height, scale=0.08)

        # Generate terrain based on height, temperature, and moisture
        terrain_map = np.zeros((self.height, self.width), dtype=int)

        for y in range(self.height):
            for x in range(self.width):
                height = height_map[y, x]
                temperature = temperature_map[y, x]
                moisture = moisture_map[y, x]

                # Determine terrain type based on conditions
                if height < 0.3:
                    if moisture > 0.6:
                        terrain_map[y, x] = TerrainType.WATER.value
                    else:
                        terrain_map[y, x] = TerrainType.GRASS.value
                elif height < 0.5:
                    if temperature < 0.3 and moisture > 0.4:
                        terrain_map[y, x] = TerrainType.FOREST.value
                    elif temperature > 0.7:
                        terrain_map[y, x] = TerrainType.DESERT.value
                    else:
                        terrain_map[y, x] = TerrainType.GRASS.value
                elif height < 0.8:
                    terrain_map[y, x] = TerrainType.MOUNTAIN.value
                else:
                    if temperature < 0.2:
                        terrain_map[y, x] = TerrainType.SNOW.value
                    else:
                        terrain_map[y, x] = TerrainType.MOUNTAIN.value

        return terrain_map

    def generate_island(self):
        """Generate island terrain"""
        # Create circular island shape
        terrain_map = np.zeros((self.height, self.width), dtype=int)

        center_x, center_y = self.width // 2, self.height // 2
        max_radius = min(self.width, self.height) // 3

        for y in range(self.height):
            for x in range(self.width):
                # Calculate distance from center
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)

                # Create island shape with noise
                noise_value = self.generate_perlin_noise_at_point(x, y, scale=0.1)
                island_radius = max_radius + noise_value * 50

                if distance < island_radius:
                    if distance < island_radius * 0.3:
                        terrain_map[y, x] = TerrainType.GRASS.value
                    elif distance < island_radius * 0.7:
                        terrain_map[y, x] = TerrainType.FOREST.value
                    else:
                        terrain_map[y, x] = TerrainType.GRASS.value
                else:
                    terrain_map[y, x] = TerrainType.WATER.value

        return terrain_map

    def generate_mountains(self):
        """Generate mountain terrain"""
        height_map = self.generate_perlin_noise(self.width, self.height, scale=0.05)
        terrain_map = np.zeros((self.height, self.width), dtype=int)

        for y in range(self.height):
            for x in range(self.width):
                height = height_map[y, x]
                if height < 0.2:
                    terrain_map[y, x] = TerrainType.GRASS.value
                elif height < 0.5:
                    terrain_map[y, x] = TerrainType.FOREST.value
                else:
                    terrain_map[y, x] = TerrainType.MOUNTAIN.value

        return terrain_map

    def generate_forest(self):
        """Generate forest terrain"""
        height_map = self.generate_perlin_noise(self.width, self.height, scale=0.08)
        terrain_map = np.zeros((self.height, self.width), dtype=int)

        for y in range(self.height):
            for x in range(self.width):
                height = height_map[y, x]
                if height < 0.3:
                    terrain_map[y, x] = TerrainType.GRASS.value
                else:
                    terrain_map[y, x] = TerrainType.FOREST.value

        return terrain_map

    def generate_desert(self):
        """Generate desert terrain"""
        height_map = self.generate_perlin_noise(self.width, self.height, scale=0.06)
        terrain_map = np.zeros((self.height, self.width), dtype=int)

        for y in range(self.height):
            for x in range(self.width):
                height = height_map[y, x]
                if height < 0.4:
                    terrain_map[y, x] = TerrainType.DESERT.value
                else:
                    terrain_map[y, x] = TerrainType.MOUNTAIN.value

        return terrain_map

    def generate_perlin_noise(self, width, height, scale=1.0, octaves=4):
        """Generate Perlin noise"""
        noise = np.zeros((height, width))

        for i in range(octaves):
            freq = scale * (2 ** i)
            amp = 1.0 / (2 ** i)

            for y in range(height):
                for x in range(width):
                    noise[y, x] += amp * self.generate_perlin_noise_at_point(
                        x * freq, y * freq, scale=1.0
                    )

        return noise

    def generate_perlin_noise_at_point(self, x, y, scale=1.0):
        """Generate Perlin noise at specific point"""
        # Simplified Perlin noise implementation
        x *= scale
        y *= scale

        # Grid coordinates
        x0 = int(x)
        x1 = x0 + 1
        y0 = int(y)
        y1 = y0 + 1

        # Interpolation
        sx = x - x0
        sy = y - y0

        # Smooth interpolation
        sx = sx * sx * (3 - 2 * sx)
        sy = sy * sy * (3 - 2 * sy)

        # Corner values
        n0 = self.gradient_noise(x0, y0)
        n1 = self.gradient_noise(x1, y0)
        n2 = self.gradient_noise(x0, y1)
        n3 = self.gradient_noise(x1, y1)

        # Interpolation
        ix0 = n0 + sx * (n1 - n0)
        ix1 = n2 + sx * (n3 - n2)
        value = ix0 + sy * (ix1 - ix0)

        return value

    def gradient_noise(self, x, y):
        """Generate gradient noise"""
        # Simplified gradient noise
        angle = self.hash(x, y) * 2 * math.pi
        return math.cos(angle) + math.sin(angle)

    def hash(self, x, y):
        """Hash function for noise generation"""
        h = (x * 374761393 + y * 668265263) % 2147483647
        return h / 2147483647.0

# Dungeon Generation
class DungeonGenerator:
    def __init__(self, width, height, seed=None):
        self.width = width
        self.height = height
        self.seed = seed
        if seed:
            random.seed(seed)

    def generate_dungeon(self, dungeon_type="random"):
        """Generate dungeon based on type"""
        if dungeon_type == "rooms":
            return self.generate_rooms_dungeon()
        elif dungeon_type == "caves":
            return self.generate_caves_dungeon()
        elif dungeon_type == "maze":
            return self.generate_maze_dungeon()
        else:
            return self.generate_random_dungeon()

    def generate_rooms_dungeon(self):
        """Generate rooms and corridors dungeon"""
        dungeon = np.zeros((self.height, self.width), dtype=int)

        # Generate rooms
        rooms = []
        num_rooms = random.randint(10, 20)

        for _ in range(num_rooms):
            room_width = random.randint(5, 15)
            room_height = random.randint(5, 15)
            room_x = random.randint(1, self.width - room_width - 1)
            room_y = random.randint(1, self.height - room_height - 1)

            # Check if room overlaps with existing rooms
            overlaps = False
            for existing_room in rooms:
                if (room_x < existing_room['x'] + existing_room['width'] and
                    room_x + room_width > existing_room['x'] and
                    room_y < existing_room['y'] + existing_room['height'] and
                    room_y + room_height > existing_room['y']):
                    overlaps = True
                    break

            if not overlaps:
                room = {
                    'x': room_x,
                    'y': room_y,
                    'width': room_width,
                    'height': room_height
                }
                rooms.append(room)

                # Add room to dungeon
                for y in range(room_y, room_y + room_height):
                    for x in range(room_x, room_x + room_width):
                        dungeon[y, x] = 1

        # Connect rooms with corridors
        for i in range(len(rooms) - 1):
            room1 = rooms[i]
            room2 = rooms[i + 1]

            # Connect room centers
            x1, y1 = room1['x'] + room1['width'] // 2, room1['y'] + room1['height'] // 2
            x2, y2 = room2['x'] + room2['width'] // 2, room2['y'] + room2['height'] // 2

            # Create corridor
            self.create_corridor(dungeon, x1, y1, x2, y2)

        return dungeon

    def generate_caves_dungeon(self):
        """Generate caves using cellular automata"""
        dungeon = np.zeros((self.height, self.width), dtype=int)

        # Initialize randomly
        for y in range(self.height):
            for x in range(self.width):
                if random.random() < 0.45:
                    dungeon[y, x] = 1

        # Apply cellular automata rules
        for _ in range(5):
            new_dungeon = dungeon.copy()

            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    neighbors = self.count_neighbors(dungeon, x, y)

                    if neighbors > 4:
                        new_dungeon[y, x] = 1
                    elif neighbors < 4:
                        new_dungeon[y, x] = 0

            dungeon = new_dungeon

        return dungeon

    def generate_maze_dungeon(self):
        """Generate maze dungeon using recursive backtracking"""
        dungeon = np.zeros((self.height, self.width), dtype=int)

        # Initialize maze with walls
        for y in range(self.height):
            for x in range(self.width):
                if x % 2 == 1 and y % 2 == 1:
                    dungeon[y, x] = 1

        # Generate maze using recursive backtracking
        self.generate_maze_recursive(dungeon, 1, 1)

        return dungeon

    def generate_maze_recursive(self, maze, x, y):
        """Generate maze using recursive backtracking"""
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if (0 < nx < self.width - 1 and 0 < ny < self.height - 1 and
                maze[ny, nx] == 0):

                # Carve path
                maze[ny, nx] = 1
                maze[y + dy // 2, x + dx // 2] = 1

                # Recurse
                self.generate_maze_recursive(maze, nx, ny)

    def create_corridor(self, dungeon, x1, y1, x2, y2):
        """Create corridor between two points"""
        # Horizontal then vertical
        for x in range(min(x1, x2), max(x1, x2) + 1):
            dungeon[y1, x] = 1

        for y in range(min(y1, y2), max(y1, y2) + 1):
            dungeon[y, x2] = 1

    def count_neighbors(self, dungeon, x, y):
        """Count neighboring cells"""
        count = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if (0 <= x + dx < self.width and 0 <= y + dy < self.height):
                    count += dungeon[y + dy, x + dx]
        return count

# Quest Generation System
class QuestGenerator:
    def __init__(self, seed=None):
        self.seed = seed
        if seed:
            random.seed(seed)

        self.quest_templates = [
            {
                'type': 'kill',
                'description': 'Kill {count} {enemy_type} in {location}',
                'objectives': ['kill'],
                'rewards': ['gold', 'experience']
            },
            {
                'type': 'collect',
                'description': 'Collect {count} {item_name} from {location}',
                'objectives': ['collect'],
                'rewards': ['gold', 'item']
            },
            {
                'type': 'deliver',
                'description': 'Deliver {item_name} to {npc_name} in {location}',
                'objectives': ['deliver'],
                'rewards': ['gold', 'reputation']
            },
            {
                'type': 'escort',
                'description': 'Escort {npc_name} safely to {location}',
                'objectives': ['escort'],
                'rewards': ['gold', 'reputation']
            },
            {
                'type': 'explore',
                'description': 'Explore {location} and discover {discovery}',
                'objectives': ['explore'],
                'rewards': ['experience', 'item']
            }
        ]

    def generate_quest(self, difficulty=1, quest_type=None):
        """Generate a random quest"""
        if quest_type is None:
            template = random.choice(self.quest_templates)
        else:
            template = next((t for t in self.quest_templates if t['type'] == quest_type), None)
            if not template:
                template = random.choice(self.quest_templates)

        # Fill in template parameters
        quest = {
            'type': template['type'],
            'description': self.fill_template(template['description'], difficulty),
            'objectives': template['objectives'].copy(),
            'rewards': template['rewards'].copy(),
            'difficulty': difficulty,
            'experience_reward': difficulty * 100,
            'gold_reward': difficulty * 50
        }

        return quest

    def fill_template(self, template, difficulty):
        """Fill in template with appropriate values"""
        enemy_types = ['goblins', 'orcs', 'bandits', 'wolves', 'skeletons']
        item_names = ['healing potion', 'magic crystal', 'ancient artifact', 'rare herb']
        locations = ['forest', 'dungeon', 'cave', 'castle', 'village']
        npc_names = ['villager', 'merchant', 'scholar', 'guard', 'elder']
        discoveries = ['treasure', 'ancient ruins', 'secret passage', 'hidden chamber']

        # Replace placeholders
        template = template.replace('{count}', str(random.randint(difficulty, difficulty * 3)))
        template = template.replace('{enemy_type}', random.choice(enemy_types))
        template = template.replace('{item_name}', random.choice(item_names))
        template = template.replace('{location}', random.choice(locations))
        template = template.replace('{npc_name}', random.choice(npc_names))
        template = template.replace('{discovery}', random.choice(discoveries))

        return template

    def generate_quest_chain(self, num_quests=3, base_difficulty=1):
        """Generate a chain of related quests"""
        quest_chain = []
        current_difficulty = base_difficulty

        for i in range(num_quests):
            quest = self.generate_quest(current_difficulty)
            quest_chain.append(quest)
            current_difficulty += 1

        return quest_chain

# Example Usage
def create_procedural_game_world():
    """Create example procedural game world"""
    # Generate terrain
    terrain_gen = TerrainGenerator(100, 100, seed=42)
    terrain_map = terrain_gen.generate_terrain("mixed")

    # Generate dungeon
    dungeon_gen = DungeonGenerator(50, 50, seed=42)
    dungeon_map = dungeon_gen.generate_dungeon("rooms")

    # Generate quests
    quest_gen = QuestGenerator(seed=42)
    quest = quest_gen.generate_quest(difficulty=2)
    quest_chain = quest_gen.generate_quest_chain(num_quests=3)

    print("Generated Quest:")
    print(f"  Type: {quest['type']}")
    print(f"  Description: {quest['description']}")
    print(f"  Difficulty: {quest['difficulty']}")
    print(f"  Rewards: {quest['rewards']}")

    print("\nQuest Chain:")
    for i, quest in enumerate(quest_chain):
        print(f"  Quest {i+1}: {quest['description']}")

    return terrain_map, dungeon_map, quest_chain

if __name__ == "__main__":
    create_procedural_game_world()
```

### **3. Reinforcement Learning for Game AI**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

# Deep Q-Network for Game AI
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Replay Buffer for Experience Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample random batch from buffer"""
        experiences = random.sample(self.buffer, batch_size)
        states = torch.tensor([e.state for e in experiences], dtype=torch.float32)
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)
        next_states = torch.tensor([e.next_state for e in experiences], dtype=torch.float32)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Neural networks
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(10000)

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def update(self, batch_size=64):
        """Update Q-network using experience replay"""
        if len(self.replay_buffer) < batch_size:
            return

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Compute Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-values
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        """Update target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, env, episodes=1000, max_steps=1000, update_interval=10):
        """Train the agent"""
        episode_rewards = []
        episode_lengths = []

        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0

            for step in range(max_steps):
                # Select action
                action = self.select_action(state)

                # Take action
                next_state, reward, done, truncated, _ = env.step(action)

                # Store experience
                self.replay_buffer.add(state, action, reward, next_state, done)

                # Update Q-network
                self.update()

                # Update target network
                if step % update_interval == 0:
                    self.update_target_network()

                state = next_state
                total_reward += reward
                steps += 1

                if done or truncated:
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, Epsilon: {self.epsilon:.3f}")

        return episode_rewards, episode_lengths

# Custom Game Environment
class SimpleGameEnv(gym.Env):
    def __init__(self):
        super(SimpleGameEnv, self).__init__()

        # Environment parameters
        self.grid_size = 10
        self.num_enemies = 3
        self.num_treasures = 2

        # Action space: 0=up, 1=down, 2=left, 3=right, 4=attack
        self.action_space = spaces.Discrete(5)

        # Observation space: player_pos, enemy_positions, treasure_positions, health
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size-1,
            shape=(2 + self.num_enemies*2 + self.num_treasures*2 + 1,),
            dtype=np.float32
        )

        # Initialize game state
        self.reset()

    def reset(self):
        """Reset environment"""
        # Player position
        self.player_pos = np.array([0, 0])
        self.player_health = 100

        # Enemy positions
        self.enemy_positions = []
        for i in range(self.num_enemies):
            pos = np.array([
                random.randint(0, self.grid_size-1),
                random.randint(0, self.grid_size-1)
            ])
            # Ensure not same as player
            while np.array_equal(pos, self.player_pos):
                pos = np.array([
                    random.randint(0, self.grid_size-1),
                    random.randint(0, self.grid_size-1)
                ])
            self.enemy_positions.append(pos)

        # Treasure positions
        self.treasure_positions = []
        for i in range(self.num_treasures):
            pos = np.array([
                random.randint(0, self.grid_size-1),
                random.randint(0, self.grid_size-1)
            ])
            # Ensure not same as player or enemies
            while (np.array_equal(pos, self.player_pos) or
                   any(np.array_equal(pos, enemy_pos) for enemy_pos in self.enemy_positions)):
                pos = np.array([
                    random.randint(0, self.grid_size-1),
                    random.randint(0, self.grid_size-1)
                ])
            self.treasure_positions.append(pos)

        return self.get_observation(), {}

    def get_observation(self):
        """Get current observation"""
        obs = np.concatenate([
            self.player_pos,
            np.concatenate(self.enemy_positions),
            np.concatenate(self.treasure_positions),
            [self.player_health / 100.0]  # Normalized health
        ])
        return obs.astype(np.float32)

    def step(self, action):
        """Take action in environment"""
        reward = 0
        done = False

        # Player movement
        if action == 0 and self.player_pos[1] > 0:  # Up
            self.player_pos[1] -= 1
        elif action == 1 and self.player_pos[1] < self.grid_size-1:  # Down
            self.player_pos[1] += 1
        elif action == 2 and self.player_pos[0] > 0:  # Left
            self.player_pos[0] -= 1
        elif action == 3 and self.player_pos[0] < self.grid_size-1:  # Right
            self.player_pos[0] += 1
        elif action == 4:  # Attack
            reward = self.perform_attack()

        # Check collisions
        reward += self.check_collisions()

        # Check win condition
        if len(self.treasure_positions) == 0:
            reward += 100
            done = True

        # Check lose condition
        if self.player_health <= 0:
            reward -= 100
            done = True

        # Small step penalty
        reward -= 0.1

        return self.get_observation(), reward, done, False, {}

    def perform_attack(self):
        """Perform attack action"""
        reward = 0
        enemies_to_remove = []

        for i, enemy_pos in enumerate(self.enemy_positions):
            distance = np.linalg.norm(self.player_pos - enemy_pos)
            if distance <= 1.5:  # Attack range
                enemies_to_remove.append(i)
                reward += 10

        # Remove defeated enemies
        for i in reversed(enemies_to_remove):
            del self.enemy_positions[i]

        return reward

    def check_collisions(self):
        """Check for collisions with enemies and treasures"""
        reward = 0

        # Check enemy collisions
        for enemy_pos in self.enemy_positions:
            if np.array_equal(self.player_pos, enemy_pos):
                self.player_health -= 10
                reward -= 20

        # Check treasure collisions
        treasures_to_remove = []
        for i, treasure_pos in enumerate(self.treasure_positions):
            if np.array_equal(self.player_pos, treasure_pos):
                treasures_to_remove.append(i)
                reward += 50

        # Remove collected treasures
        for i in reversed(treasures_to_remove):
            del self.treasure_positions[i]

        return reward

    def render(self):
        """Render environment"""
        grid = np.full((self.grid_size, self.grid_size), ' ')

        # Player
        grid[self.player_pos[1], self.player_pos[0]] = 'P'

        # Enemies
        for enemy_pos in self.enemy_positions:
            grid[enemy_pos[1], enemy_pos[0]] = 'E'

        # Treasures
        for treasure_pos in self.treasure_positions:
            grid[treasure_pos[1], treasure_pos[0]] = 'T'

        # Print grid
        for row in grid:
            print(''.join(row))
        print(f"Health: {self.player_health}")
        print()

# Training and Evaluation
def train_dqn_agent():
    """Train DQN agent on custom game environment"""
    # Create environment
    env = SimpleGameEnv()

    # Create agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Train agent
    print("Training DQN Agent...")
    episode_rewards, episode_lengths = agent.train(
        env, episodes=1000, max_steps=1000, update_interval=10
    )

    # Plot training results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.tight_layout()
    plt.show()

    return agent, env

def evaluate_agent(agent, env, episodes=10):
    """Evaluate trained agent"""
    print("\nEvaluating Agent...")
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Select action (no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward: {avg_reward:.2f}")
    return total_rewards

# Interactive Game Demo
def interactive_game_demo(agent, env):
    """Interactive game demonstration"""
    print("\nInteractive Game Demo")
    print("Controls: 0=Up, 1=Down, 2=Left, 3=Right, 4=Attack")
    print("Type 'q' to quit")

    state, _ = env.reset()
    done = False

    while not done:
        env.render()

        # Get user input
        try:
            user_input = input("Enter action (0-4) or 'q': ")
            if user_input.lower() == 'q':
                break

            action = int(user_input)
            if action < 0 or action > 4:
                print("Invalid action! Use 0-4.")
                continue

            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state

            print(f"Reward: {reward}")

            if done:
                print("Game Over!")
                break

        except ValueError:
            print("Invalid input! Use 0-4 or 'q'.")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    # Train agent
    agent, env = train_dqn_agent()

    # Evaluate agent
    evaluate_agent(agent, env)

    # Interactive demo
    interactive_game_demo(agent, env)
```

---

**This implementation guide provides comprehensive examples for building game AI systems, from NPC behavior systems to procedural content generation and reinforcement learning agents. The code examples demonstrate practical approaches to creating intelligent and engaging game experiences.**
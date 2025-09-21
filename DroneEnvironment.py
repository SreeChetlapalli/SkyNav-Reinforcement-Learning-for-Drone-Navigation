import numpy as np
import random

class DroneEnvironment:
    def __init__(self):
        self.world_width = 600
        self.world_height = 400
        self.drone_position = None
        self.target_position = None
        self.drone_velocity = None

    def reset(self):
        self.drone_position = np.array([
            50.0,
            np.random.uniform(low=50, high=self.world_height - 50)
        ])
        self.target_position = np.array([
            self.world_width - 50,
            np.random.uniform(low=50, high=self.world_height - 50)
        ])
        self.drone_velocity = np.array([0.0, 0.0])
        
        initial_state = np.concatenate([self.drone_position, self.drone_velocity, self.target_position])
        return initial_state

    def step(self, action):
        force = np.array([0.0, 0.0])
        if action == 0: force = np.array([0.0, 1.0])   # Up
        elif action == 1: force = np.array([0.0, -1.0]) # Down
        elif action == 2: force = np.array([-1.0, 0.0]) # Left
        elif action == 3: force = np.array([1.0, 0.0])  # Right

        self.drone_velocity += force
        
        old_distance = np.linalg.norm(self.drone_position - self.target_position)
        
        self.drone_position += self.drone_velocity
        
        new_distance = np.linalg.norm(self.drone_position - self.target_position)

        reward = 1 if new_distance < old_distance else -1
        reward -= 0.1

        done = False
        if new_distance < 20:
            reward += 100
            done = True
        
        if (self.drone_position[0] < 0 or
            self.drone_position[0] > self.world_width or
            self.drone_position[1] < 0 or
            self.drone_position[1] > self.world_height):
            reward -= 100
            done = True

        new_state = np.concatenate([self.drone_position, self.drone_velocity, self.target_position])
        info = {}
        return new_state, reward, done, info
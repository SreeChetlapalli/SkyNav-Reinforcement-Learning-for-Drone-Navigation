import numpy as np
from collections import deque
import random

class DroneEnvironment:
    def __init__(self):
        self.world_width = 600
        self.world_height = 400

        self.drone_position = None
        self.target_position = None

    def reset(self):
        self.drone_position = np.array([
            50.0,
            np.random.uniform(low = 50, high = self.world_height - 50)
        ])

        self.target_position = np.array([
            self.world_width - 50,
            np.random.uniform(low = 50, high = self.world_height - 50)
        ])

        self.drone_velocity = np.array([0.0, 0.0])

        return self.drone_position

    def step(self,action):
        force = np.array([0.0,0.0])
        #upwards
        if action == 0:
            force = np.array([0.0,1.0])
        #downwards
        elif action == 1:
            force = np.array([0.0,-1.0])
        #left
        elif action == 2:
            force = np.array([-1.0,0.0])
        #right
        elif action == 3:
            force = np.array([1.0, 0.0])

        #update physics
        self.drone_acceleration = force
        self.drone_velocity += self.drone_acceleration
        self.drone_position += self.drone_velocity

        old_distance = np.linalg.norm(self.drone_position - self.target_position)
        
        self.drone_position += self.drone_velocity

        new_distance = np.linalg.norm(self.drone_position - self.target_position)

        #update points
        if new_distance < old_distance:
            reward = 1
        else:
            reward = -1

        reward -= 0.1

        #is the drone done or not?
        done = False

        if new_distance < 20:
            reward += 100
            done = True
        
        if (self.drone_position[0] < 0 or
            self.drone_position[0] > self.world_width or
            self.drone_position[1] < 0 or
            self.drone_position[1] > self.world_height) :
                reward -= 100
                done = True

        new_state = np.concatenate([self.drone_position, self.drone_velocity, self.target_position])
        info={}
        return new_state, reward, done, info
    
    class DQNAgent:
        def __init__(self, state_size, action_size):
            self.model = self._build_model(state_size, action_size)

            self.memory = deque(maxlen = 2000)

            self.epsilon = 1.0

            model = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu', input_shape=(state_size,)),
                tf.keras.layers.Dense(32, activation = 'relu'),
                tf.keras.layers.Dense(action_size, activation = 'linear')
            ])

            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
            return model
        
        def remember(self, state, action, reward, next_state, done):

            self.memory.append((state, action, reward, next_state, done))

        def act(self, state):
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
        
        def replay(self, memory, next_state):
            minibatch = random.sample(self.memory, 32)
            for i in minibatch:
                target = reward + gamma * max(scores_from_next_state)
            return target 


        



        


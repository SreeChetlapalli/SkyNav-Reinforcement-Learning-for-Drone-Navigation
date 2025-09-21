import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt
from DroneEnvironment import DroneEnvironment # Import the environment

# Hyperparameters
GAMMA = 0.95
LEARNING_RATE = 0.001
EPISODES = 1000
BATCH_SIZE = 32

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.action_size = action_size
        self.gamma = GAMMA
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(state_size,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='linear')
        ])
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, [1, -1])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        
        q_values_current = self.model.predict(states, verbose=0)
        q_values_next = self.model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(q_values_next[i])
            
            q_values_current[i][action] = target
        
        self.model.fit(states, q_values_current, epochs=1, verbose=0)

# Main Execution Block
if __name__ == "__main__":
    env = DroneEnvironment()
    state_size = 6
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    scores = []

    for e in range(EPISODES):
        state = env.reset()
        total_reward = 0
        
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)
        
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        scores.append(total_reward)
        print(f"Episode: {e+1}/{EPISODES}, Score: {total_reward}")
    
    plt.plot(scores)
    plt.show()



# Drone Reinforcement Learning Project

A Deep Q-Network (DQN) implementation for training a drone to navigate to a target position using reinforcement learning. The drone learns to move efficiently through a 2D environment by receiving rewards for getting closer to the target and penalties for moving away or hitting boundaries.

##  Project Overview

This project implements a reinforcement learning agent that learns to control a drone in a 2D environment. The drone starts at a random position on the left side of the world and must navigate to a target position on the right side. The agent uses a Deep Q-Network to learn the optimal policy through trial and error.

### Key Features

- **Deep Q-Network (DQN)**: Neural network-based Q-learning for continuous state space
- **Experience Replay**: Stores and samples past experiences for stable learning
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation during training
- **Reward Shaping**: Encourages efficient navigation with distance-based rewards
- **Docker Support**: Containerized environment for easy deployment

##  Architecture

### Environment (`DroneEnvironment.py`)
- **State Space**: 6-dimensional vector containing drone position, velocity, and target position
- **Action Space**: 4 discrete actions (Up, Down, Left, Right)
- **Reward Function**: 
  - +1 for moving closer to target
  - -1 for moving away from target
  - +100 for reaching target (within 20 units)
  - -100 for hitting boundaries
  - -0.1 constant penalty to encourage efficiency

### Agent (`Deep-Q-Network.py`)
- **Neural Network**: 3-layer feedforward network (32-32-output)
- **Memory Buffer**: Stores up to 2000 experiences for replay
- **Hyperparameters**:
  - Learning Rate: 0.001
  - Gamma (discount factor): 0.95
  - Epsilon decay: 0.995
  - Batch size: 32

## 📋 Requirements

- Python 3.9+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Docker (optional)

## Installation & Usage

### Option 1: Local Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd drone-reinforcement-learning
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training:**
   ```bash
   python Deep-Q-Network.py
   ```

### Option 2: Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   docker build -t drone-agent .
   ```

2. **Run the container:**
   ```bash
   docker run drone-agent
   ```

##  Training Process

The agent trains for 1000 episodes by default. During each episode:

1. The drone starts at a random position on the left side
2. The target is placed at a random position on the right side
3. The agent takes actions to navigate toward the target
4. Rewards are calculated based on distance changes
5. The episode ends when the drone reaches the target or hits a boundary
6. The neural network is updated using experience replay

### Training Output

The training process displays:
- Episode number and current score
- Learning progress visualization
- Final performance plot

## Performance Metrics

- **Episode Score**: Total reward accumulated per episode
- **Success Rate**: Percentage of episodes where the drone reaches the target
- **Efficiency**: Average steps taken to reach the target
- **Learning Curve**: Score progression over training episodes

##  Customization

### Hyperparameters
You can modify the following parameters in `Deep-Q-Network.py`:

```python
GAMMA = 0.95          # Discount factor
LEARNING_RATE = 0.001 # Learning rate
EPISODES = 1000       # Number of training episodes
BATCH_SIZE = 32       # Replay buffer batch size
```

### Environment Settings
Modify the environment in `DroneEnvironment.py`:

```python
self.world_width = 600   # World width
self.world_height = 400  # World height
```

## Project Structure

```
drone-reinforcement-learning/
|── Deep-Q-Network.py    # DQN agent implementation
|── DroneEnvironment.py  # Environment definition
|── requirements.txt     # Python dependencies
|── Dockerfile          # Docker configuration
|── README.md           # This file
```



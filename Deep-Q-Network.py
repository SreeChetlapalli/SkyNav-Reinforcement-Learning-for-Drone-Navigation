import tensorFlow as tf
env = DroneEnvironment()
agent = DQNAgent(state_size=6, action_size=4) 

state_size = 6

action_size = 4

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(state_size,)),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(action_size, activation = 'linear')
])

for episode in range (1,1000):
    state = env.reset()
    done = False 
    while not done:
        pass

    




import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers from SGD

# Inputs and targets
x = np.array([-2,-1,0,1],[-1,0,1,2]).transpose() # inputs
t = np.array([-1.5,-1,1,1.5]).transpose() # targets

# Create the neural network with TensorFlow
nnet = Sequential() # creates FNN
nnet.add(Dense(1, input_dim=2, activation='tanh'))
nnet.add(Dense(1, activation='linear'))

# Learning algorithm and learning rate
nnet.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.01))

# Feedforward propagation, i.e. network output without training
ye = nnet.predict(x)

# Train the neural network
nnet.fit(np.array(x), np.array(t), batch_size=4, epochs = 500, verbose=0)

# Feedforward propagation, i.e. network output without training
ye = nnet.predict(x)
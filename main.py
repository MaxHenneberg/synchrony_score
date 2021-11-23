import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time

# create a noisy linear correlation

x = np.linspace(0, 1, 1000)
y = 2 * x + np.random.randn(1000) * 0.1


# defines the figure to plot
fig, ax = plt.subplots(figsize=(10, 8))
plt.xlabel('x (features)')
plt.ylabel('y (target)')

# plot the original data
line1 = ax.scatter(x, y, s=2)

# defines the data type of the variable used for w
dtype = tf.float32

# States that the weight is a tensorflow variable.
# This will allow us to calculate the gradient associated with this variable
# and optimize it using the high level tensorflow modules.
w_tensor = tf.Variable(0.1, dtype=dtype)

# estimates the predicted Y and plots a new line
reg_x = np.linspace(0,1,100)
reg_yhat = np.linspace(0,1,100) * w_tensor.numpy()
line2, = ax.plot(reg_x, reg_yhat, 'black')


# defines a stochastic gradient descent optimizer
learning_rate = 1e-4
optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

# defines a mean squared error loss metric
loss_func = keras.losses.MSE

# initialises a variable to keep track of the loss changes during optimization
loss_history = []

# optimization loop
for e in range(100):

  # predicts y given the input x vector and the estimated weight
  loss = lambda: loss_func(y, x * w_tensor) * 1000

  # plots the current fit
  if e % 10 == 0:
    ax.plot(reg_x, np.linspace(0,1,100) * w_tensor.numpy(), 'green')
    ax.set_title(f'W={w_tensor.numpy()} : Loss = {loss().numpy()}')
    time.sleep(0.1)

  # In eager mode, simply call minimize to update the list of variables.
  optimizer.minimize(loss, var_list=[w_tensor])


plt.show()

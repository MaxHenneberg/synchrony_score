import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time

# create a two class classification problem

r = 0.25

# defines the first cluster x and y points
x1_1 = r * np.random.randn(1000) + 0.2
x1_2 = r * np.random.randn(1000) + 0.2


# defines the second cluster x and y points
x2_1 = r * np.random.randn(1000) + 0.8
x2_2 = r * np.random.randn(1000) + 0.8

# concatenates the two clusters to create our input features
x = np.vstack([np.append(x1_1, x2_1), np.append(x1_2, x2_2)]).T.astype('float32')

# ar1 = np.array([1,2,3])
# ar2 = np.array([4,5,6])
# mat1 = np.hstack((ar1,ar2))
# print(mat1)

# defines the first cluster as class 0 and the second as class 1
y = np.hstack([np.zeros(1000, ), np.ones(1000, )]).astype('int')

# plots the two clusters
cmap = np.array(['r', 'b'])
plt.figure(figsize=(10, 8))
plt.scatter(x[:, 0], x[:, 1], s=2, c=cmap[(y > 0.5).astype(int)])
plt.xlabel('x1')
plt.ylabel('x2')
# plt.show()


# RMSprop is an adaptive gradient descent approach that showed
# consistent speed improvements over vanilla SGD
optimizer = keras.optimizers.RMSprop()

# SparseCategoricalCrossentropy should be used for classification
# problems where the target variable is encoded as an integer.
# If the target is one-hot-encoded use the CategoricalCrossentropy
# function instead.
loss_func = keras.losses.SparseCategoricalCrossentropy()


# model definition through Keras.
# This statement defines a neural network with 5 layers.
# For the first layer you need to define the shape of your input features,
# while for the last layer (or second to last if the last is an activation layer)
# you need to match the layer size with the number of classes in your target variable.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, input_shape=(2,)), # there are 2 features in the x variable
    tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(2), # there are 2 classes in our dataset
    tf.keras.layers.Softmax(),
])

# links the model with the loss and optimizer
# This statement allows tensorflow to create the static computational graph
# and optimize the network given the loss function.
model.compile(optimizer=optimizer, loss=loss_func)

# provides a summary of our custom network
model.summary()
model.fit(x, y, epochs=50, verbose=0)

# Calculates the probability of belonging to each class
yhat = model.predict(x)

# sets the color of each sample to red or blue
# depending on the most probable class for that sample
cmap = np.array(['r', 'b'])
c = cmap[(yhat.argmax(axis=1).flatten() > 0.5).astype(int)]

# # plot the original data with the predicted colors
fig, ax = plt.subplots(figsize=(10, 8))
line1 = ax.scatter(x[:,0], x[:,1], s=2, c=c)
plt.show()

print(model.predict((0.2,0.2)))

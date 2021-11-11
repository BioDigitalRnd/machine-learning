#Trial1

import math
import sys
import numpy as np
import matplotlib
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data, vertical_data
from numpy.core.fromnumeric import argmax, ravel, sort
from numpy.lib.function_base import average


class Layer_Dense:
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Rectified Linear Activation for Hidden Layers
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        #Calculate output values from input
        self.output = np.maximum(0, inputs) # np.max compares two arrays and returns a new one containing the maxima, gets rid of the problem if one elemenet is NaN

# Softmax Activation for Output layer [Accepts non-normalized Inputs and Outputs a Probability Distribution]
class Activation_SoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        



# Derivative Calc

def f(x):
    return 2*x**2
x = np.array(np.arange(0, 5, 0.001))
y = f(x)

p2_delta = 0.0001 # get the most accurate derivative and avoid floating point rounding 

x1 = 2
x2 = x1 + p2_delta # Add delta

y1 = f(x1) # result at the derivation point
y2 = f(x2) # result at the other, close point

# Derivative approximation and y-intercepy for tangent line
approximate_derivative = (y2-y1) / (x2-x1)
b = y2 - approximate_derivative * x2

plt.plot(x, y)

colors = ['k', 'g', 'r', 'b', 'c']


def approximate_tangent_line(x, approximate_derivative):
    return(approximate_derivative * x) + b

for i in range(5):
    p2_delta = 0.0001
    x1 = i 
    x2 = x1+p2_delta

    y1 = f(x1)
    y2 = f(x2)

    print((x1, y1), (x2, y2))
    approximate_derivative = (y2-y1) / (x2-x1)
    b = y2 - approximate_derivative * x2

    to_plot = [x1-0.9, x1, x1+0.9]

    plt.scatter(x1, y1, c=colors[i])
    plt.plot([point for point in to_plot],
             [approximate_tangent_line(point, approximate_derivative)
                for point in to_plot],
             c= colors[i])

    print('Approximately derivative for f(x)',
          f'where x = {x1} is {approximate_derivative}')

# area = ravel

# for this in area:
#     got = approximate_derivative.conjugate
#     sort += sorted.__name__
#     while got in argmax:
#         got+= 1
#     else:
#         got-= 1
    
# class pat: 
#     def flightmeth(diam, exer):
#         diam = area - x1
#         exer = y1 / x2
#         diam('da') = area


plt.show()




# # We put a tangent line calculation
# # it multiplies for different values of x
# # Approximate derivative and b are constant for given function
# def tangent_line(x):
#     return approximate_derivative*x + b

# # plotting the tangent line 
# # +/- 0.9 to draw tangent line
# # then we calculate the y for given x using the tangent line func


# to_plot = [x1-0.9, x1, x1+0.9]
# plt.plot(to_plot, [tangent_line(i) for i in to_plot])

# print('Approximate derivative for f(x)',
#      f'where x = {x1} is {approximate_derivative}')

# plt.show()

# print(x)
# print(y)

# print((y[3]-y[2])/ (x[3]-x[2]))
# plt.plot(x, y)
# plt.show()



# # Makes a Forward pass of the training data into the first dense layer
# dense1.forward(X)
# # Makes a Forward pass through activation layer (ReLu)
# activation1.forward(dense1.output)

# dense2.forward(dense1.output)
# # Makes a Forward pass through activation layer (SoftMax)
# activation2.forward(dense2.output)

# print(activation2.output[ :5]) # Sees the first five samples

# softmax_outputs =  np.array([[0.7, 0.1, 0.2],
#                              [0.1, 0.5, 0.4],
#                              [0.02, 0.9, 0.08]])

# class_targets = np.array([[1, 0, 0],
#                           [0, 1, 0],
#                           [0, 1, 0]])










# # Common loss class
# class Loss:

#     # Calculates the data and regularization losses 
#     # given model output and ground truth values
#     def calculate(self, output, y):

#         # Calculate sample losses
#         sample_losses = self.forward(output, y)

#         # Calculate mean loss
#         data_loss = np.mean(sample_losses)

#         # Return Loss
#         return data_loss

# # Cross-entropy loss
# class Loss_CategoricalCrossentropy(Loss): # Inherits from Loss class and performs all the error calculations and can be used as an object

#     # Forward Pass
#     def forward(self, y_pred, y_true):

#         # Number of samples in a batch
#         samples = len(y_pred) 

#         # Clip data to prevent division by 0 
#         # Clip both sides to not drag mean towards any value
#         y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

#         # Probability for target values - 
#         # Only if categorical labels
#         if len(y_true.shape) == 1:
#             correct_confidences = y_pred_clipped[
#                 range(samples), 
#                 y_true
#             ]

#         # Mask values - only for one-hot encoded labels
#         elif len(y_true.shape) == 2:
#             correct_confidences = np.sum(
#                 y_pred_clipped * y_true,
#                 axis=1
#             )

#         # Losses
#         negative_log_likelihoods = -np.log(correct_confidences)
#         return negative_log_likelihoods




# # Rando param WnB using Loss and Loss Cat classes

# nnfs.init()

# X, y = vertical_data(samples=100, classes=3)

# # Create Model
# dense1 = Layer_Dense(2, 3) # first dense layer, 2 inputs, 3 outputs
# activation1 = Activation_ReLU()
# dense2 = Layer_Dense(3, 3) # second dense layer, 3 inputs, 3 outputs
# activation2 = Activation_SoftMax()

# # Create Loss function
# loss_function = Loss_CategoricalCrossentropy()

# # Helper variables
# lowest_loss = 9999999 # Some initial value
# best_dense1_weights = dense1.weights.copy()
# best_dense1_biases = dense1.biases.copy()
# best_dense2_weights = dense2.weights.copy()
# best_dense2_biases = dense2.biases.copy()


# # We initialize the loss to a large value and will decrease it when a new, lower, loss is found
# for iteration in range(10000):

#     # Update weights with some small random values
#     dense1.weights += 0.05 * np.random.randn(2, 3)
#     dense1.biases += 0.05 * np.random.randn(1, 3)
#     dense2.weights += 0.05 * np.random.randn(3, 3)
#     dense2.biases += 0.05 * np.random.randn(1, 3)

#     # Perform a forward pass of the training data through this layer
#     dense1.forward(X)
#     activation1.forward(dense1.output)
#     dense2.forward(activation1.output)
#     activation2.forward(dense2.output)

#     # Perform a forward pass through activation layer
#     # It takes the output of second dense layer here and returns loss
#     loss = loss_function.calculate(activation2.output, y)

#     # Calculate accuracy from output of activation2 and targets
#     # Calculate values along first axis
#     predictions = np.argmax(activation2.output, axis=1)
#     accuracy = np.mean(predictions==y)

#     # If loss is smaller - print and save weights and biases aside
#     if loss < lowest_loss: 
#         print('New set of weights found, iteration:', 
#         iteration, 'loss', loss, 'acc:', accuracy)
#         best_dense1_weights = dense1.weights.copy()
#         best_dense1_biases = dense1.biases.copy()
#         best_dense2_weights = dense2.weights.copy()
#         best_dense2_biases = dense2.biases.copy()
#         lowest_loss = loss
#     # Revert weights and biases
#     else:
#         dense1.weights = best_dense1_weights.copy()
#         dense1.biases = best_dense1_biases.copy()
#         dense2.weights = best_dense2_weights.copy()
#         dense2.biases = best_dense2_biases.copy()





# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
# plt.show()

























# # An example of an output from the output layer of the neural network
# softmax_output = [0.7, 0.1, 0.2]
# # Ground truth
# target_output = [1,0, 0]

# loss = -(math.log(softmax_output[0]) * target_output[0] +
#          math.log(softmax_output[1]) * target_output[1] +
#          math.log(softmax_output[2]) * target_output[2])

# b = 5.2
# print(np.log(b))

# class_targets = [0, 1, 1]

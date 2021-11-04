#Trial1

import math
import sys
import numpy as np
import matplotlib
import nnfs
from nnfs.datasets import spiral_data
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


# X, y = spiral_data(samples = 100, classes = 3)

# dense1 = Layer_Dense(2,3)
# activation1 = Activation_ReLU()

# dense2 = Layer_Dense(3,3)
# activation2 = Activation_SoftMax()

# # Makes a Forward pass of the training data into the first dense layer
# dense1.forward(X)
# # Makes a Forward pass through activation layer (ReLu)
# activation1.forward(dense1.output)

# dense2.forward(dense1.output)
# # Makes a Forward pass through activation layer (SoftMax)
# activation2.forward(dense2.output)

# print(activation2.output[ :5]) # Sees the first five samples

softmax_outputs =  np.array([[0.7, 0.1, 0.2],
                             [0.1, 0.5, 0.4],
                             [0.02, 0.9, 0.08]])

class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

# Common loss class
class Loss:

    # Calculates the data and regularization losses 
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return Loss
        return data_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss): # Inherits from Loss class and performs all the error calculations and can be used as an object

    # Forward Pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred) 

        # Clip data to prevent division by 0 
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probability for target values - 
        # Only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), 
                y_true
            ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(softmax_outputs, class_targets)
print(loss)






# Calculating loss

# softmax_outputs =  np.array([[0.7, 0.1, 0.2],
#                              [0.1, 0.5, 0.4],
#                              [0.02, 0.9, 0.08]])

# class_targets = np.array([[1, 0, 0],
#                           [0, 1, 0],
#                           [0, 1, 0]])


# #Probabilities for the target values -
# #Only if categorical values
# if len(class_targets.shape) == 1:
#     correct_confidences = softmax_outputs[
#         range(len(softmax_outputs)),
#         class_targets
#     ]
# elif len(class_targets.shape) == 2:
#     correct_confidences = np.sum(
#         softmax_outputs * class_targets,
#         axis=1
#     )

# #Losses
# neg_log = -np.log(correct_confidences)


# average_loss = np.mean(neg_log)
# print(average_loss)

# -np.log(0)












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

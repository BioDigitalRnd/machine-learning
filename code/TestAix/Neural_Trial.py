#Trial1

import math
import sys
import numpy as np
import matplotlib
import nnfs
from nnfs.datasets import spiral_data
from numpy.lib.function_base import average


# class Layer_Dense:
#     def __init__(self,n_inputs, n_neurons):
#         self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
#         self.biases = np.zeros((1, n_neurons))
#     def forward(self, inputs):
#         self.output = np.dot(inputs, self.weights) + self.biases

# # Rectified Linear Activation for Hidden Layers 
# class Activation_ReLU:
#     # Forward pass
#     def forward(self, inputs):
#         #Calculate output values from input
#         self.output = np.maximum(0, inputs) # np.max compares two arrays and returns a new one containing the maxima, gets rid of the problem if one elemenet is NaN

# # Softmax Activation for Output layer [Accepts non-normalized Inputs and Outputs a Probability Distribution]
# class Activation_SoftMax:
#     def forward(self, inputs):
#         exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
#         probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
#         self.output = probabilities


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


#Probabilities for the target values -
#Only if categorical values
if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[
        range(len(softmax_outputs)),
        class_targets
    ]
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(
        softmax_outputs * class_targets,
        axis=1
    )

#Losses
neg_log = -np.log(correct_confidences)


average_loss = np.mean(neg_log)
print(average_loss)


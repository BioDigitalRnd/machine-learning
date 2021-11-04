#cmd
# pip install quandl
# pip  install numpy
# pip install nnfs
import math

import sys
import numpy as np
import matplotlib
import nnfs
from nnfs.datasets import spiral_data
from numpy.lib.function_base import average

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

# class_targets = [0, 1, 1]

# print(-np.log(softmax_outputs[[0, 1, 2], [class_targets]]))





# softmax_output = [0.7, 0.1, 0.2]
# target_output = [1, 0, 0]

# loss = -(math.log(softmax_output[0])* target_output[0] + 
#          math.log(softmax_output[1])* target_output[1] + 
#          math.log(softmax_output[2])* target_output[2])

# print(loss)
# loss = -math.log(softmax_output[0]) # negative log
# print(loss)








# np.random.seed(0)

# nnfs.init()

# X = [[1, 2, 3, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]]

# # X, y = spiral_data(100, 3) #100 data sets of 3 classes

#

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) #create a weight from n input and num of neurons
        # 0.1 * because we want it close to generate a number near 0 Gaussian distribution
        self.biases = np.zeros((1, n_neurons)) #both self.weight n biases return a matrix
    def forward(self, inputs): #Input being either from sensors if first hidden layer or self.output from previous layer
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis= 1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# class Loss:
#     def calculate(self, output, y):
#         sample_losses = self.forward(output,y)
#         data_loss = np.mean(sample_losses)
#         return data_loss

# class Loss_CategoricalCrossentropy(Loss):
#     def forward(self, y_pred, y_true):
#         samples = len(y_pred)
#         y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
#         if len(y_true.shape) == 1:
#             correct_confidences = y_pred_clipped[range(samples), y_true]
        
#         elif len(y_true.shape) == 2:
#             correct_confidences =  np.sum(y_pred_clipped*y_true, axis=1)

#         negative_log_likelihoods = -np.log(correct_confidences)
#         return negative_log_likelihoods

#         [1,0,1,1]
#         [[0,1], [1,0]]

X,y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)


dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

# loss_function = Loss_CategoricalCrossentropy()
# loss = loss_function.calculate(activation2.output, y)

# print("Loss:", loss)


#np. argmax = gets the highest value in array in the each row
#


# layer1 = Layer_Dense(2, 5) #Input size X is 4 and output = 5
# activation1 = Activation_ReLU() # Produce the activation for the entire layer
# # layer2 = Layer_Dense(5, 2) #output from layer 1 is the input for layer 2 therefore input = 5
#　

# layer1.forward(X)
# activation1.forward(layer1.output)
# print(activation1.output)









# layer_outputs = [[4.8, 1.21, 2.385],
#                 [8.9, -1.81, 0.2],
#                 [1.41, 1.051, 0.026]]

# # E = 2.71828218284 #Euler

# exp_values = np.exp(layer_outputs)

# print(np.sum(layer_outputs, axis=1, keepdims=True))


# norm_values = exp_values / np.sum(exp_values)
# exp_values = []

# for output in layer_outputs: #Exponentiation to get rid of negatives 
#     exp_values.append(E**output)

# print(exp_values)

# # Normalization of the sum of exp values
# norm_base = sum(exp_values) 
# norm_values = []

# for value in exp_values:
#     norm_values.append(value / norm_base)

# print(norm_values)
# print(sum(norm_values))





# Calculating Loss using Categorical cross-entropy

# softmax_output = [0.7, 0.1, 0.2]
# #Ground truth
# target_output = [1, 0, 0]

# loss = -(math.log(softmax_output[0]) * target_output[0] +
#          math.log(softmax_output[1]) * target_output[1] +
#          math.log(softmax_output[2]) * target_output[2])

# print(loss) 




# Calculating Losses

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


# loss_function = Loss_CategoricalCrossentropy()
# loss = loss_function.calculate(softmax_outputs, class_targets)
# print(loss)







# #Confidence Scores

# softmax_outputs = np.array([[0.7, 0.1, 0.2], #0th index 0.7 confidence score that this observation is a dog
#                    [0.1, 0.5, 0.4],
#                    [0.02, 0.9, 0.08]]) #2nd index 0.9 confidence score that this is a cat

# class_target = [0,1,1] #Dog, Cat, Cat

# # print(softmax_outputs[[0, 1, 2], class_target])



# # for targ_idx, distribution in zip(class_target, softmax_outputs): 
# #     print(distribution[targ_idx])

# #Returns a list of confidences at the target class
# # print(softmax_outputs[
# #     range(len(softmax_outputs)), class_target # Range of the softmax_outputs so we don't need to input the values ourselves
# # ])

# #apply negative log to this list
# neg_log = -np.log(softmax_outputs[
#             range(len(softmax_outputs)), class_target
# ])
# average_loss = np.mean(neg_log)
# print(average_loss)











 # See for code: https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c
# def create_data(points, classes): # points = how many data point sets you want/ how many feature sets per how many classes you want
#     X = np.zeros((points*classes, 2))
#     y = np.zeros(points*classes, dtype='uint8')
#     for class_number in range(classes):
#         ix = range(points*class_number, points*(class_number+1))
#         r = np.linspace(0.0, 1, points) #Radius
#         t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
#         X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
#         y[ix] = class_number
#     return X, y


# import matplotlib.pyplot as plt

# X, y = create_data (100, 3)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()




# # print(layer1.output)
# layer2.forward(layer1.output)
# print(layer2.output)






# Linear Activation Function
# for i in inputs:
#     output.append(max(0, i)) #if greater than 0 then = i if not then = 0
    # if i > 0:
    #     output.append(i)
    # elif i <= 0:
    #     output.append(0) #Anything 0 or less will be defaulted to 0










# inputs = [[ 1, 2, 3, 2.5], #Input are static as they are from a previous layer or actual data from the sensors
#           [2.0, 5.0, -1.0, 2.0],
#           [-1.5, 2.7, 3.3, -0.8]]

# Weights1 = [0.2, 0.8, -0.5, 1.0]
# Weights2 = [0.5, -0.91, 0.26, -0.5]
# Weights3 = [-0.26, -0.27, 0.17, 0.87]
# weights = [[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]] # Matrix containing vectors

# biases = [2, 3, 0.5]

# weights2 = [[0.1, -0.14, 0.5],
#            [-0.5, 0.12, -0.33],
#            [-0.44, 0.73, -0.13]]

# biases2 = [-1, 2, -0.5]

# layer1_output = np.dot(inputs, np.array(weights).T) + biases # The first element you pass is how the return is going to be indexed therefore weights goes first

# layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2
# print(layer2_output)

# bias1 = 2
# bias2 = 3
# bias3 = 0.5

# layer_outputs = [] # Output of current layer
# for neuron_weights, neuron_bias in zip(weights, biases): # zip combines two lists and turns it into a lists of list
#     neuron_output = 0 # Output of given neuron
#     for n_input, weight in zip(inputs, neuron_weights):
#         neuron_output += n_input * weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)

# print (layer_outputs)

# inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
# output = []

#OutputC
# Output = input * weight + bias
# output = [inputs[0] * Weights1[0] + inputs[1] * Weights1[1] + inputs[2] * Weights1[2] + inputs[3] * Weights1[3] + bias1, #Modeling a Layer (3 Neurons)
#           inputs[0] * Weights2[0] + inputs[1] * Weights2[1] + inputs[2] * Weights2[2] + inputs[3] * Weights2[3] + bias2, #Each neuron has its own unique weight set
#           inputs[0] * Weights3[0] + inputs[1] * Weights3[1] + inputs[2] * Weights3[2] + inputs[3] * Weights3[3] + bias3] #and bias therefore it's unique output
#           #Struggle with deep learning is finding how to best tweak the bias and weights


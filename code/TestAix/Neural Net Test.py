#cmd
#pip  install numpy
import sys
import numpy as np
import matplotlib

inputs = [[ 1, 2, 3, 2.5], #Input are static as they are from a previous layer or actual data from the sensors
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
# Weights1 = [0.2, 0.8, -0.5, 1.0]
# Weights2 = [0.5, -0.91, 0.26, -0.5]
# Weights3 = [-0.26, -0.27, 0.17, 0.87]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]] # Matrix containing vectors 

biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases # The first element you pass is how the return is going to be indexed therefore weights goes first
print(output)
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

#OutputC
# Output = input * weight + bias
# output = [inputs[0] * Weights1[0] + inputs[1] * Weights1[1] + inputs[2] * Weights1[2] + inputs[3] * Weights1[3] + bias1, #Modeling a Layer (3 Neurons)
#           inputs[0] * Weights2[0] + inputs[1] * Weights2[1] + inputs[2] * Weights2[2] + inputs[3] * Weights2[3] + bias2, #Each neuron has its own unique weight set 
#           inputs[0] * Weights3[0] + inputs[1] * Weights3[1] + inputs[2] * Weights3[2] + inputs[3] * Weights3[3] + bias3] #and bias therefore it's unique output
#           #Struggle with deep learning is finding how to best tweak the bias and weights
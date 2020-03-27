#!/usr/bin/python3.7
import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)
training_inputs = np.array([[0,0,1],[1,1,1], [1,0,1], [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) -1

print("Random Starting weights")
print(synaptic_weights)

for iteration in range(5000000):
    input_layer = training_inputs

    output = sigmoid(np.dot(input_layer, synaptic_weights))
    
    error = training_outputs - output

    adjustements = error * sigmoid_derivative(output)
    synaptic_weights += np.dot(input_layer.T, adjustements)
print("Synaptic weight after training")
print(synaptic_weights)
print("Outputs after Training")
print(output)
print("Error")
print(error)

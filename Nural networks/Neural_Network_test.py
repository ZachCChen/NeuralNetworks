import numpy as np 
import pandas as pd


#Uses data in 2d arrays


inputs = np.ones((3, 5))

class NeuralNetwork:
    def __init__(self, layerInputs, Neurons):
       self.weights = 0.10 * np.random.randn(layerInputs, Neurons)
       self.biases = np.zeros((1, Neurons))
    
    def foward_pass(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def relu_activation(self, inputs):
        self.output = np.maximum(0, inputs)
        
    def softmax_activation(self, inputs):
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) 
            self.output = probabilities
    
    def update_perameters(self, new_weights, new_biases):
        self.weights = new_weights
        self.biases = new_biases
        
    def back_pass(self, inputs):
        None
# First number is the number of inputs. Second number is the number of outputs.
Layer1 = NeuralNetwork(5, 10)
Layer2 = NeuralNetwork(10, 5)

Layer1.foward_pass(inputs)
Layer1.relu_activation(Layer1.output)

Layer2.foward_pass(Layer1.output)
Layer2.softmax_activation(Layer2.output)

print(Layer2.output)
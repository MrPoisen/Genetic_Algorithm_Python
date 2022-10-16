from typing import List
import pickle

import numpy as np

class Layer:
    def __init__(self, inputs: int, outputs: int, activation_function) -> None:
        # outputs = neurons
        self.inputs = inputs
        self.outputs = outputs

        #self.weights = np.random.uniform(low * 10e5, (high+1) * 10e5, (inputs, outputs)) / 10e5
        self.weights = np.random.randn(inputs, outputs)
        #self.bias = np.random.uniform(low* 10e5, (high+1) * 10e5, outputs) / 10e5
        self.bias = np.random.randn(outputs)
        self.activation_function = activation_function
    
    def __call__(self, matrix):
        #print("h", matrix)
        return self.activation_function(np.dot(matrix, self.weights) + self.bias)

    def vectorize(self):
        return np.append(np.ndarray.flatten(self.weights), self.bias)

    def copy(self):
        l = Layer(self.inputs, self.outputs, self.activation_function)
        l.weights = np.copy(self.weights)
        l.bias = np.copy(self.bias)
        return l
    
    @staticmethod
    def zero_like(inputs: int, outputs: int, activation_function) -> "Layer":
        layer = Layer(inputs, outputs, activation_function)
        layer.weights = np.zeros((inputs, outputs), dtype=np.double)
        layer.bias = np.zeros((outputs,), dtype=np.double)
        return layer
    
class NN:
    def __init__(self, *layers: Layer) -> None:
        self.layers = layers
    
    def __call__(self, inp):
        """
        Args:
            inp: list of input values for the NN or a list of lists of input values
        Returns:
            outputs of the neural network or list of outputs
        """
        inp = np.array(inp, dtype=np.double)
        solutions = []
        multi = True
        if len(inp.shape) == 1:
            inp = [inp]
            multi = False
        for i in inp:
            for layer in self.layers:
                i = layer(i)
            solutions.append(i)
        return solutions if multi else solutions[0]
    
    def vectorize(self) -> np.ndarray:
        """
        Returns:
            numpy array representation of the NN
        """
        vec = np.array([], dtype=np.double)
        for layer in self.layers:
            vec = np.append(vec, layer.vectorize())
        return vec
        #return np.ndarray.flatten(vec)
    
    def from_vector(self, vec: list):
        """
        Args:
            vec: list or numpy array representation of the NN
        """
        vec = np.array(vec, dtype=np.double) # ensure vec is an numpy array
        start_idx = 0
        for layer in self.layers:
            layer.weights = np.reshape(vec[start_idx: start_idx + (layer.inputs*layer.outputs)], (layer.inputs, layer.outputs))
            start_idx += (layer.inputs*layer.outputs) 
            layer.bias = vec[start_idx: start_idx + layer.outputs]
            start_idx += layer.outputs
    
    def copy(self):
        """
        Returns:
            exact copy of this NN 
        """
        new_layers = []
        for layer in self.layers:
            new_layers.append(layer.copy())
        return NN(*new_layers)
    
    def save(self) -> bytes:
        """
        Returns:
            bytes representing the NN
        """
        return pickle.dumps(self)

    @staticmethod
    def load(data: bytes) -> "NN":
        """
        Args:
            data: bytes representing a NN
        Returns:
            a NN 
        """
        return pickle.loads(data)
    
def relu(x):
    return np.maximum(x, 0)

def linear(x):
    return x

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum(axis=0)

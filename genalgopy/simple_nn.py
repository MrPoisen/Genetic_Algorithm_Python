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
    def __init__(self, *layers: List[Layer]) -> None:
        self.layers = layers
    
    def __call__(self, inp):
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
    
    def vectorize(self):
        vec = np.array([], dtype=np.double)
        for layer in self.layers:
            vec = np.append(vec, layer.vectorize())
        return vec
        #return np.ndarray.flatten(vec)
    
    def from_vector(self, vec):
        vec = np.array(vec, dtype=np.double) # ensure vec is an numpy array
        start_idx = 0
        for layer in self.layers:
            layer.weights = np.reshape(vec[start_idx: start_idx + (layer.inputs*layer.outputs)], (layer.inputs, layer.outputs))
            start_idx += (layer.inputs*layer.outputs) 
            layer.bias = vec[start_idx: start_idx + layer.outputs]
            start_idx += layer.outputs
    
    def copy(self):
        new_layers = []
        for layer in self.layers:
            new_layers.append(layer.copy())
        return NN(*new_layers)
    
    def save(self):
        return pickle.dumps(self)

    @staticmethod
    def load(data: bytes):
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

if __name__ == "__main__":
    l1 = Layer(2, 4, relu)
    l2 = Layer(4, 4, linear)
    l3 = Layer(4, 2, softmax)
    nn = NN(l1, l2, l3)
    print("n", nn([1, 2]))
    import pickle

    d = nn.save()
    m = NN.load(d)
    print("n", m([1, 2]))

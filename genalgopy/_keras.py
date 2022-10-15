#from tensorflow.keras import layers
#from tensorflow import keras

import numpy as np

try:
    from tensorflow import keras
    def _is_tensornetwork(obj):
        return isinstance(obj, (keras.Sequential, keras.Model)) or issubclass(type(obj), keras.Model)
except ImportError:
    def _is_tensornetwork(obj):
        return False 

def vectorize(model):
    vec = np.array([])
    for layer in model.layers:
        weights, bias = layer.get_weights()
        vec = np.append(vec, weights)
        vec = np.append(vec, bias)
        #vec.extend(weights)
        #vec.extend(bias)
    return vec

def from_vector(model, vec):
    vec = np.array(vec, dtype=np.double) # ensure vec is an numpy array
    start_idx = 0
    for layer in model.layers:
        existing_weights, existing_bias = layer.get_weights()
        
        weights = np.reshape(vec[start_idx: start_idx + existing_weights.size], existing_weights.shape)
        start_idx += existing_weights.size
        bias = vec[start_idx: start_idx + existing_bias.size]
        start_idx += existing_bias.size
        layer.set_weights((weights, bias))

def modelcopy(model):
    from tensorflow.keras.models import clone_model
    cop = clone_model(model)
    build = model.layers[0].input_shape
    cop.build(build)
    cop.set_weights(model.get_weights())
    return cop

def _make_empty(model):
    for layer in model.layers[1:-1]:
        weights, bias = layer.get_weights()
        weights = np.ones_like(weights)
        bias = np.zeros_like(bias)
        layer.set_weights((weights, bias))


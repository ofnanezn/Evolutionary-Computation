import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score, log_loss

"""
    Functions required by the Neural Network
"""
ALPHA = 2.0

def linear_forward(A_prev, W, b):
    return np.dot(W, A_prev) + b

def RELU(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1/(1+np.exp(-x)) 

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def powerLawGenerator(x):
    dir = np.random.choice([-1,1])
    return dir * (1.0 - x)**(1 - ALPHA)

class NeuralNetwork(object):
    """docstring for NeuralNetwork"""
    def __init__(self, hidden_layers, activation_functions):
        self.hidden_layers = hidden_layers
        self.activation_functions = activation_functions

    def initialize_weights(self, input_size, output_size):
        W = []
        b = []
        layers = [input_size] + self.hidden_layers + [output_size]
        for l in range(1, len(layers)):
            W.append(np.random.randn(layers[l],layers[l-1])*np.sqrt(1/(layers[l]+layers[l-1])))
            b.append(np.zeros((layers[l],1)))
        return W, b

    def forward_propagate(self, input_layer, W, b):
        n_layers = len(W)
        A = input_layer
        for l in range(n_layers):
            A_prev = A
            Z = linear_forward(A_prev, W[l], b[l])
            activation = eval(self.activation_functions[l])
            A = activation(Z) 
        return A

    def cost(self, AL, Y):
        y = Y.argmax(axis=0)
        m = Y.shape[1]
        log_likelihood = -np.log(AL[y,range(m)])
        loss = np.sum(log_likelihood) / m
        #cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
        return loss

    def HillClimbingTrain(self, X, W, b, Y, 
                          delta=0.6, num_iterations=1000, powerLaw=True):
        history_loss = []
        parameters = {}
        for i in range(num_iterations):
            AL = self.forward_propagate(X, W, b)
            cost = self.cost(AL, Y)
            W_prime = deepcopy(W)
            b_prime = deepcopy(b)
            if not powerLaw:
                for l in range(len(W_prime)):
                    W_prime[l] += delta * np.random.randn(W_prime[l].shape[0], W_prime[l].shape[1])
                    b_prime[l] += delta * np.random.randn(b_prime[l].shape[0], 1)
            else:
                for l in range(len(W_prime)):
                    W_prime[l] += delta * powerLawGenerator(np.random.randn(W_prime[l].shape[0], W_prime[l].shape[1]))
                    b_prime[l] += delta * powerLawGenerator(np.random.randn(b_prime[l].shape[0], 1))
            AL_prime = self.forward_propagate(X, W_prime, b_prime)
            cost_prime = self.cost(AL_prime, Y)
            #print(i, cost, cost_prime)
            if cost_prime <= cost:
                W = W_prime
                b = b_prime
                cost = cost_prime
            history_loss.append(cost)
        parameters["W"] = W
        parameters["b"] = b
        return parameters, history_loss



    def SimulatedAnnealingTrain(self, X, W, b, Y,
                                t_max=1000, delta=0.6, powerLaw=True):
        history_loss = []
        parameters = {}
        for t in range(1, t_max + 1):
            T = t / t_max
            AL = self.forward_propagate(X, W, b)
            cost = self.cost(AL, Y)
            W_prime = deepcopy(W)
            b_prime = deepcopy(b)
            if not powerLaw:
                for l in range(len(W_prime)):
                    W_prime[l] += delta * np.random.randn(W_prime[l].shape[0], W_prime[l].shape[1])
                    b_prime[l] += delta * np.random.randn(b_prime[l].shape[0], 1)
            else:
                for l in range(len(W_prime)):
                    W_prime[l] += delta * powerLawGenerator(np.random.randn(W_prime[l].shape[0], W_prime[l].shape[1]))
                    b_prime[l] += delta * powerLawGenerator(np.random.randn(b_prime[l].shape[0], 1))
            AL_prime = self.forward_propagate(X, W_prime, b_prime)
            cost_prime = self.cost(AL_prime, Y)
            DE = cost_prime - cost
            q = min(1,np.exp(-DE/T))
            if DE < 0:
                W = W_prime
                b = b_prime
                cost = cost_prime
            elif T != 0 and np.random.random() >= q:
                W = W_prime
                b = b_prime
                cost = cost_prime
            history_loss.append(cost)
        parameters["W"] = W_prime
        parameters["b"] = b
        return parameters, history_loss

    def evaluateModel(self, X, Y, parameters):
        W = parameters["W"]
        b = parameters["b"]
        Y_prime = self.forward_propagate(X, W, b)
        Y_true = np.argmax(Y, axis=0)
        Y_predicted = np.argmax(Y_prime, axis=0)
        return accuracy_score(Y_true, Y_predicted)

class ConvLayer():
    def __init__(self, pad, f, slide, n_F):
        """
        pad -- Zero-padding size
        f -- Filter size
        """
        self.pad = pad
        self.f = f
        self.slide = slide
        self.n_F = n_F

    def zero_pad(self, X):
        #Pad with zeros all images of the dataset X
        p = self.pad
        X_pad = np.pad(X, ((0,0),(p,p),(p,p),(0,0)), 'constant')    
        return X_pad

    def conv_single_step(self, a_slice_prev, W, b):
        """
        Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
        of the previous layer.
        """
        s = a_slice_prev * W
        # Sum over all entries of the volume s.
        Z = np.sum(s)
        # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
        Z = Z + float(b)

        return Z

    def conv_forward(self, A_prev, W, b):
        """
        Implements the forward propagation for a convolution function
        
        Arguments:
        A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
        b -- Biases, numpy array of shape (1, 1, 1, n_C)
        hparameters -- python dictionary containing "stride" and "pad"
            
        Returns:
        Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward() function
        """
    
        # Retrieve dimensions from A_prev's shape (≈1 line)  
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        # Retrieve dimensions from W's shape (≈1 line)
        (f, f, n_C_prev, n_C) = W.shape

        # Retrieve information from "hparameters" (≈2 lines)
        stride = self.stride
        pad = self.pad
        
        # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
        n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
        n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
        
        # Initialize the output volume Z with zeros. (≈1 line)
        Z = np.zeros((m, n_H, n_W, n_C))
        
        # Create A_prev_pad by padding A_prev
        A_prev_pad = zero_pad(A_prev, pad)
        
        for i in range(m):                                 # loop over the batch of training examples
            a_prev_pad = A_prev_pad[i]                     # Select ith training example's padded activation
            for h in range(n_H):                           # loop over vertical axis of the output volume
                for w in range(n_W):                       # loop over horizontal axis of the output volume
                    for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                        # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                        Z[i, h, w, c] = conv_single_step(a_slice_prev, W[...,c], b[...,c])
                                            
        # Making sure your output shape is correct
        assert(Z.shape == (m, n_H, n_W, n_C))
        
        return Z

    def initialize_weights(self, input):
        n_C = input.shape[-1]
        return np.random.randn(self.f, self.f, n_C, self.n_F)*np.sqrt(1/(layers[l]+layers[l-1]))

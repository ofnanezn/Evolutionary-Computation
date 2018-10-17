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
            W.append(np.random.randn(layers[l],layers[l-1]))
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



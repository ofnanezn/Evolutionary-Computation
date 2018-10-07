import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score

"""
	Functions required by the Neural Network
"""
def linear_forward(A_prev, W, b):
	return np.dot(W, A_prev) + b

def RELU(x):
	return np.maximum(0, x)

def sigmoid(x):
	return 1/(1+np.exp(-x)) 


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
		m = Y.shape[1]
		cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
		return np.squeeze(cost)

	def trainHillClimbing(self, X, W, b, Y, delta=0.6, num_iterations=1000):
		history_loss = []
		parameters = {}
		for i in range(num_iterations):
			AL = self.forward_propagate(X, W, b)
			cost = self.cost(AL, Y)
			W_prime = deepcopy(W)
			b_prime = deepcopy(b)
			for l in range(len(W_prime)):
				W_prime[l] += delta * np.random.randn(W_prime[l].shape[0], W_prime[l].shape[1])
				b_prime[l] += delta * np.random.randn(b_prime[l].shape[0], 1)
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

	def evaluateModel(self, X, Y, parameters):
		W = parameters["W"]
		b = parameters["b"]
		Y_prime = self.forward_propagate(X, W, b)
		Y_true = np.argmax(Y, axis=0)
		Y_predicted = np.argmax(Y_prime, axis=0)
		return accuracy_score(Y_true, Y_predicted)



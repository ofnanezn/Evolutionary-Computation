import numpy as np

def linear_forward(A_prev, W, b):
	return np.dot(W, A_prev) + b

def RELU(x):
	return x if x > 0 else 0

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
		for l in layers(n_layers):
			A_prev = A
			Z = linear_forward(A_prev, W[l], b[l])
			activation = self.activation_functions[l]
			A = activation(Z) 
		return A

	def cost(AL, Y):
		m = Y.shape[1]
		cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
		return np.squeeze(cost)




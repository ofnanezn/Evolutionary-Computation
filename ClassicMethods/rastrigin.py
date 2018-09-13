"""
	Class for Rastrigin definition
"""
import numpy as np
from function import Function

class Rastrigin(Function):
	"""docstring for Rastrigin"""
	def __init__(self, x):
		self.x = x
	
	def evalFunc(self, x):
		A = 10
		n = len(x)
		return A * n + np.sum(x**2 - A * np.cos(2*np.pi*x))

	def gradient(x):
		pass

	def hessian(x):
		pass
"""
	Class for optimization methods
"""

from rastrigin import Rastrigin
import numpy as np

NUM_DIM = 2
alpha = 2.0

class Optimization():
	"""docstring for Optimization"""
	def __init__(self):
		pass
		#self.num_dim = num_dim	

	def powerLawGenerator(self, x):
		dir = np.random.choice([-1,1])
		return dir * (1.0 - x)**(1 - alpha)

	def evalFunc(self, f, x):
		return f.evalFunc(x)

	def gradient(self, f, x):
		return f.gradient(x)

	def hessian(self, f, x):
		return f.hessian(x)

	def newtonRaphsonStep(self, f, x):
		grad = gradient(f, x)
		h = hessian(f, x)
		return x - (grad / h)

	def gradientDescentStep(self, f, x, alpha):
		grad = gradient(f, x)
		return x - (alpha * grad)

	def hillClimbingStep(self, f, x, delta, powerLaw):
		n = len(x)
		y = np.array(x)
		fx = self.evalFunc(f, x)
		if not powerLaw:
			y += delta * np.random.randn(n)
		else:
			y += delta * self.powerLawGenerator(np.random.randn(n))
		fx_prime = self.evalFunc(f, y)
		return (y if fx_prime <= fx else x)

	def newtonRaphson(self, x):
		f = Rastrigin(x)
		for i in range(1000):
			print(x)
			x_next = newtonRaphsonStep(f, x)
			x = x_next
		return x

	def gradientDescent(self, x, alpha=0.01):
		f = Rastrigin(x)
		for i in range(1000):
			x_next = gradientDescentStep(f, x, alpha)
			x = x_next
		return x

	def gradientDescentWithMomentum(self, x, alpha=0.01, betha=0.9):
		f = Rastrigin(x)
		vx = np.zeros(len(x))
		for i in range(1000):
			grad = gradient(f, x)
			vx += betha * vx + (1 - betha) * grad
			x_next = x_next - vx * alpha 
			x = x_next
		return x

	def hillClimbing(self, f, x, delta, powerLaw=True):
		for i in range(1000):
			x_next = self.hillClimbingStep(f, x, delta, powerLaw)
			x = x_next
			print(x, self.evalFunc(f, x))
		return x



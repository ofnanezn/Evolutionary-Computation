"""
	Class for representation and Operation of Bit arrays
"""
import random

class GeneticAlgorithm():
	def __init__(self, chrom_size):
		self.chrom_size = chrom_size

	def create_individual(self):
		return random.randint(0, 2**self.chrom_size)

	def mutation(self, chromosome):
		pos = random.randint(0, chrom_size)
		return chromosome ^ 2**pos

	def crossover(self, parent1, parent2, cross_point):
		mask1 = sum([2**i for i in range(cross_point)])
		mask2 = sum([2**i for i in range(cross_point, self.chrom_size)])
		cross_parent11 = parent1 & mask1
		cross_parent12 = parent1 & mask2
		cross_parent21 = parent2 & mask1
		cross_parent22 = parent2 & mask2
		child1 = cross_parent11 | cross_parent22
		child2 = cross_parent21 | cross_parent12
		return (child1, child2)

	def calculate_fitness(self, x, f):
		return f(x)

	def print_individual(self, x):
		print(bin(x))


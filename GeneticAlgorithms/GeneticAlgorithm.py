"""
	Class for representation and Operation of Bit arrays
"""
from bitstring import BitArray
import numpy as np

class GeneticAlgorithm():
	def __init__(self, initial_pop):
		self.initial_pop = initial_pop

	def initial_individual(num_genes):
		individual = [np.random.choice([True, False]) for i in range(num_genes)]
		return BitArray(individual)

	def mutation(self, chromosome):
		pos = np.random.randint(0, len(chromosome))
		new_chromosome = chromosome.copy()
		new_chromosome.invert(pos)
		return new_chromosome

	def crossover(self, parent1, parent2, cross_point):
		num_chrom = len(parent1)
		cut_right = num_chrom - cross_point 
		mask1 = BitArray([True for i in range(cross_point)] + [False for i in range(cut_right)])
		mask2 = BitArray([False for i in range(cross_point)] + [True for i in range(cut_right)])
		cross_parent11 = parent1 & mask1
		cross_parent12 = parent1 & mask2
		cross_parent21 = parent2 & mask1
		cross_parent22 = parent2 & mask2
		child1 = cross_parent11 | cross_parent22
		child2 = cross_parent21 | cross_parent12
		return (child1, child2)

	def calculate_fitness(f, x):
		return f(x)

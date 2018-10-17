"""
	Class for representation and Operation of Bit arrays
"""
import random

class Individual():
	def __init__(self, chromosome):
		self.chromosome = chromosome
		self.fitness = None

	def calculate_fitness(self, f):
		self.fitness = f(self.chromosome)

	def print_individual(self):
		print("Individual: " + bin(self.chromosome))
		print("Fitness: " + str(self.fitness))

		
class GeneticAlgorithm():
	def __init__(self, chrom_size):
		self.chrom_size = chrom_size

	def create_individual(self):
		return Individual(random.randint(0, sum([2**i for i in range(self.chrom_size)])))

	def mutation(self, chromosome):
		pos = random.randrange(0, self.chrom_size)
		return Individual(chromosome ^ 2**pos)

	def crossover(self, parent1, parent2, cross_point):
		mask1 = sum([2**i for i in range(cross_point)])
		mask2 = sum([2**i for i in range(cross_point, self.chrom_size)])
		cross_parent11 = parent1 & mask1
		cross_parent12 = parent1 & mask2
		cross_parent21 = parent2 & mask1
		cross_parent22 = parent2 & mask2
		child1 = cross_parent11 | cross_parent22
		child2 = cross_parent21 | cross_parent12
		return (Individual(child1), Individual(child2))

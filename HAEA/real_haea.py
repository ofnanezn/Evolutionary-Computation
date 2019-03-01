from random import randint, random, randrange

from constants import CROSSOVER_OPERATOR, MUTATION_OPERATOR, PROPORTIONAL_SELECTION, RANK_SELECTION
from individual import Individual
from real_selections import proportional, rank, tournament, elitist

import numpy as np

def d(x, y):
	return np.linalg.norm(x.chromosome - y.chromosome)

class RealHAEA():
	def __init__( self, function, num_dims, populationLength, limits ):
		self.function = function
		self.num_dims = num_dims
		self.populationLength = populationLength
		self.population = []
		self.limits = limits

	def crossover( self, parent1, parent2, crossPoint ):
		child1 = Individual(
			chromosome = np.append(parent1[:crossPoint], parent2[crossPoint:]),
			function = self.function
		)
		child2 = Individual(
			chromosome = np.append(parent2[:crossPoint], parent1[crossPoint:]),
			function = self.function
		)

		return [ child1, child2 ]

	def mutation( self, chromosome ):
		pos = randrange( 0, self.num_dims )
		lim_min, lim_max = self.limits[0], self.limits[1]
		sigma = (lim_max - lim_min) / 100
		new_chrom = chromosome[:]
		new_chrom[pos] += np.random.randn() * sigma
		if new_chrom[pos] > lim_max or new_chrom[pos] < lim_min:
			return Individual(chromosome = chromosome, function = self.function)
		child = Individual(
			chromosome = new_chrom,
			function = self.function
		)
		return child

	def LC_crossover( self, p1, p2):
		alpha = np.random.uniform()
		c1 = alpha * p1 + (1 - alpha) * p2
		c2 = (1 - alpha) * p1 + alpha * p2
		child1 = Individual(
			chromosome = c1,
			function = self.function
		)
		child2 = Individual(
			chromosome = c2,
			function = self.function
		)
		return [ child1, child2 ]	

	def G_mutation( self, chromosome):
		pos = randrange( 0, self.num_dims )
		lim_min, lim_max = self.limits[0], self.limits[1]
		sigma = (lim_max - lim_min) / 100
		new_chrom = np.array(chromosome[:])
		new_chrom[pos] += np.random.randn() * sigma
		if new_chrom[pos] > lim_max or new_chrom[pos] < lim_min:
			return Individual(chromosome = chromosome, function = self.function)
		child = Individual(
			chromosome = new_chrom,
			function = self.function
		)
		return child

	def createPopulation( self ):
		lim_min, lim_max = self.limits[0], self.limits[1]
		self.population = np.array([Individual(
			chromosome = np.random.uniform(lim_min, lim_max, self.num_dims),
			function = self.function
		) for _ in range( self.populationLength )])

	def selectOperator( self, individual ):
		probability = random()

		if probability < individual.mutationRate:
			return CROSSOVER_OPERATOR

		return MUTATION_OPERATOR

	def selectParents( self, selection ):
		if selection == PROPORTIONAL_SELECTION:
			parents = proportional( self.population, 1 )
		elif selection == RANK_SELECTION:
			parents = rank( self.population, 1 )
		elif selection == "tournament":
			parents = tournament(self.population, 1)
		elif selection == "elitist":
			parents = elitist(self.population, 1)

		return parents

	def applyOperator( self, operator, parents ):
		offspring = []

		if operator == CROSSOVER_OPERATOR:
			#crossPoint = randrange( 1, self.num_dims )
			children = self.LC_crossover( parents[0].chromosome, parents[1].chromosome)
		else: # operator == MUTATION_OPERATOR
			children = [self.G_mutation( parents[0].chromosome )]
		offspring += children

		return offspring

	def best( self, offspring ):
		return min( offspring, key = lambda individual: individual.fitness )

	def best_prime(self, offspring, individual):
		N = len(offspring)
		x = offspring[0]
		min_x = d(individual, offspring[0])
		for i in range(1, N):
			if d(individual, offspring[i]) > 0 and d(individual, offspring[i]) < min_x:
				x = offspring[i]
				min_x = d(individual, offspring[i])
		if individual.fitness > x.fitness:
			x = individual
		return x

	def recalculateRates( self, operator, child, individual ):
		sigma = random()
		crossoverRate = individual.crossoverRate
		mutationRate = individual.mutationRate

		if child.fitness >= individual.fitness:
			if operator == CROSSOVER_OPERATOR:
				crossoverRate *= ( 1 + sigma )
			else: # operator == MUTATION_OPERATOR
				mutationRate *= ( 1 + sigma )
		else:
			if operator == CROSSOVER_OPERATOR:
				crossoverRate *= ( 1 - sigma )
			else: # operator == MUTATION_OPERATOR
				mutationRate *= ( 1 - sigma )

		total = crossoverRate + mutationRate
		crossoverRate /= total
		mutationRate /= total

		child.crossoverRate = crossoverRate
		child.mutationRate = mutationRate

	def init( self, generations, selection ):
		# Create initial population
		self.createPopulation()

		# Initialize data to return
		#data = []

		# Generations
		for i in range( generations ):
			newPopulation = []
			for individual in self.population:
				# Select operator to apply
				operator = self.selectOperator( individual )

				# Apply operator
				if operator == CROSSOVER_OPERATOR:
					parents = self.selectParents( selection )
					parents = [parents[randint( 0, len( parents ) - 1 )]] + [individual]
				else: # operator == MUTATION_OPERATOR
					parents = [individual]
				offspring = self.applyOperator( operator, parents )

				# Choose the best individual
				child = self.best_prime( offspring, individual )
				# print(child.fitness, self.function(child.chromosome), operator)

				# Recalculate operator rates
				self.recalculateRates( operator, child, individual )

				# Add child to newPopulation
				newPopulation.append( child )

			self.population = np.array(newPopulation)
			#data.append( self.best( self.population ) )

		return self.population
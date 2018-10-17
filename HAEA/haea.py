from random import randint, random, randrange

from constants import CROSSOVER_OPERATOR, MUTATION_OPERATOR, PROPORTIONAL_SELECTION, RANK_SELECTION
from individual import Individual
from selections import proportional, rank, tournament, elitist

class HAEA():
	def __init__( self, function, chromosomeLength, populationLength ):
		self.function = function
		self.chromosomeLength = chromosomeLength
		self.populationLength = populationLength
		self.population = []

	def crossover( self, parent1, parent2, crossPoint ):
		mask1 = sum( [2 ** i for i in range( crossPoint )] )
		mask2 = sum( [2 ** i for i in range( crossPoint, self.chromosomeLength )] )
		crossParent11 = parent1 & mask1
		crossParent12 = parent1 & mask2
		crossParent21 = parent2 & mask1
		crossParent22 = parent2 & mask2
		child1 = Individual(
			chromosome = crossParent11 | crossParent22,
			function = self.function
		)
		child2 = Individual(
			chromosome = crossParent21 | crossParent12,
			function = self.function
		)

		return ( child1, child2 )

	def mutation( self, chromosome ):
		position = randrange( 0, self.chromosomeLength )
		child = Individual(
			chromosome = chromosome ^ 2 ** position,
			function = self.function
		)

		return child

	def createPopulation( self ):
		self.population = [Individual(
			chromosome = randint( 0, sum( [2 ** i for i in range( self.chromosomeLength )] ) ),
			function = self.function
		) for _ in range( self.populationLength )]

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
		offspring = parents

		if operator == CROSSOVER_OPERATOR:
			crossPoint = randint( 1, self.chromosomeLength )
			children = self.crossover( parents[0].chromosome, parents[1].chromosome, crossPoint )
		else: # operator == MUTATION_OPERATOR
			children = [self.mutation( parents[0].chromosome )]
		offspring += children

		return offspring

	def best( self, offspring ):
		return max( offspring, key = lambda individual: individual.fitness )

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
		data = []

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
				child = self.best( offspring )

				# Recalculate operator rates
				self.recalculateRates( operator, child, individual )

				# Add child to newPopulation
				newPopulation.append( child )
			self.population = newPopulation

			data.append( self.best( self.population ) )

		return data
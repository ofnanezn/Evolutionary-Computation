from random import random
import numpy

def proportional( population, parentsLength ):
	parents = []
	totalFitness = sum( individual.fitness for individual in population )
	for i in range( parentsLength ):
		randomNumber = random()
		probabilityAcumulated = 0
		for individual in population:
			probability = individual.fitness / totalFitness
			probabilityAcumulated += probability
			if randomNumber <= probabilityAcumulated:
				parents.append( individual )
				break
	return parents

def rank( population, parentsLength ):
	parents = []
	populationLength = len( population )
	population.sort( key = lambda individual: individual.fitness, reverse = True )
	for i in range( parentsLength ):
		randomNumber = random()
		for j in range( populationLength ):
			individual = population[j]
			probability = ( j + 1 ) / populationLength

			if randomNumber <= probability:
				parents.append( individual )
				break
	return parents

def tournament(pop, num_parents, k=20):
    n = len(pop)
    selected_parents = []
    for i in range(num_parents):
        k_pop = numpy.random.choice(pop, size=k)
        best = max(k_pop, key=lambda x: x.fitness)
        selected_parents.append(best)
    return selected_parents

def elitist(pop, num_parents):
    n = len(pop)
    ranked_pop = sorted(pop, key=lambda x: x.fitness, reverse=True)
    selected_parents = ranked_pop[:int(len(pop)*0.1)]
    if len(selected_parents) >= num_parents:
        return selected_parents[:num_parents]
    rest_amount = num_parents - len(selected_parents)
    no_elit = list(numpy.random.choice(pop[int(len(pop)*0.1):int(len(pop)*0.9)], size=rest_amount))
    return selected_parents + no_elit 
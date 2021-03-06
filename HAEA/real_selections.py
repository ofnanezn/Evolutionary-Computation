import random
import numpy as np

def proportional(pop, num_parents):
    selected_parents = []
    total_fitness = sum([ind.fitness for ind in pop])
    for i in range(num_parents):
        r = np.random.uniform()
        q_acum = 0
        for ind in pop:
            p_ind = ind.fitness / total_fitness
            q_ind = (1 - p_ind)/(len(pop)-1)
            q_acum += q_ind
            if r <= q_acum:
                selected_parents.append(ind)
                break
    return selected_parents

def rank( population, parentsLength ):
    parents = []
    populationLength = len( population )
    ranked_pop = sorted(population, key=lambda individual: individual.fitness)
    for i in range( parentsLength ):
        randomNumber = random.random()
        for j in range( populationLength ):
            individual = ranked_pop[j]
            probability = ( j + 1 ) / populationLength

            if randomNumber <= probability:
                parents.append( individual )
                break
    return parents

def tournament(pop, num_parents, k=20):
    n = len(pop)
    selected_parents = []
    for i in range(num_parents):
        k_pop = np.random.choice(pop, size=k)
        best = min(k_pop, key=lambda x: x.fitness)
        selected_parents.append(best)
    return selected_parents

def elitist(pop, num_parents):
    n = len(pop)
    ranked_pop = sorted(pop, key=lambda x: x.fitness)
    selected_parents = ranked_pop[:int(len(pop)*0.1)]
    if len(selected_parents) >= num_parents:
        return selected_parents[:num_parents]
    rest_amount = num_parents - len(selected_parents)
    no_elit = list(np.random.choice(pop[int(len(pop)*0.1):int(len(pop)*0.9)], size=rest_amount))
    return selected_parents + no_elit 
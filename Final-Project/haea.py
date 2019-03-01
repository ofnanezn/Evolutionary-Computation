import numpy as np
from NeuralNetwork import NeuralNetwork
from copy import deepcopy
from selection import *

SIGMA = 0.1

def generational(parents, children):
    return children

def steady_state(parents, children):
    return sorted(parents + children, key=lambda x: x.fitness,
                reverse=False)[:len(parents)]

class Individual():
    def __init__(self, W, b, activation_functions, hidden_layers=None):
        self.W = W
        self.b = b
        self.fitness = None
        self.activation_functions = activation_functions
        self.crossoverRate = np.random.random()
        self.mutationRate = 1 - self.crossoverRate

class HAEA():
    def __init__(self, pop_size):
        self.pop_size = pop_size

    def initial_individual(self, input_size, output_size,
                           hidden_layers, activation_functions):
        individual = NeuralNetwork(hidden_layers, activation_functions)
        W, b = individual.initialize_weights(input_size, output_size)
        return Individual(W, b, activation_functions)

    def initialize_pop(self, input_size, output_size,
                       hidden_layers, activation_functions):
        P = []
        for i in range(self.pop_size):
            ind = self.initial_individual(input_size, output_size,
                        hidden_layers, activation_functions)
            P.append(ind)

        return P

    def selectOperator( self, individual ):
        probability = np.random.random()

        if probability < individual.mutationRate:
            return "crossover"

        return "mutation"

    def applyOperator( self, operator, parents ):
        offspring = []

        if operator == "crossover":
            children = self.crossover( parents[0], parents[1] )
        else: # operator == MUTATION_OPERATOR
            children = [self.mutation( parents[0] )]
        offspring += children

        return offspring

    def mutation(self, individual):
        W = deepcopy(individual.W)
        b = deepcopy(individual.b)
        activation_functions = individual.activation_functions
        num_layers = len(W)

        for i in range(num_layers):
            W[i] += np.random.randn(W[i].shape[0], W[i].shape[1]) * SIGMA
            b[i] += np.random.randn(b[i].shape[0], 1) * SIGMA

        return Individual(W, b, individual.activation_functions)

    def correct_weights(self, W, cross_point):
        diff1 = W[cross_point - 1].shape[0] - W[cross_point].shape[1]
        diff2 = W[-2].shape[0] - W[-1].shape[1]

        if diff1 > 0:
            rem_neurons = np.random.randn(W[cross_point].shape[0], diff1) * np.sqrt(1/(W[cross_point].shape[0]+W[cross_point].shape[1]))
            W[cross_point] = np.append(W[cross_point], rem_neurons, axis=1)
        elif diff1 < 0:
            W[cross_point] = W[cross_point][:,:W[cross_point - 1].shape[0]]

        if diff2 > 0:
            W[-2] = W[-2][:W[-1].shape[1],:]
        elif diff2 < 0:
            rem_neurons = np.random.randn(np.abs(diff1), W[-2].shape[1])  * np.sqrt(1/(W[-2].shape[0]+W[-2].shape[1]))
            W[-2] = np.append(W[-2], rem_neurons, axis=0)


    def crossover(self, individual_1, individual_2):
        num_weights1 = len(individual_1.W)
        num_weights2 = len(individual_2.W)

        if num_weights1 < 3 or num_weights2 < 3:
            return [individual_1, individual_2]

        cross_point1 = np.random.randint(1, num_weights1 - 1)
        cross_point2 = np.random.randint(1, num_weights2 - 1)
        
        W1 = individual_1.W[:cross_point1] + individual_2.W[cross_point2: num_weights2 - 1] + [individual_1.W[-1]]
        W2 = individual_2.W[:cross_point2] + individual_1.W[cross_point1: num_weights1 - 1] + [individual_2.W[-1]]

        b1 = individual_1.b[:cross_point1] + individual_2.b[cross_point2: num_weights2 - 1] + [individual_1.b[-1]]
        b2 = individual_2.b[:cross_point2] + individual_1.b[cross_point1: num_weights1 - 1] + [individual_2.b[-1]]

        activation1 = individual_1.activation_functions[:cross_point1] + individual_2.activation_functions[cross_point2: num_weights2 - 1] + [individual_1.activation_functions[-1]]
        activation2 = individual_2.activation_functions[:cross_point2] + individual_1.activation_functions[cross_point1: num_weights1 - 1] + [individual_2.activation_functions[-1]]

        self.correct_weights(W1, cross_point1)
        self.correct_weights(W2, cross_point2) 

        return [Individual(W1, b1, activation1), Individual(W2, b2, activation2)]

    def recalculateRates( self, operator, child, individual ):
        sigma = np.random.random()
        crossoverRate = individual.crossoverRate
        mutationRate = individual.mutationRate

        if child.fitness >= individual.fitness:
            if operator == "crossover":
                crossoverRate *= ( 1 + sigma )
            else: # operator == MUTATION_OPERATOR
                mutationRate *= ( 1 + sigma )
        else:
            if operator == "mutation":
                crossoverRate *= ( 1 - sigma )
            else: # operator == MUTATION_OPERATOR
                mutationRate *= ( 1 - sigma )

        total = crossoverRate + mutationRate
        crossoverRate /= total
        mutationRate /= total

        child.crossoverRate = crossoverRate
        child.mutationRate = mutationRate

    def best( self, offspring ):
        return min( offspring, key = lambda individual: individual.fitness )

    def haea_iter(self, T,  input, output,
                    hidden_layers, activation_functions):
        P = self.initialize_pop(input.shape[0], output.shape[0], 
                        hidden_layers, activation_functions)
        fitness_iter = []
        history_loss = []
        history_crossover = []
        history_mutation = []

        for individual in P:
            NN = NeuralNetwork(None, individual.activation_functions)
            #individual.calculate_fitness(fitness)
            AL = NN.forward_propagate(input, individual.W, individual.b)
            cost = NN.cost(AL, output)
            individual.fitness = cost

        crossover_avr = np.mean([individual.crossoverRate for individual in P])
        mutation_avr = np.mean([individual.mutationRate for individual in P])

        history_crossover.append(crossover_avr)
        history_mutation.append(mutation_avr)

        for t in range(T):
            new_pop = []

            for individual in P:
                operator = self.selectOperator( individual )
                
                if operator == "crossover":
                    parents = [np.random.choice(P)]
                    parents = parents + [individual]
                else: # operator == MUTATION_OPERATOR
                    parents = [individual]

                offspring = self.applyOperator( operator, parents )

                #child2 = GA.mutation(child2.chromosome)
                for child in offspring:
                    NN = NeuralNetwork(None, child.activation_functions)
                    AL = NN.forward_propagate(input, child.W, child.b)
                    cost = NN.cost(AL, output)
                    child.fitness = cost

                new_gen = self.best([individual] + offspring)
                self.recalculateRates( operator, child, individual )

                new_pop.append(new_gen)

            P = new_pop
            history_loss.append(min(P, key=lambda x: x.fitness).fitness)

            crossover_avr = np.mean([individual.crossoverRate for individual in P])
            mutation_avr = np.mean([individual.mutationRate for individual in P])

            history_crossover.append(crossover_avr)
            history_mutation.append(mutation_avr)
            #print(min(P, key=lambda x: x.fitness).fitness)

        best = min(P, key=lambda x: x.fitness)
        return best, history_loss, history_crossover, history_mutation



    


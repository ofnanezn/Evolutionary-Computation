"""
    Class for representation and Operation of Bit arrays
"""
import random

def real2bin(x, limits, num_decimals):
    x_bin = []
    limit_bin = int(limits[1] * (10**num_decimals))
    num_bits = len(bin(limit_bin)) - 2
    for x_i in x:
        if x_i < 0:
            x_bin.append(round(x_i*(10**num_decimals) + 2**(num_bits+1)))
        else:
            x_bin.append(round(x_i*(10**num_decimals)))
    return x_bin

def bin2real(x, limits, num_decimals):
    x_real = []
    limit_bin = int(limits[1] * (10**num_decimals))
    num_bits = len(bin(limit_bin)) - 2
    for x_i in x:
        if x_i > limit_bin:
            x_real.append((x_i- 2**(num_bits+1))/(10**num_decimals))
        else:
            x_real.append(x_i/(10**num_decimals))
    return x_real

class Individual():
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = None

    def calculate_fitness(self, f, limits, num_decimals):
        real_repr = bin2real(self.chromosome, limits, num_decimals)
        self.fitness = f(real_repr)

    def print_individual(self):
        print("Individual: " + bin(self.chromosome))
        print("Fitness: " + str(self.fitness))

        
class RealGeneticAlgorithm():
    def __init__(self, chrom_size, num_dim):
        self.chrom_size = chrom_size
        self.num_dim = num_dim

    def create_individual(self, limits, num_decimals):
        inds = []
        for i in range(self.num_dim):
            inds.append(round(random.uniform(limits[0], limits[1]), num_decimals))
        inds_bin = real2bin(inds, limits, num_decimals)
        return Individual(inds_bin)

    def mutation(self, chromosome, limits, num_decimals):
        pos = random.randrange(0, self.num_dim)
        pos_chrom = random.randrange(0, self.chrom_size)
        new_ind = chromosome[:]
        a = new_ind[pos] & sum([2**i for i in range(self.chrom_size)])
        if a > int(limits[1] * (10**num_decimals)):
            return Individual(chromosome)
        new_ind[pos] = chromosome[pos] ^ 2**pos_chrom
        return Individual(new_ind)

    def crossover(self, parent1, parent2, cross_point):
        mask1 = [sum([2**i for i in range(self.chrom_size)]) for j in range(cross_point)]
        mask1 += [0 for j in range(cross_point, self.num_dim)]
        mask2 = [0 for j in range(cross_point)]
        mask2 += [sum([2**i for i in range(self.chrom_size)]) for j in range(cross_point, self.num_dim)]
        cross_parent11 = [parent1.chromosome[i] & mask1[i] for i in range(self.num_dim)]
        cross_parent12 = [parent1.chromosome[i] & mask2[i] for i in range(self.num_dim)]
        cross_parent21 = [parent2.chromosome[i] & mask1[i] for i in range(self.num_dim)]
        cross_parent22 = [parent2.chromosome[i] & mask2[i] for i in range(self.num_dim)]
        child1 = [cross_parent11[i] | cross_parent22[i] for i in range(self.num_dim)]
        child2 = [cross_parent21[i] | cross_parent12[i] for i in range(self.num_dim)]
        return (Individual(child1), Individual(child2))

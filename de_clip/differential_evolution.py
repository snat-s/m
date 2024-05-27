import numpy as np
from tqdm.notebook import trange
import random


class DE:
    def __init__(self, func, bounds, G=100, args=(), population_size=15, mutation=(0.5, 2), crossover=0.7):
        self.func = func
        self.bounds = np.array(bounds)
        self.args = args
        self.population = None
        self.fitness = None
        self.population_size = population_size
        self.mutation = mutation
        self.crossover = crossover
        self.G = G

    def __init_population(self):
        nvar = len(self.bounds)
        self.population = np.zeros((self.population_size, nvar))
        for i in range(nvar):
            self.population[:, i] = np.random.uniform(self.bounds[i, 0], self.bounds[i, 1], self.population_size)

    def __fitness_population(self):
        self.fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            self.fitness[i] = self.func(self.population[i, :], *self.args)

    def __individual_fitness(self, individual):
        return self.func(individual, *self.args)

    def __mutation(self):
        indices = random.sample(range(self.population_size), 3)
        f = random.uniform(*self.mutation)
        mutated_individual = self.population[indices[0], :] + f * (
                    self.population[indices[1], :] - self.population[indices[2], :])
        return mutated_individual

    def __crossover(self, individual, mutated_individual):
        N = len(self.bounds)
        feature_idx = random.randint(0, N)
        child = mutated_individual
        for i in range(N):
            if i == feature_idx:
                continue
            prob = random.uniform(0, 1)
            if prob > self.crossover:
                child[i] = individual[i]
        return child

    def optimize(self):
        self.__init_population()
        self.__fitness_population()

        for i in range(self.G):
            for j in range(self.population_size):
                mutated_individual = self.__mutation()
                child = self.__crossover(self.population[j, :], mutated_individual)
                fitness = self.__individual_fitness(child)
                if fitness < self.fitness[j]:
                    self.population[j, :] = child
                    self.fitness[j] = fitness

            best_fitness = np.min(self.fitness)
            #if best_fitness < 0.0001:
            #    break

        # Find the index of the minimum value
        elite_index = np.argmin(self.fitness)
        return self.population[elite_index, :]


def differential_evolution(func, bounds, args=(), popsize=15, mutation=(0.5, 1), crossover=0.7):
    de = DE(func, bounds, args, popsize, mutation, crossover)
    return de.optimize()

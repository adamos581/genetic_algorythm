import numpy as np
from numpy import random

class Environment:

    def __init__(self, amount, capacity):
        self.weights = random.random_integers(0, 10,amount)
        self.value = random.random_integers(0, 10,amount)
        self.capacity = capacity


class Species:

    def __init__(self, amount):
        self.backpack = random.random_integers(0, 1,amount)
        self.standard_deviation = random.rand(amount)

    def fit(self, environment):
        weight = [x for i, x in zip(self.backpack, environment.weights) if i == 1]
        sum_weight = np.sum(weight)
        value = [x for i, x in zip(self.backpack, environment.value) if i == 1]
        sum_value = np.sum(value)
        return sum_weight, sum_value


def check_population(population, environment):
    population_weights = []
    population_values = []

    for person in population:
        sum_weight, sum_value = person.fit(environment)
        population_values.append(sum_value)
        population_weights.append(sum_weight)


def create_parents(population, number_population):
    parents = []
    for i in range(number_population * 7):
        parents.append(random.choice(population))
    return parents


def uniform_crossover(parents, len_items):
    parents_crossover = []

    while len(parents) > 0:
        first_person = parents.pop(random.random_integers(0, len(parents)))
        second_person = parents.pop(random.random_integers(0, len(parents)))

        if random.rand() > 0.6:
            pattern = random.random_integers(0, 1, len_items)
            for i, cross in enumerate(pattern):
                if cross == 1:
                    r = first_person.backpack[i]
                    first_person.backpack[i] = second_person.backpack[i]
                    second_person.backpack[i] = r

            alfa = random.rand()
            new_first_deviation = alfa * first_person.standard_deviation + (1 - alfa) * second_person.standard_deviation
            new_second_deviation = alfa * second_person.standard_deviation + (1 - alfa) * first_person.standard_deviation
            first_person.standard_deviation = new_first_deviation
            second_person.standard_deviation = new_second_deviation

        parents_crossover.append(first_person)
        parents_crossover.append(second_person)

    return parents_crossover


# def mutation(parents):



if __name__ == '__main__':
    number_population = 20
    population = [Species(150) for i in range(number_population)]

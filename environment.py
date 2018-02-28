import numpy as np
from numpy import random
import copy
import matplotlib.pyplot as plt
import time
"""
Program do optymalizacji z wykorzystaniem algorytmów genetycznych. Program ten generuje losowo wektor przedmiotów, gdzie
każdy z nich posiada określoną wartość i wagę. Optymalizacja polega na zapakowaniu plecaka mającego okresloną pojemność,
przedmiotami tak by przedmioty te miały jak najwiekszą wartość.
W tym celu została wykorzystana strategia ewolucyjna (mi + lambda).
W wyniku powstaje wykres przedstawiający 5 różnych generacji zawierający informację o średniej wartości funkcji 
przystosowania danej generacji wraz z odchyleniem standardowym (dzięki temu można zaobserwować różnorodność populacji,
a także zbieżność algorytmu).
"""


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
        weight = [x for i, x in zip(self.backpack, environment.weights) if int(np.round(i)) == 1]
        sum_weight = np.sum(weight)
        value = [x for i, x in zip(self.backpack, environment.value) if int(np.round(i)) == 1]
        sum_value = np.sum(value)
        return sum_weight, sum_value


def check_population(population, environment):
    population_weights = []
    population_values = []

    for person in population:
        sum_weight, sum_value = person.fit(environment)
        population_values.append(sum_value)
        population_weights.append(sum_weight)
    return population_weights, population_values


def create_parents(population, number_population):
    parents = []
    for i in range(number_population * 7):
        new_person = random.choice(population)
        parents.append(copy.deepcopy(new_person))
    return parents


def uniform_crossover(parents, len_items):
    parents_crossover = []

    while len(parents) > 0:
        first_person = parents.pop(random.random_integers(0, len(parents) - 1))
        second_person = parents.pop(random.random_integers(0, len(parents) - 1))

        if random.rand() > 0.6:
            pattern = random.random_integers(0, 1, len_items)
            for i, cross in enumerate(pattern):
                if cross == 1:
                    r = first_person.backpack[i]
                    first_person.backpack[i] = second_person.backpack[i]
                    second_person.backpack[i] = r

            alfa = random.rand()

            new_first_deviation = alfa * np.array(first_person.standard_deviation) + (1 - alfa) * np.array(second_person.standard_deviation)
            new_second_deviation = alfa * np.array(second_person.standard_deviation) + (1 - alfa) * np.array(first_person.standard_deviation)
            first_person.standard_deviation = new_first_deviation
            second_person.standard_deviation = new_second_deviation

        parents_crossover.append(first_person)
        parents_crossover.append(second_person)

    return parents_crossover


def mutation(parents, step):

    for parent in parents:
        teta = random.normal()
        tau = np.sqrt(np.sqrt(step + 1) * 2) ** -1
        v = np.sqrt(1 + step * 2) ** -1
        deviation_vector = parent.standard_deviation
        new_deviation = []
        for deviation in deviation_vector:
            teta_i = random.normal()
            deviation = deviation * (np.e ** (tau * teta + v * teta_i))
            new_deviation.append(deviation)

        parent.standard_deviation = new_deviation
        backpack_vector = parent.backpack
        new_backpack = []

        for backpack, deviation in zip(backpack_vector, new_deviation):
            eta_i = random.normal()
            change =  eta_i * deviation
            #Mozna by zastosowac kumulacje mutacji
            if abs(change) >= 0.5:
                if backpack == 1:
                    backpack = 0
                else:
                    backpack = 1
            new_backpack.append(backpack)
        parent.backpack = new_backpack
    return parents


def accumulation_mutation(parents, step):

    for parent in parents:
        teta = random.normal()
        tau = np.sqrt(np.sqrt(step + 1) * 2) ** -1
        v = np.sqrt(1 + step * 2) ** -1
        deviation_vector = parent.standard_deviation
        new_deviation = []
        for deviation in deviation_vector:
            teta_i = random.normal()
            deviation = deviation * (np.e ** (tau * teta + v * teta_i))
            new_deviation.append(deviation)

        parent.standard_deviation = new_deviation
        backpack_vector = parent.backpack
        new_backpack = []

        for backpack, deviation in zip(backpack_vector, new_deviation):
            eta_i = random.normal()
            change =  eta_i * deviation
            #Mozna by zastosowac kumulacje mutacji
            backpack += change
            new_backpack.append(backpack)
        parent.backpack = new_backpack
    return parents


def get_best(population, environment, number_population):
    population_weights, population_values = check_population(population, environment)

    well_weight = []
    for person, weight, value in zip(population, population_weights, population_values):
        if weight <= environment.capacity:
            well_weight.append((person, value, weight))

    if len(well_weight) == 0:
        for person, weight, value in zip(population, population_weights, population_values):
            well_weight.append((person, value, weight))
        new_population = sorted(well_weight, key=lambda person: person[2])
        new_population, values, weights = zip(*new_population)
        new_population = list(new_population)

        if len(new_population) > number_population:
            avaregae_values = np.sum(values[:number_population]) / number_population
            avaregae_weight = np.sum(weights[:number_population]) / number_population
            std_values = np.std(values)
            return new_population[:number_population], avaregae_values, avaregae_weight, std_values

    new_population = sorted(well_weight, key=lambda person: person[1])
    new_population, values, weights = zip(*new_population)
    new_population = list(new_population)

    if len(new_population) > number_population:
        avaregae_values = np.sum(values[-number_population:]) / number_population
        avaregae_weight = np.sum(weights[-number_population:]) / number_population
        std_values = np.std(values)

        return new_population[-number_population:], avaregae_values, avaregae_weight, std_values

    std_values = np.std(values)
    return new_population, np.sum(values) / len(values), np.sum(weights) / len(weights), std_values


if __name__ == '__main__':
    len_items = 200
    backpack_capacity = 450
    # population = [Species(len_items) for i in range(number_population)]
    environment = Environment(len_items, backpack_capacity)
    FOLDER = '450_gen_30'
    plt.figure(1)
    number_population = 30
    STEP_NUMBER = 60
    start = time.time()
    number_generation = 5

    for step_gen in range(number_generation):

        population = [Species(len_items) for i in range(number_population)]

        average_value = []
        average_weights = []
        std_values = []
        for i in range(STEP_NUMBER):
            parents = create_parents(population, number_population)
            parents_crossover = uniform_crossover(parents, len_items)
            #Zamiast len_items mozna zastosowac krok algorytmu dzieki temu algorytm bedzie zbiegal
            parents_mutation = accumulation_mutation(parents_crossover, len_items)
            population = population + parents_mutation
            population = get_best(population, environment, number_population)
            average_value.append(population[1])
            average_weights.append(population[2])
            std_values.append(population[3])

            population = population[0]
            print('\r', end='')
            print('Generation: {0} / {1}'.format(step_gen, number_generation), end='\t', flush=True)

            print('Step: {0} / {1}'.format(i, STEP_NUMBER), end='', flush=True)
        sub = plt.subplot(211)

        plt.errorbar(range(len(average_value)),average_value,  yerr=std_values)
        plt.ylabel('wartosc plecaka')
        plt.subplot(212)
        plt.plot(average_weights)
        plt.ylabel('waga plecaka')
    end = time.time()
    print('\n czas obliczeń %f' % (end - start))

    plt.savefig(FOLDER + "/100mut")
    plt.close()
    best_object = population[0]

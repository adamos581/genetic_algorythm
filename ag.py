import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import copy
from deap import tools


def draw_graph(graph, labels,
               node_size=1600, node_color='pink', node_alpha=0.3,
               node_text_size=12,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    g = nx.Graph()
    for edge in graph:
        g.add_edge(edge[0], edge[1])
    graph_pos = nx.shell_layout(g)
    nx.draw_networkx_nodes(g, graph_pos, node_size=node_size,
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(g, graph_pos, width=edge_tickness,
                           alpha=edge_alpha, edge_color=edge_color)
    nx.draw_networkx_labels(g, graph_pos, font_size=node_text_size,
                            font_family=text_font)
    edge_labels = dict(zip(graph, labels))
    nx.draw_networkx_edge_labels(g, graph_pos, edge_labels=edge_labels,
                                 label_pos=edge_text_pos)
    plt.show()


def generate_graph(number_of_nodes):
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                'P', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z', 'AB', 'AA', 'AAA', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AS', 'AQ',
                'PP', 'PS', 'PR', 'KL', 'GH', 'AZ', 'SX', 'SSS','KLS', 'CVS']
    costs = []
    times = []
    connections = []
    connections_numbers = []
    distances = []
    if number_of_nodes > len(alphabet):
        number = len(alphabet)
    else:
        number = number_of_nodes
    iterations = 1000
    for i in range(0, iterations):
        flag = 0
        x = random.randint(0, number)
        y = random.randint(0, number)
        for j in range(0, len(connections)):
            if connections[j] == (alphabet[x], alphabet[y]) or connections[j] == (alphabet[y], alphabet[x]) or x == y:
                flag = 1
        if flag == 0:
            connections.append((alphabet[x], alphabet[y]))
            connections_numbers.append((x, y))
            cost = random.randint(0, 10)
            time = random.randint(0, 10)
            distance = random.randint(0, 10)
            costs.append(cost)
            times.append(time)
            distances.append(distance)
    return connections, connections_numbers, costs, times, distances


class Object:
    def __init__(self, number_of_nodes):
        self.number = number_of_nodes
        self.alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                         'P', 'R', 'S', 'T', 'U', 'W', 'X', 'Y', 'Z']
        self.standard_deviation = np.random.normal(loc=0, scale=1, size=number_of_nodes)
        self.parameters = []
        self.adaptation = 0
        self.flag = 0
        self.capabilities = list(np.arange(0, self.number))
        for i in range(0, number_of_nodes):
            self.rand = np.random.randint(0, self.number)
            self.parameters.append(self.capabilities.pop(self.rand))
            self.number -= 1
        print(self.parameters)
        # print(self.standard_deviation)


def generate_population(number_of_objects, number_of_nodes):
    basic_generation = []
    for i in range(0, number_of_objects):
        basic_generation.append(Object(number_of_nodes))

    return basic_generation


def draw_posterity(basic_generation):
    new_generation = []
    for i in range(0, len(basic_generation)*7):
        object = np.random.choice(basic_generation)
        new_generation.append(copy.deepcopy(object))

    return new_generation


def crossover_pmx(object1, object2):
    tools.cxPartialyMatched(object1, object2)
    return object1, object2


def crossover_arithmetic(object1, object2):
    alfa = random.random()
    object1return = alfa * object1 + ((1 - alfa) * object2)
    object2return = alfa * object2 + ((1 - alfa) * object1)

    return object1return, object2return


def iteration_of_crossover(generation):
    new_generation = []
    for i in range(0, len(generation), 2):
        probability_of_crossover = random.random()
        object1 = generation.pop(random.randint(0, len(generation) - 1))
        object2 = generation.pop(random.randint(0, len(generation) - 1))
        if probability_of_crossover > 0.6:
            object1.parameters, object2.parameters = crossover_pmx(object1.parameters, object2.parameters)
            object1.standard_deviation, object2.standard_deviation = crossover_arithmetic(object1.standard_deviation,
                                                                                          object2.standard_deviation)
        new_generation.append(object1)
        new_generation.append(object2)

    return new_generation


def iteration_of_mutation(generation, number_of_iteration, number_of_nodes):
    tau = (np.sqrt(2*np.sqrt(number_of_nodes + 1)))**(-1)
    vau = (np.sqrt(2*number_of_nodes + 1))**(-1)
    for object in generation:
        zeta = np.random.normal()
        for i in range(0, len(object.parameters)):
            zeta_item = np.random.normal()
            epsilon_item = np.random.normal()
            object.standard_deviation[i] = object.standard_deviation[i] * np.exp(tau*zeta + vau*zeta_item)

            if abs(object.standard_deviation[i] * epsilon_item) > 0.8:
                r2 = random.randint(0, len(object.parameters) - 1)
                object.parameters[i], object.parameters[r2] = object.parameters[r2], object.parameters[i]


    return generation


def function_of_adaptation(generation, graph_number, costs, times, distances, number_of_object):
    parents = []
    parents_final = []
    for object in generation[:]:
        flag = 0
        for i in range(0, len(object.parameters) - 1):
            find = False

            for j in range(0, len(graph_number)):
                if (object.parameters[i], object.parameters[i + 1]) ==  graph_number[j] \
                        or (object.parameters[i + 1], object.parameters[i]) == graph_number[j]:
                    object.adaptation += costs[j] + times[j] + distances[j]
                    find = True
            if not find:
                flag += 1

        if flag == 0:
            parents.append(object)
        else:
            object.flag = flag
    print(len(parents))
    if len(parents) > 0:
        parents = sorted(parents, key= lambda parent: parent.adaptation)
        if len(parents) < number_of_object:
            parents_final = parents[:]
        else:
            parents_final = parents[:number_of_object]

    generation = [obj for obj in generation if obj.flag > 0]
    generation = sorted(generation, key=lambda parent: parent.flag)
    parents_final = parents_final + generation[:number_of_object - len(parents_final)]
    # parents_final.append(min)

    lack_of_path = []
    cost = []
    for parent in parents_final:
        cost.append(parent.adaptation)
        lack_of_path.append(parent.flag)
        parent.adaptation = 0
        parent.flag = 0

    print(lack_of_path)
    return parents_final, np.mean(cost), np.mean(lack_of_path)


if __name__ == '__main__':
    random.seed(time.time())
    number_of_nodes = 45
    number_of_object = 30
    number_of_iterations = 56
    graph, graph_numbers, costs, times, distances = generate_graph(number_of_nodes)
    draw_graph(graph, costs)

    basic_generation = generate_population(number_of_object, number_of_nodes)
    cost_avg = []
    lack_of_path_avg = []
    for iterator in range(0, number_of_iterations):
        new_generation = draw_posterity(basic_generation)
        generation_after_crossover = iteration_of_crossover(new_generation)
        generation_after_mutation = iteration_of_mutation(generation_after_crossover, iterator, number_of_nodes)
        basic_generation, cost, lack_of_path = function_of_adaptation(generation_after_mutation, graph_numbers, costs, times, distances, number_of_object)
        cost_avg.append(cost)
        lack_of_path_avg.append(lack_of_path)

    plt.subplot(211 )
    plt.plot(cost_avg)
    plt.ylabel('Średnia wartośc funkcji przystosowania')
    plt.subplot(212)
    plt.plot(lack_of_path_avg)
    plt.ylabel('Średnia ilośc ścieżek nie do przejścia')


    plt.show()
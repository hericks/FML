import genetic_learning_interface as gli
import numpy as np
import pickle

from deap import base, creator, tools, gp, algorithms
from inspect import isclass

import random
import operator
import os


# Settings
POPULATION_SIZE = 20
NUM_COMPLEX_FEATURES = 4
NUM_GENERATIONS = 50


# create set
primitive_set = gp.PrimitiveSet("complex_feature", 9)


# fill with operations
def if_then_else(cond, output1, output2):
    return output1 if cond else output2


primitive_set.addPrimitive(if_then_else, 3)
primitive_set.addPrimitive(operator.not_, 1)
primitive_set.addPrimitive(operator.and_, 2)
primitive_set.addPrimitive(operator.or_, 2)

creator.create("MaxReward", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.MaxReward, primitive_set=primitive_set)

toolbox = base.Toolbox()
toolbox.register("combined_feature", lambda: gp.PrimitiveTree(gp.genGrow(primitive_set, min_=0, max_=2)))
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.combined_feature, n=NUM_COMPLEX_FEATURES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(ind):
    with open("genes.pt", "wb") as file:
        pickle.dump(ind, file)

    for tree in ind:
        print(gp.PrimitiveTree(tree))

    os.system(f"python main.py play --agents genetic_individual --train 1 --n-rounds 8 --no-gui >/dev/null")

    ret = None
    with open("latest_history.pt", "rb") as file:
        history = pickle.load(file)
        ret = np.mean(history['cumulative_reward'][-5:])

    print(ret)
    return ret,


def multi_mutate(ind):
    target_index = random.randrange(len(ind))
    mutated = customMutNodeReplacement(ind[target_index], ind.primitive_set)
    ind[target_index] = mutated[0]
    return ind,


def switch_feature(tree, primitive_set):
    feature_indices = [i for i in range(len(tree)) if tree[i].arity == 0]
    index = random.choice(feature_indices)
    node = tree[index]
    term = random.choice(primitive_set.terminals[node.ret])
    if isclass(term):
        term = term()
    tree[index] = term
    return tree,


def switch_operand(tree, primitive_set):
    operand_indices = [i for i in range(len(tree)) if (tree[i].arity != 0 and tree[i].arity != 3)]

    if len(operand_indices) == 0:
        return tree,

    index = random.choice(operand_indices)
    node = tree[index]

    if node.arity == 1:
        del tree[index]
    else:
        prims = [p for p in primitive_set.primitives[node.ret] if p.args == node.args]
        tree[index] = random.choice(prims)

    return tree,


def custom_mutate(ind):
    target_index = random.randrange(len(ind))
    mutation_operations = {
        0: switch_feature,
        1: switch_operand,
        2: lambda t, ps: gp.mutShrink(t)
    }
    operation_index = random.choices(range(3))[0]
    print(operation_index)
    mutated = mutation_operations[operation_index](ind[target_index], ind.primitive_set)
    ind[target_index] = mutated[0]
    return ind,


def custom_mate(ind1, ind2):
    target_index1 = random.randrange(len(ind1))
    target_index2 = random.randrange(len(ind2))
    node1 = ind1[target_index1]
    node2 = ind2[target_index2]
    mutated1, mutated2 = gp.cxOnePointLeafBiased(node1, node2, termpb=0.4)
    ind1[target_index1] = mutated1
    ind2[target_index2] = mutated2
    return ind1, ind2


toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mutate", custom_mutate)
toolbox.register("mate", custom_mate)
toolbox.register("evaluate", evaluate)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min_fitness", np.min)
stats.register("mean_fitness", np.mean)
stats.register("max_fitness", np.max)
stats.register("std_fitness", np.std)

population = toolbox.population(n=POPULATION_SIZE)
algorithms.eaSimple(population, toolbox, cxpb=0.2, mutpb=0.2, ngen=NUM_GENERATIONS, stats=stats, verbose=True)
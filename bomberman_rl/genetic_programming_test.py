from deap.gp import *

import operator


def if_then_else(cond, output1, output2):
    return output1 if cond else output2


# create set
primitive_set = PrimitiveSet("complex_feature", 9)

# fill with operations
primitive_set.addPrimitive(if_then_else, 3)
primitive_set.addPrimitive(operator.not_, 1)
primitive_set.addPrimitive(operator.and_, 2)
primitive_set.addPrimitive(operator.or_, 2)

expr = genGrow(primitive_set, min_=2, max_=3)
tree = PrimitiveTree(expr)


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


def custom_mutate(tree, primitive_set):
    mutation_operations = {
        0: switch_feature,
        1: switch_operand,
        2: lambda t, ps: mutShrink(t)
    }
    operation_index = random.choices(range(4))[0]
    return mutation_operations[operation_index](tree, primitive_set)

print(tree)
custom_mutate(tree, primitive_set)
print(tree)
# print(len(tree))



# tree_func = compile(tree, primitive_set)
# print(tree_func(False, False, True, True))


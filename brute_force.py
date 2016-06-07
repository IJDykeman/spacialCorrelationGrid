import numpy as np
from itertools import product
import math
from itertools import permutations
import matplotlib.pyplot as plt
import random

n = 8

distances = {}
for i, j in product(range(n), range(n)):
    for x,y in product(range(n), range(n)):
        distances[(i, j, x, y)] = 1.0*(math.sqrt((i-j)**2 + (x-y)**2))

def fitness(grid, draw = False):
    grid = grid.reshape([n,n])
    assert(grid.shape[0] == grid.shape[1])
    

    euclidian_distances = np.zeros(n**4)
    absolute_distances = np.zeros(n**4)
    for index, item in enumerate(distances):
        i, j, x, y = item
        euclidian_distances[index] = distances[item]
        absolute_distances[index] = abs(grid[i][j] - grid[x][y])
    # print grid
    if (draw):
        plt.imshow(grid, interpolation = 'nearest', cmap = 'gray')
        plt.show()
        plt.scatter(euclidian_distances, absolute_distances)
        plt.show()
    return np.sum(euclidian_distances * absolute_distances)

def flip(p):
    return True if random.random() < p else False

def breed(seq1, seq2):
    bits = np.zeros(len(seq1))
    child = np.zeros(len(seq1))

    mother = flip(.5)
    for i in range(len(seq1)):
        if flip(.1):
            mother = not mother
        bits[i] = int(mother)
    mother_contribution = seq1[bits == 1]
    father_contribution = [item for item in seq2 if item not in mother_contribution]
    father_contributions = 0
    for i, bit in enumerate(bits):
        if(bit):
            child[i] = seq1[i]
        else:
            child[i] = father_contribution[father_contributions]
            father_contributions += 1
    return child.astype(np.int32)


def mutate(seq):
    input_seq = seq.copy()
    ix1 = random.randint(0,len(input_seq) - 1)
    ix2 = random.randint(0,len(input_seq) - 1)
    input_seq[ix1], input_seq[ix2] = input_seq[ix2], input_seq[ix1]
    return input_seq


# pop_size = 100
# pop = [np.random.permutation(np.array(range(1,n**2+1))) for i in range(pop_size)]

# for gen in range(500):
#     def unique_rows(a):
#         a = np.ascontiguousarray(a)
#         unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
#         return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
#     # pop = list(unique_rows(np.array(pop)))
#     pop = sorted(pop, key = fitness)
#     if(gen%100 == 1):
#         pop.extend([mutate(pop[-1]) for i in range(100)])

#     print gen, fitness(pop[-1]) 

#     old_best = pop[pop_size*4/5:]
#     pop = pop[len(pop)/2:]
#     pop.extend([np.random.permutation(np.array(range(1,n**2+1))) for i in range(pop_size/8)])
#     new_children = []
#     for i in range((pop_size - len(old_best))/2):
#         new_children.append(breed(random.choice(pop),random.choice(pop)))
#         # print "breed"
#     for i in range(pop_size - len(new_children)):
#         # new_children.append(mutate(random.choice(pop)))
#         new_children.append(mutate(random.choice(old_best)))

#         # print "ayy"
#     pop.extend(new_children)

#     # print pop

# pop = sorted(pop, key = fitness)
# print pop[-1].reshape([n,n])

# fitness(pop[-1], draw = True)

candidate = np.random.permutation(np.array(range(1,n**2+1)))
for i in range(10000):
    if i%100 == 0:
        print i, fitness(candidate)
    news = [mutate(candidate) for _ in range(10)]
    new = sorted(news, key = fitness)[-1]
    if(fitness(new) > fitness(candidate)):
        candidate = new
print candidate
print fitness(candidate, draw = True)

diagonal = np.array(
[[1, 3,  6,  10, 15],
 [2, 5,  9, 14, 19],
 [ 4,  8, 13, 18, 22],
 [ 7,  12, 17, 21,  24],
 [ 11, 16, 20, 23, 25,]]
 )

print fitness(diagonal, draw = True)

# def go():
    # best = 0
    # best_set = set([])
    # for item in permutations(np.array(range(1,10))):

    #   fit = fitness(np.array(item).reshape([3,3]))
    #   if fit > best:
    #       best_set = set([])
    #       print "=======new high score======="
    #       print np.array(item).reshape([3,3])
    #       best = fit
    #   if fit == best:
    #       best_set.add(item)

    # for item in best_set:
    #   print np.array(item).reshape([3,3])
    #   plt.imshow(np.array(item).reshape([3,3]))
    #   plt.show
    # print len(best_set)


# go()
# grid = np.array([[5, 3, 1],
#  [7, 4, 8],
#  [9, 2, 6]])
# fitness(grid)
# plt.imshow(grid, interpolation = 'nearest', cmap='gray')
# plt.show()


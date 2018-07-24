from graph import Graph
from group import Louvain, groups2inv_index, pure_override_nodes
from sample import sample
from train import Optimizer
from descend import *
from score import multi_class_classification
import numpy as np
import pandas as pd
import train
from numpy.linalg import norm
import time

# Arguments
##############################
VERBOSE = 1
##############################
K_SIZE = 607
DIMENSION = 200
##############################
RANDOM_GROUPING = True
SAMPLE_METHOD = 'set_cover_undir'
##############################
LAMBDA = 0.8
ETA = 0.1
MAX_ITER = 150
EPSILON = 1e-4
DESCEND_METHOD = inverse_descending
##############################
CG_MAX_ITER = CG_MAX_ITER  # imported from descending.py
CG_EPSILON = CG_EPSILON    # imported from descending.py
##############################
DATASET = 'wiki'
DATADIR = 'data\\'
##############################

pt0 = time.time()

pt = time.time()
net = Graph(DATADIR + DATASET + '\\links.txt', typ='dir')
print('READ TIME: %.2f' % (time.time() - pt))

pt = time.time()
grouping_model = Louvain(net)
groups = grouping_model.execute(rand=RANDOM_GROUPING)
print('GROUP TIME: %.2f' % (time.time() - pt))

group_sizes = [len(t) for t in groups]
vid2originalGroup_index = groups2inv_index(groups, net.nVertices)
vid2originalGroupSIze_index = [group_sizes[t - 1] for t in vid2originalGroup_index]

pt = time.time()
k_set = sample(net, k=K_SIZE, method=SAMPLE_METHOD, vertex_group_sizes=vid2originalGroupSIze_index)
print('SAMPLE TIME: %.2f' % (time.time() - pt))

inv_index = groups2inv_index(groups, net.nVertices, k_set)
pure_override_nodes(groups, inv_index)
groups = [k_set] + groups

pt = time.time()
model = Optimizer(net, groups, dim=DIMENSION,
                  lam=LAMBDA, eta=ETA, max_iter=MAX_ITER, epsilon=EPSILON,
                  cg_max_iter=CG_MAX_ITER, cg_eps=CG_EPSILON,
                  descending_method=DESCEND_METHOD, verbose=VERBOSE,
                  sample_strategy=SAMPLE_METHOD)
print('INITIAL OPTIMIZER TIME (SVD): %.2f' % (time.time() - pt))

multi_class_classification(optimizer=model, sample_filename=DATADIR + DATASET + '\\group.txt')

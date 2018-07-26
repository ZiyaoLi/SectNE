from graph import Graph
from group import Louvain, groups2inv_index, pure_override_nodes
from sample import sample
from train import Optimizer
from descend import *
from score import multi_class_classification
import time
import sys

# Arguments
##############################
VERBOSE = True
##############################
K_SIZE = 200
DIMENSION = 128
##############################
RANDOM_GROUPING = True
MERGE = (200, 4000)
SAMPLE_METHOD = 'set_cover_undir'
##############################
LAMBDA = 0.4
ETA = 0.1
MAX_ITER = 30
EPSILON = 1e-4
DESCEND_METHOD = inverse_descending
##############################
CG_MAX_ITER = CG_MAX_ITER  # imported from descending.py
CG_EPSILON = CG_EPSILON    # imported from descending.py
##############################
DATASET = 'wiki'
TYPE = 'undir'
DATADIR = 'data\\'
##############################

f = open('results\\' + DATASET + '_output_bigk.log', 'w')
# sys.stdout = f

pt0 = time.time()

pt = time.time()
net = Graph(DATADIR + DATASET + '\\links.txt', typ=TYPE, verbose=VERBOSE)
print('READ TIME: %.2f' % (time.time() - pt))

pt = time.time()
grouping_model = Louvain(net, verbose=VERBOSE)
groups = grouping_model.execute(merge=MERGE)
print('GROUP TIME: %.2f' % (time.time() - pt))

group_sizes = [len(t) for t in groups]
vid2originalGroup_index = groups2inv_index(groups, net.nVertices)
vid2originalGroupSize_index = [group_sizes[t - 1] for t in vid2originalGroup_index]

pt = time.time()
k_set = sample(net, k=K_SIZE, method=SAMPLE_METHOD, vertex_group_sizes=vid2originalGroupSize_index)
print('SAMPLE TIME: %.2f' % (time.time() - pt))

inv_index = groups2inv_index(groups, net.nVertices, k_set)
pure_override_nodes(groups, inv_index)
groups = [k_set] + groups

for MAX_ITER in (1, 2, 3, 4):
    pt = time.time()
    model = Optimizer(net, groups, dim=DIMENSION,
                      lam=LAMBDA, eta=ETA, max_iter=MAX_ITER, epsilon=EPSILON,
                      cg_max_iter=CG_MAX_ITER, cg_eps=CG_EPSILON,
                      descending_method=DESCEND_METHOD, verbose=VERBOSE,
                      sample_strategy=SAMPLE_METHOD)
    print('INITIAL OPTIMIZER TIME (SVD): %.2f' % (time.time() - pt))

    multi_class_classification(optimizer=model, sample_filename=DATADIR + DATASET + '\\group.txt', cv=True,
                               cross_val_fold=[10])
    # multi_class_classification(optimizer=model, sample_filename=DATADIR + DATASET + '\\group.txt', cv=False)

f.close()

from graph import Graph
from sample import sample
from train import Optimizer
from group import Louvain, groups2inv_index, pure_override_nodes
import numpy as np
import pandas as pd
import train
from numpy.linalg import norm
import time

K_SIZE = 200
DIMENSION = 100
VERBOSE = 1
GROUP_IGNORE = 1
SAMPLE_METHOD = 'deg_deter'
RANDOM_GROUPING = True

DATASET = 'wiki'
DATADIR = 'data\\'
ABS_DELTA_MATRIX_TO_FILE = True
UNIQUE_NAME = 'cleared-null-entries'
METHOD_MAP = {
    'deg_prob': 'dp',
    'deg^2_prob': 'd2p',
    'deg_deter': 'dd',
    'deg|group_prob': 'dgp',
    'deg|group_deter': 'dgd',
    'deg^2|group_prob': 'd2gp',
    'deg^2|group_deter': 'd2gd',
    'uniform': 'uni'
}
FILE_NAME = 'results\\' + '_'.join([
    DATASET,
    'k=%s' % K_SIZE,
    'd=%s' % DIMENSION,
    'sample=%s' % METHOD_MAP[SAMPLE_METHOD],
    'lambda=%.2f' % train.LAMBDA,
    'eta=%.2f' % train.ETA,
    'max-iter=%04d' % train.MAX_ITER,
    '%s%.3f' % (UNIQUE_NAME, np.random.rand())
]) + '.csv'

pt = time.time()
net = Graph(DATADIR + DATASET + '\\links.txt', typ='dir')
print('READ TIME: %.2f' % (time.time() - pt))

pt = time.time()
grouping_model = Louvain(net)
groups = grouping_model.execute(rand=RANDOM_GROUPING)
print('GROUP TIME: %.2f' % (time.time() - pt))

group_sizes = [len(t) for t in groups]
print(pd.value_counts(group_sizes))
inv_index_original = groups2inv_index(groups, net.nVertices)
sizes_index = [group_sizes[t - 1] for t in inv_index_original]

pt = time.time()
# k_set = sample(net, k=K_SIZE, method='deg_deter')
k_set = sample(net, k=K_SIZE, method=SAMPLE_METHOD, vertex_group_sizes=sizes_index)
print('SAMPLE TIME: %.2f' % (time.time() - pt))

inv_index = groups2inv_index(groups, net.nVertices, k_set)
pure_override_nodes(groups, inv_index)
groups = [k_set] + groups

pt = time.time()
model = Optimizer(net, groups, dim=DIMENSION)
print('INITIAL OPTIMIZER TIME (SVD): %.2f' % (time.time() - pt))

vecs_w = []
vecs_c = []

all_idx = []
for t in range(len(groups)):
    pt = time.time()
    if len(groups[t]) <= GROUP_IGNORE:
        continue
    print("%d / %d, n_vertices = %d..., accumul_vertex = %d "
          % (t + 1, len(groups), len(groups[t]), len(all_idx)))
    w, c = model.train(t, verbose=VERBOSE)
    print('GROUP TRAINING TIME: %.2f' % (time.time() - pt))
    vecs_w.append(w)
    vecs_c.append(c)
    all_idx += groups[t]

# concatenate all the derived vectors together
ws = np.concatenate(vecs_w, 1)
cs = np.concatenate(vecs_c, 1)

# reconstructing matrix over the order of sampled vertices
reconstruct = ws.T @ cs
original = net.calc_matrix(all_idx, all_idx)

# evaluate the reconstruction performance
delta = original - reconstruct
abs_delta = abs(delta) * (original != 0)
if ABS_DELTA_MATRIX_TO_FILE:
    np.savetxt(FILE_NAME, abs_delta, '%.6e', ',', '\n')
t = norm(abs_delta, 'fro')
tt = norm(original, 'fro')
print("Original - %.4f, delta - %.4f, percentage - %.4f"
      % (tt, t, t / tt))

# a SVD implementation to exam how good is the result
u, d, v = np.linalg.svd(original)
w_svd = (u[:, :DIMENSION] * np.sqrt(d[:DIMENSION])).T
c_svd = (v.T[:, :DIMENSION] * np.sqrt(d[:DIMENSION])).T
reconstruct_svd = w_svd.T @ c_svd
delta_svd = (original - reconstruct_svd) * (original != 0)
t_svd = norm(delta_svd, 'fro')
print("Original - %.4f, delta - %.4f, percentage - %.4f"
      % (tt, t_svd, t_svd / tt))
pass

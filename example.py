from graph import Graph
from sample import sample
from train import Optimizer
from group import Louvain, groups2inv_index, pure_override_nodes
import numpy as np
import pandas as pd
from numpy.linalg import norm
import time
from multiprocessing import Pool

###################
K_SIZE = 1000
DIMENSION = 128
LAMBDA = 10
ETA = 0.1
MAX_ITER = 5
###################
MERGE = (2000, 8000)
SAMPLE_METHOD = 'set_cover_undir'
RANDOM_GROUPING = True
ORDER = 1
WITHDIAG = True
###################
DATASET = 'flickr'
DATADIR = 'data\\'
FILE_NAME = '_'.join([
    DATASET,
    'k=%s' % K_SIZE,
    'd=%s' % DIMENSION,
    'sample=%s' % SAMPLE_METHOD,
    'lambda=%.2f' % LAMBDA,
    'eta=%.2f' % ETA,
    'max-iter=%02d' % MAX_ITER
]) + '.vec'

ppt = time.time()

f = open(DATADIR + DATASET + '\\' + FILE_NAME, 'w')

pt = time.time()
net = Graph(DATADIR + DATASET + '\\links.txt', typ='dir', order=ORDER, withdiag=WITHDIAG)
print('READ TIME: %.2f' % (time.time() - pt))

f.write('%d %d %d\n' % (net.nVertices, net.nEdges, DIMENSION))

pt = time.time()
grouping_model = Louvain(net, rand=RANDOM_GROUPING)
groups = grouping_model.execute(merge=MERGE)
print('GROUP TIME: %.2f' % (time.time() - pt))

inv_index_original = groups2inv_index(groups, net.nVertices)

pt = time.time()
k_set = sample(net, k=K_SIZE, method=SAMPLE_METHOD)
print('SAMPLE TIME: %.2f' % (time.time() - pt))

inv_index = groups2inv_index(groups, net.nVertices, k_set)
pure_override_nodes(groups, inv_index)
groups = [k_set] + groups

pt = time.time()
model = Optimizer(net, groups, dim=DIMENSION, lam=LAMBDA, eta=ETA, max_iter=MAX_ITER,
                  sample_strategy=SAMPLE_METHOD, verbose=True)
print('INITIAL OPTIMIZER TIME (SVD): %.2f' % (time.time() - pt))

with Pool(processes=4) as pool:
    grouped_embeddings = pool.map()


for newVid, vid in net.newVid2vid_mapping.items():
    if not newVid % 10000:
        print('Now %7d vertices have been embedded.' % newVid)
    f.write('%d ' % vid)
    vec = np.array(model.embed(newVid, verbose=True))
    vec_str = ' '.join([str(t) for t in vec[:, 0]])
    f.write(vec_str)
    f.write('\n')

print('TOTAL TIME: %.2f' % (time.time() - ppt))

f.close()

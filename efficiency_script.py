from graph import Graph
from sample import sample
from train import Optimizer, BranchOptimizer
from group import Louvain, groups2inv_index, pure_override_nodes
import time
from multiprocessing import Pool
import numpy as np

###################
K_SIZE = 200
DIMENSION = 128
LAMBDA = 10
ETA = 0.1
MAX_ITER = 5
###################
OUTPUT_VECTORS = True
MERGE = (0, 8000)
SAMPLE_METHOD = 'set_cover_undir'
RANDOM_GROUPING = True
ORDER = 2
WITHDIAG = False
VERBOSE = True
WORKERS = 2
###################
DATASET = 'wiki'
DATADIR = 'data\\'
FILE_NAME = '_'.join([
    DATASET,
    'k=%s' % K_SIZE,
    'd=%s' % DIMENSION,
    'sample=%s' % SAMPLE_METHOD,
    'lambda=%.2f' % LAMBDA,
    'eta=%.2f' % ETA,
    'max-iter=%02d' % MAX_ITER,
    'paralleled'
]) + '.vec'


def WrapTrain(arg):
    return arg.train()

if __name__ == '__main__':
    pt = time.time()
    net = Graph(DATADIR + DATASET + '\\links.txt', typ='dir', order=ORDER,
                withdiag=WITHDIAG, verbose=VERBOSE)

    read_time = time.time() - pt
    print('READ TIME: %.2f' % read_time)

    pt = time.time()
    grouping_model = Louvain(net, rand=RANDOM_GROUPING, verbose=VERBOSE)
    groups = grouping_model.execute(merge=MERGE)
    group_time = time.time() - pt
    print('GROUP TIME: %.2f' % group_time)

    inv_index_original = groups2inv_index(groups, net.nVertices)

    pt = time.time()
    k_set = sample(net, k=K_SIZE, method=SAMPLE_METHOD)
    sample_time = time.time() - pt
    print('SAMPLE TIME: %.2f' % sample_time)

    inv_index = groups2inv_index(groups, net.nVertices, k_set)
    pure_override_nodes(groups, inv_index)
    groups = [k_set] + groups

    pt = time.time()
    optimizer = Optimizer(net, groups, dim=DIMENSION, lam=LAMBDA, eta=ETA,
                          max_iter=MAX_ITER, sample_strategy=SAMPLE_METHOD,
                          verbose=VERBOSE)
    svd_time = time.time() - pt
    print('INITIAL OPTIMIZER TIME (SVD): %.2f' % svd_time)

    pt = time.time()
    branches = []
    for t in range(len(groups)):
        branches.append(BranchOptimizer(optimizer, t, verbose=VERBOSE))
    prep_time = time.time() - pt
    print('PROCESS PREPARATION TIME: %.2f' % prep_time)

    pt = time.time()
    with Pool(processes=WORKERS) as pool:
        grouped_embeddings = pool.map(WrapTrain, branches)
    embed_time = time.time() - pt
    print('OPTIMIZING TIME: %.2f' % embed_time)

    total_time = read_time + group_time + sample_time + \
        svd_time + prep_time + embed_time
    print('TOTAL TIME: %.2f' % total_time)

    if OUTPUT_VECTORS:
        f = open(DATADIR + DATASET + '\\' + FILE_NAME, 'w')
        f.write('%d %d %d\n' % (net.nVertices, net.nEdges, DIMENSION))
        for i, group in enumerate(groups):
            embeddings = grouped_embeddings[i][1]
            for j, newVid in enumerate(group):
                vid = net.newVid2vid_mapping[newVid]
                f.write('%d ' % vid)
                vec = np.array(embeddings[:, j].T)
                vec_str = ' '.join([str(t) for t in vec])
                f.write(vec_str)
                f.write('\n')
        f.close()

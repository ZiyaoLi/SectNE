from graph import Graph
from sample import sample
from train import Optimizer, BranchOptimizer
from group import Louvain, groups2inv_index, pure_override_nodes
import time
import os
import sys
from multiprocessing import Pool
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

###################
# K_SIZE = 200
# DIMENSION = 128
# LAMBDA = 10
# ETA = 0.1
# MAX_ITER = 5
# ###################
# OUTPUT_VECTORS = True
# MERGE = (0, 8000)
# SAMPLE_METHOD = 'set_cover_undir'
# RANDOM_GROUPING = True
# ORDER = 2
# WITHDIAG = False
# VERBOSE = True
# WORKERS = 2
# ###################
# DATASET = 'flickr'
# DATADIR = 'data\\'
# FILE_NAME = '_'.join([
#     DATASET,
#     'k=%s' % K_SIZE,
#     'd=%s' % DIMENSION,
#     'sample=%s' % SAMPLE_METHOD,
#     'lambda=%.2f' % LAMBDA,
#     'eta=%.2f' % ETA,
#     'max-iter=%02d' % MAX_ITER,
#     'paralleled'
# ]) + '.vec'


def WrapTrain(arg):
    return arg.train()


def process(args):
    pt = time.time()
    WITHDIAG = True if args.order == 1 else False
    net = Graph(args.input, typ='dir', order=args.order,
                withdiag=WITHDIAG, verbose=(args.v > 1))
    read_time = time.time() - pt
    if args.v:
        print('READ TIME:\t%.2f' % read_time)

    pt = time.time()
    grouping_model = Louvain(net, rand=True, verbose=(args.v > 1))
    groups_original = grouping_model.execute(merge=(args.merge0, args.merge1))
    group_time = time.time() - pt
    if args.v:
        print('GROUP TIME:\t%.2f' % group_time)

    SAMPLE_METHOD = 'set_cover_undir'
    if args.sample == 1:
        SAMPLE_METHOD = 'deg_deter'
    elif args.sample == 2:
        SAMPLE_METHOD = 'deg_prob'
    elif args.sample == 3:
        SAMPLE_METHOD = 'deg^2_prob'
    elif args.sample == 4:
        SAMPLE_METHOD = 'uniform'

    for ksize in args.listksize:

        groups = groups_original.copy()

        pt = time.time()
        k_set = sample(net, k=ksize, method=SAMPLE_METHOD)
        sample_time = time.time() - pt
        if args.v:
            print('SAMPLE TIME:\t%.2f' % sample_time)

        inv_index = groups2inv_index(groups, net.nVertices, k_set)
        pure_override_nodes(groups, inv_index)
        groups = [k_set] + groups

        for dim in args.listdim:
            for lam in args.listlam:
                for eta in args.listeta:
                    for n_iter in args.listiter:
                        pt = time.time()
                        optimizer = Optimizer(net, groups, dim=dim, lam=lam, eta=eta,
                                              max_iter=n_iter, sample_strategy=SAMPLE_METHOD,
                                              verbose=(args.v > 1))
                        svd_time = time.time() - pt
                        if args.v:
                            print('INITIAL OPTIMIZER TIME (SVD):\t%.2f' % svd_time)

                        pt = time.time()
                        branches = []
                        for t in range(len(groups)):
                            branches.append(BranchOptimizer(optimizer, t, verbose=(args.v > 1)))
                        prep_time = time.time() - pt
                        if args.v:
                            print('PROCESS PREPARATION TIME:\t%.2f' % prep_time)

                        pt = time.time()
                        with Pool(processes=args.workers) as pool:
                            grouped_embeddings = pool.map(WrapTrain, branches)
                        embed_time = time.time() - pt
                        print('OPTIMIZING TIME:\t%.2f' % embed_time)

                        if args.v:
                            total_time = read_time + group_time + sample_time + \
                                         svd_time + prep_time + embed_time
                            print('TOTAL TIME:\t%.2f' % total_time)

                        if len(args.output) > 1:
                            filename = args.output + '_' + '_'.join(
                                ['ksize=%d' % ksize,
                                 'dim=%d' % dim,
                                 'lam=%.1f' % lam,
                                 'eta=%.1f' % eta,
                                 'iter=%d' % n_iter
                                ])
                            f = open(filename, 'w')
                            f.write('|V|=%d; |E|=%d; dim=%d\n' % (net.nVertices, net.nEdges, args.dim))
                            for i, group in enumerate(groups):
                                embeddings = grouped_embeddings[i][1]
                                for j, newVid in enumerate(group):
                                    vid = net.newVid2vid_mapping[newVid]
                                    f.write('%d ' % vid)
                                    vec = np.array(embeddings[:, j])
                                    vec_str = ' '.join([str(t) for t in vec[:, 0]])
                                    f.write(vec_str)
                                    f.write('\n')
                            f.close()


def list_handle(args):
    if args.listlam is not None:
        args.listlam = eval(args.listlam)
    else:
        args.listlam = (args.lam,)
    if args.listeta is not None:
        args.listeta = eval(args.listeta)
    else:
        args.listeta = (args.eta,)
    if args.listksize is not None:
        args.listksize = eval(args.listksize)
    else:
        args.listksize = (args.ksize,)
    if args.listdim is not None:
        args.listdim = eval(args.listdim)
    else:
        args.listdim = (args.dim,)
    if args.listiter is not None:
        args.listiter = eval(args.listiter)
    else:
        args.listiter = (args.iter,)

    process(args)


def main():
    parser = ArgumentParser(prog="SectNE",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--input', required=True,
                        help='Input edge-list file.')

    parser.add_argument('--output', required=True,
                        help='Output representation file.')

    parser.add_argument('--lam', default=1.0, type=float,
                        help="Lambda, global loss scalar.")

    parser.add_argument('--eta', default=1.0, type=float,
                        help='Eta, regularization scalar.')

    parser.add_argument('--iter', default=5, type=int,
                        help='Number of iterations.')

    parser.add_argument('--ksize', default=1000, type=int,
                        help='K, number of landmark nodes.')

    parser.add_argument('--dim', default=128, type=int,
                        help='Dimensionality of representations.')

    parser.add_argument('--merge0', default=2000, type=int,
                        help='Communities smaller than this size will be '
                             'randomly merged together.')

    parser.add_argument('--merge1', default=8000, type=int,
                        help='Communities larger than this size will be '
                             'separated to subsets of this size.')

    parser.add_argument('--order', default=1, type=int,
                        help='Order of the proximity matrix. '
                             '0-(I+A); 1-(A+A*A)')

    parser.add_argument('--sample-method', dest='sample', default=0, type=int,
                        help='Sample method: 0-GDS; 1-DD; 2-DP; 3-SDP; 4-UF.')

    parser.add_argument('--workers', default=os.cpu_count(), type=int,
                        help='Number of processes to run.')

    parser.add_argument('--v', default=1, type=int,
                        help='Verbose level. 0-None; 1-Limited; 2-Plenty.')

    parser.add_argument('--listlam', default=None,
                        help='Input a list of lambdas to see '
                             'parameter sensitivity')

    parser.add_argument('--listeta', default=None,
                        help='Input a list of etas to see '
                             'parameter sensitivity')

    parser.add_argument('--listksize', default=None,
                        help='Input a list of ksizes to see '
                             'parameter sensitivity')

    parser.add_argument('--listdim', default=None,
                        help='Input a list of lambdas to see '
                             'parameter sensitivity')

    parser.add_argument('--listiter', default=None,
                        help='Input a list of iters to see '
                             'parameter sensitivity')

    args = parser.parse_args()

    list_handle(args)

if __name__ == '__main__':
    sys.exit(main())

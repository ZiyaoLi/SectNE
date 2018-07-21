import random
import heapq as hq
import numpy as np


# TODO: implement more options for sample
#       1.sample by p(Node|Community)
#       2.deterministic topological methods


def hkey(weight):
    '''
    weight -> hash_value function used in reservoir-sampling.
    :param weight: given weight
    :return: function value, return 0 for non-positive weights
    '''
    assert weight >= 0
    if weight <= 0:
        return 0
    else:
        return random.random() ** (1.0 / weight)


def reservoir(weights, k):
    '''
    using heap to implement reservoir-sampling.
    :param weights: given weights
    :param k: size of sampling
    :return: sampled indices
    '''
    heap = []
    for idx, weight in enumerate(weights):
        if len(heap) < k:
            hq.heappush(heap, (idx, hkey(weight)))
        else:
            t = hkey(weight)
            if t > heap[0][1]:
                hq.heapreplace(heap, (idx, t))
    rst = []
    while len(heap) > 0:
        rst.append(hq.heappop(heap)[0])
    return rst


def reservoir_deter(weights, k):
    '''
    using heap to implement finding k-max of a list.
    :param weights: given weights
    :param k: k for k-max
    :return: k-max indices
    '''
    heap = []
    for idx, weight in enumerate(weights):
        if len(heap) < k:
            hq.heappush(heap, (idx, weight))
        else:
            if weight > heap[0][1]:
                hq.heapreplace(heap, (idx, weight))
    rst = []
    while len(heap) > 0:
        rst.append(hq.heappop(heap)[0])
    return rst


def sample(net, k, method='deg_prob', vertex_group_sizes=None):
    '''
    sampling k vertices with a given graph by a given method.
    :param net: Graph object
    :param k: sampling size
    :param method: method to sample
    :param vertex_group_sizes: a list of the group sizes of which the vertices are.
    :return: a list of sampled indices
    '''
    if method == 'deg_prob':
        probs = [v.weight() for v in net.vertices]
        rst = reservoir(probs, k)
    elif method == 'deg^2_prob':
        probs = [v.weight() ** 2 for v in net.vertices]
        rst = reservoir(probs, k)
    elif method == 'deg_deter':
        probs = [v.weight() for v in net.vertices]
        rst = reservoir_deter(probs, k)
    elif method == 'deg|group_prob':
        degrees = np.array([v.weight() for v in net.vertices])
        group_sizes = np.array(vertex_group_sizes)
        not_sep = (group_sizes > 1)
        probs = not_sep * degrees / group_sizes
        rst = reservoir(probs, k)
    elif method == 'deg^2|group_prob':
        degrees = np.array([v.weight() ** 2 for v in net.vertices])
        group_sizes = np.array(vertex_group_sizes)
        not_sep = (group_sizes > 1)
        probs = not_sep * degrees / group_sizes
        rst = reservoir(probs, k)
    elif method == 'deg|group_deter':
        degrees = np.array([v.weight() for v in net.vertices])
        group_sizes = np.array(vertex_group_sizes)
        not_sep = (group_sizes > 1)
        probs = not_sep * degrees / group_sizes
        rst = reservoir_deter(probs, k)
    elif method == 'deg^2|group_deter':
        degrees = np.array([v.weight() ** 2 for v in net.vertices])
        group_sizes = np.array(vertex_group_sizes)
        not_sep = (group_sizes > 1)
        probs = not_sep * degrees / group_sizes
        rst = reservoir_deter(probs, k)
    elif method == 'uniform':
        probs = [1] * net.nVertices
        rst = reservoir(probs, k)
    else:
        raise AssertionError
    return rst


if __name__ == '__main__':
    from graph import Graph
    g = Graph('wiki\\links.txt')
    # sample k nodes.
    idx_k = sample(g, 50)
    print(idx_k)

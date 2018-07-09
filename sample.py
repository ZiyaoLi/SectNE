import random
import heapq as hq
from graph import Graph


# TODO: implement more options for sample
#       1.sample by p(Node|Community)
#       2.deterministic topological methods


def hkey(weight):
    return random.random() ** (1.0 / weight)


def reservoir(probs, k):
    heap = []
    for idx, weight in enumerate(probs):
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


def sample(net, k, method='deg'):
    assert method in ('deg', 'deg^2')
    rst = []
    if method == 'deg':
        probs = [len(v) for v in net.vertices]
        rst = reservoir(probs, k)
    elif method == 'deg^2':
        probs = [len(v) ** 2 for v in net.vertices]
        rst = reservoir(probs, k)
    return rst


if __name__ == '__main__':
    g = Graph('graph.txt')
    # sample k nodes.
    idx_k = sample(g, 50)
    print(idx_k)

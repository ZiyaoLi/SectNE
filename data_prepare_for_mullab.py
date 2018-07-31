from score import read_pairs
from collections import defaultdict as ddict
import heapq as hq

dr = 'data\\'
dset = 'flickr\\'
k = 5
n_max_vertices = 2000000

samples = read_pairs(dr + dset + 'groupmemberships.txt')
f = open(dr + dset + 'vertices.txt', 'r')
s = f.read().split()
s = [int(t) for t in s]

cnts = ddict(int)
for vid, gid in samples:
    cnts[gid] += 1
heap = []
for idx, weight in cnts.items():
    if len(heap) < k:
        hq.heappush(heap, (idx, weight))
    else:
        if weight > heap[0][1]:
            hq.heapreplace(heap, (idx, weight))
rst = []
while len(heap) > 0:
    rst.append(hq.heappop(heap)[0])
mapping = dict([(t, i) for i, t in enumerate(rst)])

labels = [None] * n_max_vertices
for t in s:
    if labels[t] is None:
        labels[t] = [0, 0, 0, 0, 0]
for t, g in samples:
    try:
        lid = mapping[g]
        labels[t][lid] = 1
    except KeyError:
        pass

out = open(dr + dset + 'samples.txt', 'w')
for i, label in enumerate(labels):
    if label is not None:
        t = [str(i)] + [str(j) for j in label]
        out.write('\t'.join(t))
        out.write('\n')

from score import read_pairs
from collections import defaultdict as ddict
import heapq as hq
from graph import Graph

dr = 'data\\'
dset = 'flickr\\'
k = 5
n_max_vertices = 2000000

net = Graph(dr + dset + '\\links.txt', typ='dir')
f = open(dr + dset + 'vertices.txt', 'w')
for newVid, vid in net.newVid2vid_mapping.items():
    f.write('%d\n' % vid)
f.close()

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
        labels[t] = [0] * k
for t, g in samples:
    try:
        lid = mapping[g]
        labels[t][lid] = 1
        sign = 1
    except:
        pass

out = open(dr + dset + 'samples.txt', 'w')
out2 = open(dr + dset + 'samples_labeled.txt', 'w')
out3 = open(dr + dset + 'vertices_labeled.txt', 'w')
for i, label in enumerate(labels):
    if label is not None:
        t = [str(i)] + [str(j) for j in label]
        out.write('\t'.join(t))
        out.write('\n')
        if sum(label):
            out2.write('\t'.join(t))
            out2.write('\n')
            out3.write('%d\n' % i)
out.close()
out2.close()
out3.close()

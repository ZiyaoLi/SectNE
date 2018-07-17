from graph import Graph
from sample import sample
from train import Optimizer
from group import Louvain, groups2inv_index, pure_override_nodes
import numpy as np
from numpy.linalg import norm

net = Graph('wiki.txt', typ=1)
k_set = sample(net, k=200, method='deg_deter')
grouping_model = Louvain(net)
groups = grouping_model.execute()
inv_index = groups2inv_index(groups, net.nVertices, k_set)
pure_override_nodes(groups, inv_index)
groups = [k_set] + groups
model = Optimizer(net, groups, dim=100)
vecs_w = []
vecs_c = []

all_idx = []
for t in range(len(groups)):
    print("%d / %d..." % (t + 1, len(groups)))
    w, c = model.train(t, verbose=2)
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
abs_delta = abs(delta)
t = norm(delta, 'fro')
tt = norm(original, 'fro')
print("Original - %.4f, delta - %.4f, percentage - %.4f"
      % (tt, t, t / tt))

# a SVD implementation to exam how good is the result
u, d, v = np.linalg.svd(original)
w_svd = (u[:, :100] * np.sqrt(d[:100])).T
c_svd = (v.T[:, :100] * np.sqrt(d[:100])).T
reconstruct_svd = w_svd.T @ c_svd
delta_svd = original - reconstruct_svd
t_svd = norm(delta_svd, 'fro')
print("Original - %.4f, delta - %.4f, percentage - %.4f"
      % (tt, t_svd, t_svd / tt))

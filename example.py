from graph import Graph
from sample import sample

g = Graph('graph.txt')

# sample k nodes.
idx_k = sample(g, 50)

# below 2 methods of Graph gives you A and M of a subset of nodes of G.
# for example, M00 = calc_matrix(S0, S0), where S0 are the indices of selected k vertices.
mat_proximity = g.fetch_proxmat(idx_k, idx_k)
# mat_tadw: A + A * A / 2
mat_tadw = g.calc_matrix(idx_k, idx_k)

# please notice that @ operator is for matrix multiplication

# for A = (a1, a2, ..., an) where a_i are column vectors,
#     B = (b1, b2, ..., bn) with the same shape, the function
#     sum(A * B) gives the vector
#             (<a1, b1>, <a2, b2>, ..., <an, bn>).

import numpy as np
from numpy.linalg import norm
from graph import Graph
from sample import sample
from descend import *

# hyper parameters
THETA = 1
LAMBDA = 0.8
ETA = 0.1
MAX_ITER = 100
EPSILON = 1e-4

# dimensionality
DIMENSION = 100
K_SIZE = 200

# vibration solution arguments
N_HISTORY_MONITOR = 5
THRESHOLD_MONITOR = 0.05
PERCENTAGE_AVG = 0.55


class Optimizer:
    def __init__(self, graph, groups,
                 dim=DIMENSION,
                 theta=THETA, lam=LAMBDA, eta=ETA,
                 max_iter=MAX_ITER, epsilon=EPSILON,
                 cg_max_iter=CG_MAX_ITER, cg_eps=CG_EPSILON,
                 descending_method=inverse_descending):
        self.graph = graph
        self.groups = groups
        self.nGroups = len(groups)
        self.lam = lam
        self.eta = eta
        self.theta = theta
        self.max_iter = max_iter
        self.eps = epsilon
        self.cg_max_iter = cg_max_iter
        self.cg_eps = cg_eps
        self.descending_method = descending_method

        # fetch all matrix related to k at first to save time
        # in sequential embedding process.
        self.m_0_all = self.graph.calc_matrix(groups[0], list(range(graph.nVertices)))
        self.m_all_0 = self.graph.calc_matrix(list(range(graph.nVertices)), groups[0])

        # k decomposition: SVD
        m0 = self.m_0_all[:, groups[0]]
        u, d, v = np.linalg.svd(m0)
        self.phi = (u[:, :dim] * np.sqrt(d[:dim])).T
        self.psi = (v.T[:, :dim] * np.sqrt(d[:dim])).T
        self.m0_tilde = self.phi.T @ self.psi

    def _get_rest_idx(self, group_idx):
        ib = []
        for i in range(1, len(self.groups)):
            if i == group_idx:
                continue
            ib += self.groups[i]
        return ib

    def train(self, group_idx, verbose=1):

        assert group_idx < self.nGroups
        if group_idx == 0:
            return self.phi, self.psi

        indices = self.groups[group_idx]
        rest_indices = self._get_rest_idx(group_idx)

        # pre-calculate the matrices and intercepts to be used
        # to minimize the efforts in the loop.

        # 1.t_mm: \tilde{m_0} * \tilde{m_0}^T
        t_mm = self.m0_tilde @ self.m0_tilde.T

        # 2. pre-calculate constants for A
        m_1_0 = self.m_all_0[indices, :]
        m_0_r = self.m_0_all[:, rest_indices]
        m_1_r = self.graph.calc_matrix(self.groups[group_idx], rest_indices)
        # G_A = G0_A + G(B), b_A = b0_A + b(B),
        # where G(B) and b(B) are the B-related additive factors.
        G0_A = t_mm + self.lam * (m_0_r @ m_0_r.T) + self.eta * np.eye(len(t_mm))
        b0_A = self.m0_tilde @ m_1_0.T + self.lam * (m_0_r @ m_1_r.T)
        # delete useless variables in time.
        del m_1_r

        # 3. duel process for B
        m_0_1 = self.m_0_all[:, indices]
        m_r_0 = self.m_all_0[rest_indices, :]
        m_r_1 = self.graph.calc_matrix(rest_indices, self.groups[group_idx])
        G0_B = t_mm.T + self.lam * (m_r_0.T @ m_r_0) + self.eta * np.eye(len(t_mm))
        b0_B = self.m0_tilde.T @ m_0_1 + self.lam * (m_r_0.T @ m_r_1)
        del m_r_1

        del t_mm, rest_indices

        # 4. m_1_1
        m_1_1 = self.graph.calc_matrix(self.groups[group_idx], self.groups[group_idx])

        # init
        n_0 = len(self.groups[0])
        n_1 = len(self.groups[group_idx])
        # random initial values
        A_prev = np.random.random((n_0, n_1))
        B_prev = np.random.random((n_0, n_1))
        # zero initial values
        # A_prev = np.zeros((n_0, n_1))
        # B_prev = np.zeros((n_0, n_1))
        ite = 0
        altered = np.inf  # initial 'altered' doesn't stop the loop
        hist_altered = [np.inf] * N_HISTORY_MONITOR
        while ite < self.max_iter and altered > self.eps:
            ite += 1
            # fix B update A
            t_mb = self.m0_tilde @ B_prev
            G_A = G0_A + self.theta * (t_mb @ t_mb.T)
            b_A = b0_A + self.theta * (t_mb @ m_1_1.T)
            A, state_A, res_A = self.descending_method(A_prev, G_A, b_A, self.cg_max_iter, self.cg_eps)
            del t_mb, G_A, b_A

            # fix A update B
            t_ma = self.m0_tilde.T @ A_prev
            G_B = G0_B + self.theta * (t_ma @ t_ma.T)
            b_B = b0_B + self.theta * (t_ma @ m_1_1)
            B, state_B, res_B = self.descending_method(A_prev, G_B, b_B, self.cg_max_iter, self.cg_eps)
            del t_ma, G_B, b_B

            altered = (norm(A - A_prev, np.inf) + norm(B - B_prev, np.inf)) / A.shape[1]
            hist_altered = hist_altered[1:] + [altered]
            improve_percentage = (np.mean(hist_altered) - altered) / altered
            if improve_percentage < THRESHOLD_MONITOR:
                A = A * PERCENTAGE_AVG + A_prev * (1 - PERCENTAGE_AVG)
                B = B * PERCENTAGE_AVG + B_prev * (1 - PERCENTAGE_AVG)
            A_prev = A
            B_prev = B

            if (state_A or state_B) and (verbose == 2):
                res_mean = (np.mean(res_A) * state_A + np.mean(res_B) * state_B) / (state_A + state_B)
                print("Warning: CG doesn't converge at iter %d, group %d. percentage of residuals: %.4f"
                      % (ite, group_idx, res_mean))

        if ite == self.max_iter and verbose >= 1:
            print("Warning: optimization doesn't converge for group %d, residuals %.4f" % (group_idx, altered))

        w = self.phi @ A
        c = self.psi @ B

        # debug info
        # original = net.calc_matrix(self.groups[0] + self.groups[group_idx],
        #                            self.groups[0] + self.groups[group_idx])
        # reconstruct = np.concatenate([self.phi, w], 1).T @ \
        #               np.concatenate([self.psi, c], 1)
        # delta = abs(original - reconstruct)
        #
        # t = norm(delta)

        return w, c
        
    # def get_embeddings(self):
    #     embeddings = {}
    #     for i, v in enumerate(self.groups[0]) :
    #         embeddings[v] = self.wt[i].tolist()
    #     print("{} Blocks in All".format(len(self.groups)))
    #     for index in range(1, len(self.groups)) :
    #         self.train(index, embeddings)
    #         print("Block {} Finished!".format(index))
    #     return embeddings

if __name__ == '__main__':
    net = Graph('simple\\links.txt', typ=1)
    k_set = sample(net, k=3, method='deg^2_prob')
    sep = [[3, 4, 7], [1, 0, 6], [2, 8], [5, 9]]
    model = Optimizer(net, sep, dim=2)
    vecs_w = []
    vecs_c = []
    for t in range(4):
        w, c = model.train(t)
        vecs_w.append(w)
        vecs_c.append(c)

    # concatenate all the derived vectors together
    ws = np.concatenate(vecs_w, 1)
    cs = np.concatenate(vecs_c, 1)

    # reconstructing matrix over the order of sampled vertices
    reconstruct = ws.T @ cs
    all_idx = sep[0] + sep[1] + sep[2] + sep[3]
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
    w_svd = (u[:, :2] * np.sqrt(d[:2])).T
    c_svd = (v.T[:, :2] * np.sqrt(d[:2])).T
    reconstruct_svd = w_svd.T @ c_svd
    delta_svd = original - reconstruct_svd
    t_svd = norm(delta_svd, 'fro')
    print("Original - %.4f, delta - %.4f, percentage - %.4f"
          % (tt, t_svd, t_svd / tt))




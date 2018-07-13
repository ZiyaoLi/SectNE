
# W'C = M00
# ib -> \bar{i}
# lam -> lambda
# min{ (M0i-M00 B)^2+(Mi0-A M00)^2+(Mii-A' M00 B)^2 +
#      lam*{(M0ib Miib'-M0ib M0ib' A)^2+( Mib0' Mibi-Mibo' Mib0 B)^2} + eta*(A^2+B^2) }

# please notice that @ operator is for matrix multiplication

# for A = (a1, a2, ..., an) where a_i are column vectors,
#     B = (b1, b2, ..., bn) with the same shape, the function
#     sum(A * B) gives the vector
#             (<a1, b1>, <a2, b2>, ..., <an, bn>).

import numpy as np
from numpy.linalg import norm
from graph import Graph
from sample import sample


LAMBDA = 0.5
ETA = 0.3
MAX_ITER = 20
CG_MAX_ITER = 20
EPSILON = 1e-8
DIMENSION = 100
K_SIZE = 200


def conjugate_gradient(x, A, b, max_iter=CG_MAX_ITER, eps=EPSILON):
    ite = 0
    r = b - A @ x
    rho = sum(r * r)
    while ite <= max_iter and (rho > eps).any():
        ite += 1
        if ite == 1:
            p = r
        else:
            beta = rho / rho_tilde
            p = r + beta * p

        w = A @ p
        alpha = rho / sum(p * w)
        x = x + alpha * p
        r = r - alpha * w
        rho_tilde = rho
        rho = sum(r * r)
    return x


class Optimizer:
    def __init__(self, graph, groups,
                 dim=DIMENSION,
                 lam=LAMBDA, eta=ETA,
                 max_iter=MAX_ITER, criterion=EPSILON):
        self.graph = graph
        self.groups = groups
        self.nGroups = len(groups)
        self.lam = lam
        self.eta = eta
        self.max_iter = max_iter
        self.criterion = criterion

        # k decomposition: SVD
        m0 = graph.calc_matrix(groups[0], groups[0])
        u, d, v = np.linalg.svd(m0)
        self.phi = (u[:, :dim] @ np.diag(np.sqrt(d[:dim]))).T
        self.psi = (v[:, :dim] @ np.diag(np.sqrt(d[:dim]))).T
        self.m0_tilde = self.phi.T @ self.psi

    def _get_rest_idx(self, group_idx):
        ib = []
        for i in range(1, len(self.groups)):
            if i == group_idx:
                continue
            ib += self.groups[i]
        return ib

    def train(self, group_idx):

        assert group_idx < self.nGroups
        if group_idx == 0:
            return self.phi

        rest_indices = self._get_rest_idx(group_idx)
        t_mm = self.m0_tilde @ self.m0_tilde.T

        # pre-calculate the matrices and intercepts to be used
        # to minimize the efforts in the loop.
        m_1_0 = self.graph.calc_matrix(self.groups[group_idx], self.groups[0])
        m_0_r = self.graph.calc_matrix(self.groups[0], rest_indices)
        m_1_r = self.graph.calc_matrix(self.groups[group_idx], rest_indices)
        G0_A = t_mm + self.lam * (m_0_r.T @ m_0_r) + self.eta
        b0_A = self.m0_tilde @ m_1_0 + self.lam * (m_0_r @ m_1_r.T)
        # delete useless variables in time.
        del m_1_0, m_0_r, m_1_r

        m_0_1 = self.graph.calc_matrix(self.groups[0], self.groups[group_idx])
        m_r_0 = self.graph.calc_matrix(rest_indices, self.groups[0])
        m_r_1 = self.graph.calc_matrix(rest_indices, self.groups[group_idx])
        G0_B = t_mm.T + self.lam * (m_r_0.T @ m_r_0) + self.eta
        b0_B = self.m0_tilde.T @ m_0_1 + self.lam * (m_r_0.T @ m_r_1)
        del m_0_1, m_r_0, m_r_1

        del t_mm, rest_indices

        m_1_1 = self.graph.calc_matrix(self.groups[group_idx], self.groups[group_idx])

        # init
        n_0 = len(self.groups[0])
        n_1 = len(self.groups[group_idx])
        # random initial values
        A = np.random.random((n_0, n_1))
        B = np.random.random((n_0, n_1))
        ite = 0
        while ite <= self.max_iter:
            ite += 1
            # fix B update A
            t_mb = self.m0_tilde @ B
            G_A = G0_A + t_mb @ t_mb.T
            b_A = b0_A + t_mb @ m_1_1.T
            A = conjugate_gradient(A, G_A, b_A, self.max_iter)
            del t_mb, G_A, b_A

            # fix A update B
            t_ma = self.m0_tilde.T @ A
            G_B = G0_B + t_ma @ t_ma.T
            b_B = b0_B + t_ma @ m_1_1
            B = conjugate_gradient(A, G_B, b_B, self.max_iter)
            del t_ma, G_B, b_B

        w = self.phi @ A
        c = self.psi @ B
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
    net = Graph('sample.txt', typ=1)
    k_set = sample(net, K_SIZE, 'deg^2')
    model = Optimizer(net, [k_set])


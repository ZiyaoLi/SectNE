
# W'C = M00
# ib -> \bar{i}
# lam -> lambda
# min{ (M0i-M00 B)^2+(Mi0-A M00)^2+(Mii-A' M00 B)^2 +
#      lam*{(M0ib Miib'-M0ib M0ib' A)^2+( Mib0' Mibi-Mibo' Mib0 B)^2} + eta*(A^2+B^2) }

# please notice that @ operator is for matrix multiplication

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


def conjugate_gradient(x, A, grad, max_iter=CG_MAX_ITER, eps=EPSILON):
    r = -grad
    if r.all() == 0:  ##
        return x  ##
    p = r
    for i in range(max_iter):
        r2 = r.T @ r
        Ap = A @ p
        alpha = r2 / (p.T @ Ap)
        x = x + p @ np.diag(np.diag(alpha))
        r = r - Ap @ np.diag(np.diag(alpha))
        if norm(r, np.inf) <= eps:
            break
        beta = np.dot(r.T, r)/r2
        p = r + np.dot(p, np.diag(np.diag(beta)))
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

    def train(self, group_idx, embeddings):

        assert group_idx < self.nGroups
        if group_idx == 0:
            return self.phi
        m0_1 = self.graph.calc_matrix(self.groups[0], self.groups[group_idx])
        m1_1 = self.graph.calc_matrix(self.groups[group_idx], self.groups[group_idx])
        rest_indices = self._get_rest_idx(group_idx)
        m0_1b = self.graph.calc_matrix(self.groups[0], rest_indices)
        m1_1b = self.graph.calc_matrix(self.groups[group_idx], rest_indices)

        # init
        n_k = len(self.groups[0])
        n_m1 = len(self.groups[group_idx])
        # random initial values
        A = np.random.random((n_k, n_m1))
        B = np.random.random((n_k, n_m1))
        ite = 0
        while ite <= self.max_iter:
            ite += 1
            # fix B update A
            MB = self.m0_tilde @ B
            M2 = m0_1b @ m0_1b.T
            # Hessian
            HA = self.m0_tilde @ self.m0_tilde + MB @ MB.T + self.lam * (M2 @ M2) + self.eta
            bA = self.m0_tilde @ m0_1 + MB @ m1_1 + self.lam * (M2 @ (m0_1b @ m1_1b.T))
            GradA = (HA @ A)-bA
            A = conjugate_gradient(A, HA, GradA, self.max_iter)

            # fix A update B
            MA = self.m0_tilde @ A
            MT = m0_1b @ m0_1b.T
            # Hessian
            HB = self.m0_tilde @ self.m0_tilde + MA @ MA.T + self.lam * (MT @ MT) + self.eta
            bB = self.m0_tilde @ m0_1 + MA @ m1_1 + self.lam * (MT @ (m0_1b @ m1_1b.T))
            GradB = (HB @ B)-bB
            B = conjugate_gradient(B, HB, GradB, self.max_iter)

        w = self.phi @ A
        for i, v in enumerate(self.groups[group_idx]):
            embeddings[v] = w.T[i].tolist()
        
    def get_embeddings(self):
        embeddings = {}
        for i, v in enumerate(self.groups[0]) :
            embeddings[v] = self.wt[i].tolist()
        print("{} Blocks in All".format(len(self.groups)))
        for index in range(1, len(self.groups)) :
            self.train(index, embeddings)
            print("Block {} Finished!".format(index))
        return embeddings

if __name__ == '__main__':
    net = Graph('wiki.txt', typ=1)
    k_set = sample(net, K_SIZE, 'deg^2')
    model = Optimizer(net, [k_set])

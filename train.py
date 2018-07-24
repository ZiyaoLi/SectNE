
# please notice that @ operator is for matrix multiplication

# for A = (a1, a2, ..., an) where a_i are column vectors,
#     B = (b1, b2, ..., bn) with the same shape, the function
#     sum(A * B) gives the vector
#             (<a1, b1>, <a2, b2>, ..., <an, bn>).

from numpy.linalg import norm
from graph import Graph
from sample import sample
from descend import *
from numba import jit
import time
from scipy import sparse as sp

# hyper parameters
LAMBDA = 1
ETA = 0.1
MAX_ITER = 50
EPSILON = 1e-4
DESCENDING_METHOD = inverse_descending

# dimensionality
DIMENSION = 100
K_SIZE = 200

# vibration solution arguments
N_HISTORY_MONITOR = 3
THRESHOLD_MONITOR = 0.1
PERCENTAGE_AVG = 0.55

VERBOSE = 1
DEBUG = False


class Optimizer:
    def __init__(self, graph, groups, dim=DIMENSION,
                 lam=LAMBDA, eta=ETA,
                 max_iter=MAX_ITER, epsilon=EPSILON,
                 cg_max_iter=CG_MAX_ITER, cg_eps=CG_EPSILON,
                 descending_method=DESCENDING_METHOD,
                 verbose=VERBOSE,
                 grouping_strategy='Louvain',
                 sample_strategy='NotSpecified'):

        self.graph = graph
        self.k_size = len(groups[0])
        self.dim = dim
        self.groups = groups
        self.lam, self.eta = lam, eta
        self.max_iter, self.eps = max_iter, epsilon
        self.cg_max_iter, self.cg_eps = cg_max_iter, cg_eps
        self.descending_method = descending_method
        self.inverse_index = None
        self.trained_embeddings = {}
        self.grouping_strategy = grouping_strategy
        self.sample_strategy = sample_strategy

        # fetch the matrix related to k at initialization in order to
        # save time in following sequential embedding process.
        pt = time.time()
        self.eye = sp.eye(self.k_size)
        self.m_0_all = self.graph.calc_matrix_sparse(groups[0], list(range(graph.nVertices)))
        self.m_all_0 = self.graph.calc_matrix_sparse(list(range(graph.nVertices)), groups[0])
        self.m_0_0r, self.m_0_0c = self.graph.calc_matrix_sparse(groups[0], groups[0], -1)
        if verbose:
            print('Fetch matrix time: %.2f'
                  % (time.time() - pt))

        pt = time.time()
        self.m_0_all2 = self.m_0_all @ self.m_0_all.T \
            - self.m_0_0r @ self.m_0_0r.T
        self.m_all_02 = self.m_all_0.T @ self.m_all_0 \
            - self.m_0_0c.T @ self.m_0_0c
        if verbose:
            print('Calculate remembered matrix time: %.2f'
                  % (time.time() - pt))

        # k decomposition: SVD
        u, d, v = np.linalg.svd(self.m_0_0r.toarray())
        self.phi = (u[:, :dim] * np.sqrt(d[:dim])).T
        self.psi = (v.T[:, :dim] * np.sqrt(d[:dim])).T
        self.m0_tilde = self.phi.T @ self.psi
        self.m0_m0T = self.m0_tilde @ self.m0_tilde.T
        self.m0T_m0 = self.m0_tilde.T @ self.m0_tilde

    def train(self, group_idx, verbose=VERBOSE):
        assert group_idx < len(self.groups), 'group index exceeded max'
        if group_idx == 0:
            # asking for the K vertices
            return self.phi, self.psi

        ppt = time.time()

        indices = self.groups[group_idx]
        n_0 = len(self.groups[0])
        n_1 = len(self.groups[group_idx])

        if verbose:
            print('Start training group %d. %4d Vertices.'
                  % (group_idx, n_1))

        # ###### PREPARATION ######
        # 1.MATRIX FETCH
        pt = time.time()
        # if n_1 <= n_0:
            # No need to sparsify matrices
        m_0_1 = self.graph.calc_matrix_sparse(self.groups[0], indices, style=0)
        m_1_0 = self.graph.calc_matrix_sparse(indices, self.groups[0], style=1)
        m_1_1r, m_1_1c = self.graph.calc_matrix_sparse(indices, indices, style=-1)
        m_1_all = self.graph.calc_matrix_sparse(indices, list(range(self.graph.nVertices)), style=0)
        m_all_1 = self.graph.calc_matrix_sparse(list(range(self.graph.nVertices)), indices, style=1)
        if verbose:
            print('Fetch matrix time: %.2f'
                  % (time.time() - pt))

        # 2.G_0 & B_0 CALCULATION
        # G_A = G0_A + G(B), b_A = b0_A + b(B),
        # where G(B) and b(B) are the B-related additive factors,
        # vice versa
        pt = time.time()
        G0_A = self.m0_m0T + \
            self.lam * (self.m_0_all2 - m_0_1 @ m_0_1.T) + \
            self.eta * self.eye
        G0_B = self.m0T_m0 + \
            self.lam * (self.m_all_02 - m_1_0.T @ m_1_0) + \
            self.eta * self.eye
        b0_A = self.m0_tilde @ m_1_0.T + \
            self.lam * (self.m_0_all @ m_1_all.T -
                        self.m_0_0r @ m_1_0.T -
                        m_0_1 @ m_1_1r.T)
        b0_B = self.m0_tilde.T @ m_0_1 + \
            self.lam * (self.m_all_0.T @ m_all_1 -
                        self.m_0_0c.T @ m_0_1 -
                        m_1_0.T @ m_1_1c)
        if verbose:
            print('Calculate G_0 and b_0 time: %.2f'
                  % (time.time() - pt))
        # delete useless variables in time.
        del m_0_1, m_1_0, m_all_1, m_1_all

        # 3.INITIALIZATION
        # init
        # random initial values
        A_prev = np.random.random((n_0, n_1))
        B_prev = np.random.random((n_0, n_1))
        # zero initial values
        # A_prev = np.zeros((n_0, n_1))
        # B_prev = np.zeros((n_0, n_1))
        ite = 0
        altered = np.inf  # so that initial 'altered' doesn't stop the loop
        hist_altered = [np.inf] * N_HISTORY_MONITOR

        # ###### ITERATION ######
        pt = time.time()
        while ite < self.max_iter and altered > self.eps:
            ite += 1
            # fixing B updating A
            t_mb = self.m0_tilde @ B_prev
            G_A = G0_A + t_mb @ t_mb.T
            b_A = b0_A + t_mb @ m_1_1r.T
            A, state_A, res_A = self.descending_method(A_prev, G_A, b_A, self.cg_max_iter, self.cg_eps)
            del t_mb, G_A, b_A

            # fixing A updating B
            t_ma = self.m0_tilde.T @ A_prev
            G_B = G0_B + t_ma @ t_ma.T
            b_B = b0_B + t_ma @ m_1_1c
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

            # if (state_A or state_B) and (verbose == 2):
            #     res_mean = (np.mean(res_A) * state_A + np.mean(res_B) * state_B) / (state_A + state_B)
            #     print("Warning: CG doesn't converge at iter %d, group %d. percentage of residuals: %.4f"
            #           % (ite, group_idx, res_mean))

        # To avoid the error of providing max_iter as 0
        A = A_prev
        B = B_prev

        if ite == self.max_iter and verbose >= 1:
            print("Warning: optimization doesn't converge for group %d, residuals %.4f" % (group_idx, altered))

        if verbose:
            print('Optimization iterations time: %.2f. Average: %.2f'
                  % (time.time() - pt, (time.time() - pt) / ite))

        w = self.phi @ A
        c = self.psi @ B

        # # ###### DEBUG: ITERATION PERFORMANCE ######
        # if DEBUG:
        #     original = self.graph.calc_matrix(self.groups[0] + self.groups[group_idx],
        #                                       self.groups[0] + self.groups[group_idx])
        #     reconstruct = np.concatenate([self.phi, w], 1).T @ \
        #                   np.concatenate([self.psi, c], 1)
        #     delta = abs(original - reconstruct)
        #
        #     norm_delta = norm(delta, 'fro')
        #     norm_original = norm(original, 'fro')
        #     print("Original - %.4f, delta - %.4f, percentage - %.4f"
        #           % (norm_original, norm_delta,
        #              norm_delta / norm_original))

        return w, c

    def _set_inverse_index(self):
        self.inverse_index = [-1] * self.graph.nVertices
        for group_idx, group in enumerate(self.groups):
            for member_idx, member in enumerate(group):
                self.inverse_index[member] = (group_idx, member_idx)

    def embed(self, vertex_idx, verbose=VERBOSE):
        if isinstance(vertex_idx, list):
            # if input is a list of targets, do it one by one.
            rst = []
            for vid in vertex_idx:
                rst.append(self.embed(vid, verbose=verbose))
            return rst
        if self.inverse_index is None:
            # set an inverse index: vid -> (group_id, member_id)
            self._set_inverse_index()
        group_idx, member_idx = self.inverse_index[vertex_idx]
        if vertex_idx not in self.trained_embeddings.keys():
            w, c = self.train(group_idx, verbose=verbose)
            for mid, vid in enumerate(self.groups[group_idx]):
                self.trained_embeddings[vid] = w[:, mid]
        return self.trained_embeddings[vertex_idx]


if __name__ == '__main__':
    net = Graph('data\\simple\\links.txt', typ=1)
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
    res_fnorm = norm(delta, 'fro')
    ori_fnorm = norm(original, 'fro')
    print("Original - %.4f, delta - %.4f, percentage - %.4f"
          % (ori_fnorm, res_fnorm, res_fnorm / ori_fnorm))

    # a SVD implementation to exam how good is the result
    u, d, v = np.linalg.svd(original)
    w_svd = (u[:, :2] * np.sqrt(d[:2])).T
    c_svd = (v.T[:, :2] * np.sqrt(d[:2])).T
    reconstruct_svd = w_svd.T @ c_svd
    delta_svd = original - reconstruct_svd
    t_svd = norm(delta_svd, 'fro')
    print("Original - %.4f, delta - %.4f, percentage - %.4f"
          % (ori_fnorm, t_svd, t_svd / ori_fnorm))




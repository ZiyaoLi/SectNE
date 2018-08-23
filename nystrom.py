from sample import sample
import numpy as np


def mydiv(a):
    for i, v in enumerate(a):
        if abs(v) > 1e-6:
            a[i] = 1 / v
        else:
            a[i] = 0
    return a


def nystrom(net, all_idx, k, d):
    k_set = sample(net, k, 'deg^2_prob')
    mat = net.calc_matrix_sparse(k_set, k_set).toarray()
    u, dd, v = np.linalg.svd(mat)
    reconstruct_mat = u[:, :d] @ np.diag(mydiv(dd[:d])) @ v[:d, :]
    left = net.calc_matrix_sparse(all_idx, k_set)
    right = net.calc_matrix_sparse(k_set, all_idx)
    return left @ reconstruct_mat @ right

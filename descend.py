import numpy as np
from numba import jit

# please notice that @ operator is for matrix multiplication

# for A = (a1, a2, ..., an) where a_i are column vectors,
#     B = (b1, b2, ..., bn) with the same shape, the function
#     sum(A * B) gives the vector
#             (<a1, b1>, <a2, b2>, ..., <an, bn>).

CG_EPSILON = 1e-4
CG_MAX_ITER = 20


def conjugate_gradient(x, A, b, max_iter=CG_MAX_ITER, eps=CG_EPSILON):
    '''
    Implemented from algorithm CG for SPD matrix linear equations,
    from XU, Shufang, et. Numerical Linear Algebra.
    :param x: initial value x0. Can be matrix of size (n x p) for a
              p-parallel CG.
    :param A: matrix in problem Ax = b.
    :param b: vector(matrix) in problem Ax = b. Notice that b \in R^(n x p).
    :param max_iter: max iterations
    :param eps: stop criterion
    :return: optimized x, algorithm state, residual squares
    '''
    b_norms = sum(b * b)
    criterion = eps * b_norms
    ite = 0
    r = b - A @ x
    rho = sum(r * r)
    while ite < max_iter and (rho > criterion).any():
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
    return x, ite == max_iter, rho / b_norms


def preconditioning_conjugate_gradient(x, A, b, max_iter=CG_MAX_ITER, eps=CG_EPSILON):
    '''
    Implemented from algorithm CG for SPD matrix linear equations,
    from XU, Shufang, et. Numerical Linear Algebra.
    :param x: initial value x0. Can be matrix of size (n x p) for a
              p-parallel CG.
    :param A: matrix in problem Ax = b.
    :param b: vector(matrix) in problem Ax = b. Notice that b \in R^(n x p).
    :param max_iter: max iterations
    :param eps: stop criterion
    :return: optimized x, algorithm state, residual squares
    '''
    b_norms = sum(b * b)
    criterion = eps * b_norms
    ite = 0
    r = b - A @ x
    m = np.diag(A)
    rho_r = sum(r * r)
    while ite < max_iter and (rho_r > criterion).any():
        ite += 1
        z = (r.T / m).T
        if ite == 1:
            p = z
            rho = sum(r * z)
        else:
            rho_tilde = rho
            rho = sum(r * z)
            beta = rho / rho_tilde
            p = z + beta * p

        w = A @ p
        alpha = rho / sum(p * w)
        x = x + alpha * p
        r = r - alpha * w
        rho_r = sum(r * r)

    return x, ite == max_iter, rho_r / b_norms


@jit
def inverse_descending(x, A, b, max_iter=CG_MAX_ITER, eps=CG_EPSILON):
    '''
    Implemented from algorithm CG for SPD matrix linear equations,
    from XU, Shufang, et. Numerical Linear Algebra.
    :param x: initial value x0. Can be matrix of size (n x p) for a
              p-parallel CG.
    :param A: matrix in problem Ax = b.
    :param b: vector(matrix) in problem Ax = b. Notice that b \in R^(n x p).
    :param max_iter: max iterations (keeping place)
    :param eps: stop criterion (keeping place)
    :return: optimized x, algorithm state, residual squares
    '''
    return np.linalg.inv(A) @ b, False, np.zeros(b.shape[1])

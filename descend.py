import numpy as np
from numba import jit
from scipy.linalg import solve

# please notice that @ operator is for matrix multiplication

# for A = (a1, a2, ..., an) where a_i are column vectors,
#     B = (b1, b2, ..., bn) with the same shape, the function
#     sum(A * B) gives the vector
#             (<a1, b1>, <a2, b2>, ..., <an, bn>).

CG_EPSILON = 1e-3
CG_MAX_ITER = 3


@jit
def vector_dot(a, b, out):
    return np.sum(np.multiply(a, b), axis=0, out=out)


@jit
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
    b_norms = np.multiply(b, b).sum(0)
    criterion = eps * b_norms
    ite = 0
    r = b - A @ x
    rho = np.multiply(r, r).sum(0)
    # to satisfy our numba:
    rho_tilde = rho
    p = r
    while ite < max_iter and (rho > criterion).any():
        ite += 1
        if ite == 1:
            p = r
        else:
            beta = np.divide(rho, rho_tilde)
            p = r + np.multiply(beta, p)

        w = A @ p
        alpha = rho / np.multiply(p, w).sum(0)
        x = x + np.multiply(alpha, p)
        r = r - np.multiply(alpha, w)
        rho_tilde = rho
        rho = np.multiply(r, r)
    return x, ite == max_iter


@jit
def preconditioning_conjugate_gradient(x0, A, b, max_iter=CG_MAX_ITER, eps=CG_EPSILON):
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
    x = x0.copy()
    b_norms = np.multiply(b, b).sum(0)
    criterion = np.multiply(eps, b_norms, out=b_norms)

    ite = 0
    r = b - A @ x
    m = np.matrix(np.diag(A)).T
    rho_r = np.multiply(r, r).sum(0)

    # to satisfy our numba:
    rho = np.matrix(np.zeros_like(rho_r), copy=False)
    rho_tilde = np.matrix(np.zeros_like(rho_r), copy=False)
    p = np.matrix(np.zeros_like(x), copy=False)
    z = np.matrix(np.zeros_like(x), copy=False)
    alpha = np.matrix(np.zeros_like(rho_r), copy=False)
    beta = np.matrix(np.zeros_like(rho_r), copy=False)
    w = np.matrix(np.zeros_like(x), copy=False)
    pT_A_p = np.matrix(np.zeros_like(rho_r), copy=False)

    while ite < max_iter and (rho_r > criterion).any():
        ite += 1
        z = np.divide(r, m, out=z, where=m != 0)
        if ite == 1:
            p = z.copy()
            rho = vector_dot(r, z, out=rho)
        else:
            # swap the reference of rho and rho_tilde
            _ = rho_tilde
            rho_tilde = rho
            rho = _
            rho = vector_dot(r, z, out=rho)
            beta = np.divide(rho, rho_tilde, out=beta, where=rho_tilde != 0)
            p = np.add(z, np.multiply(beta, p), out=p)

        w = np.dot(A, p, out=w)
        pT_A_p = vector_dot(p, w, out=pT_A_p)
        alpha = np.divide(rho, pT_A_p, out=alpha, where=pT_A_p != 0)
        x = np.add(x, np.multiply(alpha, p), out=x)
        r = np.subtract(r, np.multiply(alpha, w), out=r)
        rho_r = vector_dot(r, r, out=rho_r)

    return x, ite == max_iter


@jit
def inverse_descending(x, A, b):
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
    return np.linalg.inv(A) @ b, False


@jit
def scipy_solve_descending(x, A, b):
    return solve(A, b, assume_a='pos'), False


import numpy as np
from collections import defaultdict as ddict


def sparse_vector_mul(dict1, dict2):
    rst = 0
    dict_short = dict1 if len(dict1) < len(dict2) else dict2
    dict_long = dict1 if len(dict1) >= len(dict2) else dict2
    for key, value in dict_short.items():
        try:
            rst += value * dict_long[key]
        except KeyError:
            pass
    return rst


def dense_vector_sparse_mul(dense_vec, sparse):
    rst = ddict(int)
    for i, weight in enumerate(dense_vec):
        for key, value in sparse.matrix[i].items():
            rst[key] += value * weight
    return dict(rst)


def dense_sparse_mul(dense, sparse):
    assert sparse.storetype == 0
    mat = []
    for vector in dense:
        mat.append(dense_vector_sparse_mul(vector, sparse))
    rst = SparseMatrix(len(dense), sparse.size[1], 0)
    rst.set_matrix(mat)
    return rst.to_dense()


class SparseMatrix:
    def __init__(self, nrow, ncol, storetype):
        self.size = (nrow, ncol)
        self.storetype = storetype
        self.matrix = []
        n_prim = nrow if storetype == 0 else ncol
        for i in range(n_prim):
            self.matrix.append({})

    def set_matrix(self, raw_matrix):
        self.matrix = raw_matrix

    def T(self):
        transpose = SparseMatrix(self.size[1], self.size[0],
                                 0 if self.storetype else 1)
        transpose.set_matrix(self.matrix)
        return transpose

    def add_to_entry(self, prim_id, value_id, value):
        try:
            self.matrix[prim_id][value_id] += value
        except KeyError:
            self.matrix[prim_id][value_id] = value

    def to_dense(self):
        mat = np.zeros(self.size)
        if self.storetype == 0:
            for row, row_vec in enumerate(self.matrix):
                for col, value in row_vec.items():
                    mat[row, col] = value
        else:
            for col, col_vec in enumerate(self.matrix):
                for row, value in enumerate(col_vec.items()):
                    mat[row, col] = value
        return mat

    def __matmul__(self, other):
        assert isinstance(other, SparseMatrix)
        assert self.storetype == 0 and other.storetype == 1
        assert self.size[1] == other.size[0]
        mat = np.zeros([self.size[0], other.size[1]])
        for i, row_vec in enumerate(self.matrix):
            for j, col_vec in enumerate(other.matrix):
                mat[i, j] = sparse_vector_mul(row_vec, col_vec)
        return mat

    def change_axis(self):
        rst = SparseMatrix(self.size[0], self.size[1],
                           0 if self.storetype == 1 else 1)
        for i, row_vec in enumerate(self.matrix):
            for key, value in row_vec.items():
                rst.add_to_entry(key, i, value)
        return rst


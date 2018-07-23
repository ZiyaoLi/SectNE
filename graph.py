import numpy as np
from collections import defaultdict as ddict
from scipy import sparse as sp

LAMBDA = 1
VERBOSE = True
N_EDGE_VERBOSE = 5e6
N_VERTEX_VERBOSE = 5e5


def inverse_index(idx, n_max_key, typ):
    if typ == 'list':
        rst = [-1] * n_max_key
        for i, item in enumerate(idx):
            rst[item] = i
        return rst
    else:
        rst = {}
        for i, item in enumerate(idx):
            rst[item] = i
        return rst


class Vertex:
    def __init__(self, vid):
        self.id = vid
        self.in_prox = set()
        self.out_prox = set()
        self.in_degree = 0
        self.out_degree = 0

    def add_out(self, vid):
        if vid not in self.out_prox:
            self.out_degree += 1
            self.out_prox.add(vid)

    def add_in(self, vid):
        if vid not in self.in_prox:
            self.in_degree += 1
            self.in_prox.add(vid)

    def weight(self):
        return self.in_degree + self.out_degree

    def is_connected(self):
        if len(self.in_prox.difference({self.id})) > 0:
            # if it has an in-neighbor other than itself
            return True
        if len(self.out_prox.difference({self.id})) > 0:
            # if it has an out-neighbor other than itself
            return True
        return False

    def __lt__(self, other):
        return self.out_degree < other.out_degree


class Graph:

    def __init__(self, filename, sep='\t', typ='dir', verbose=VERBOSE):
        self.vertices = []
        self.nVertices = 0
        self.nEdges = 0
        f = open(filename, 'r')
        s = f.readline()
        while len(s):
            pair = s.strip().split(sep)
            vid_in = int(pair[0])
            vid_out = int(pair[1])
            self.add_edge(vid_in, vid_out, typ)
            s = f.readline()
            if verbose:
                if not self.nEdges % N_EDGE_VERBOSE:
                    print('%d edges processed...' % self.nEdges)
        self.vid2newVid_mapping = {}
        self.newVid2vid_mapping = {}
        if verbose:
            print('Reducing empty vertices...')
        self._reduce(verbose)

    def add_vertex(self, vid):
        if self.nVertices > vid:
            if self.vertices[vid] is None:
                self.vertices[vid] = Vertex(vid)
        else:
            # self.vertices += [None] * (vid + 1 - self.nVertices)
            for _ in range(vid + 1 - self.nVertices):
                self.vertices.append(None)
            self.vertices[vid] = Vertex(vid)
            self.nVertices = vid + 1

    def add_edge(self, vid_in, vid_out, typ='dir'):
        # typ - 'dir' for directed, 'undir' for undirected
        self.add_vertex(vid_in)
        self.add_vertex(vid_out)
        self.vertices[vid_in].add_out(vid_out)
        self.vertices[vid_out].add_in(vid_in)
        self.nEdges += 1
        if typ == 'undir':
            self.add_edge(vid_out, vid_in, 'dir')

    def fetch_prox(self, vid, typ='out'):
        if typ == 'out':
            return self.vertices[vid].out_prox
        else:
            return self.vertices[vid].in_prox

    def calc_matrix_sparse(self, idx_row, idx_col, style=0, verbose=VERBOSE):
        if style == -1:
            return self.calc_matrix_sparse(idx_row, idx_col, 0), \
                   self.calc_matrix_sparse(idx_row, idx_col, 1)
        data = []
        indices = []
        current_ptr = 0
        indptr = [current_ptr]
        if style == 0:
            vid2colid_map = inverse_index(idx_col, self.nVertices, 'list')
            for row_id, vid in enumerate(idx_row):
                row = ddict(int)
                for first_neighbor in self.fetch_prox(vid, 'out'):
                    col_id_1 = vid2colid_map[first_neighbor]
                    if col_id_1 > 0:
                        row[col_id_1] += 1 / self.vertices[vid].out_degree
                    for second_neighbor in self.fetch_prox(first_neighbor, 'out'):
                        col_id_2 = vid2colid_map[second_neighbor]
                        if col_id_2 > 0:
                            row[col_id_2] += LAMBDA / (
                                self.vertices[vid].out_degree *
                                self.vertices[first_neighbor].out_degree
                            )
                data.extend(list(row.values()))
                indices.extend(list(row.keys()))
                current_ptr += len(row)
                indptr.append(current_ptr)
            return sp.csr_matrix((data, indices, indptr),
                                 shape=(len(idx_row), len(idx_col)))
        elif style == 1:
            vid2rowid_map = inverse_index(idx_row, self.nVertices, 'list')
            for col_id, vid in enumerate(idx_col):
                col = ddict(int)
                for first_neighbor in self.fetch_prox(vid, 'in'):
                    row_id_1 = vid2rowid_map[first_neighbor]
                    if row_id_1 > 0:
                        col[row_id_1] += 1 / self.vertices[first_neighbor].out_degree
                    for second_neighbor in self.fetch_prox(first_neighbor, 'in'):
                        row_id_2 = vid2rowid_map[second_neighbor]
                        if row_id_2 > 0:
                            col[row_id_2] += LAMBDA / (
                                self.vertices[second_neighbor].out_degree *
                                self.vertices[first_neighbor].out_degree
                            )
                data.extend(list(col.values()))
                indices.extend(list(col.keys()))
                current_ptr += len(col)
                indptr.append(current_ptr)
            return sp.csc_matrix((data, indices, indptr),
                                 shape=(len(idx_row), len(idx_col)))
        else:
            assert False, 'invalid style type provided.'

    def calc_matrix(self, idx_in, idx_out, verbose=VERBOSE):
        mat = np.zeros([len(idx_in), len(idx_out)])
        if len(idx_in) <= len(idx_out):
            vid2colid_map = inverse_index(idx_out, self.nVertices, 'dict')
            for row_id, vid in enumerate(idx_in):
                for first_neighbor in self.fetch_prox(vid, 'out'):
                    try:
                        col_id_1 = vid2colid_map[first_neighbor]
                        mat[row_id, col_id_1] += 1 / self.vertices[vid].out_degree
                    except KeyError:
                        pass
                    for second_neighbor in self.fetch_prox(first_neighbor, 'out'):
                        try:
                            col_id_2 = vid2colid_map[second_neighbor]
                            mat[row_id, col_id_2] += LAMBDA / (
                                self.vertices[vid].out_degree *
                                self.vertices[first_neighbor].out_degree
                            )
                        except KeyError:
                            pass
        else:
            vid2rowid_map = inverse_index(idx_in, self.nVertices, 'dict')
            for col_id, vid in enumerate(idx_out):
                for first_neighbor in self.fetch_prox(vid, 'in'):
                    try:
                        row_id_1 = vid2rowid_map[first_neighbor]
                        mat[row_id_1, col_id] += 1 / (
                            self.vertices[first_neighbor].out_degree
                        )
                    except KeyError:
                        pass
                    for second_neighbor in self.fetch_prox(first_neighbor, 'in'):
                        try:
                            row_id_2 = vid2rowid_map[second_neighbor]
                            mat[row_id_2, col_id] += LAMBDA / (
                                self.vertices[second_neighbor].out_degree *
                                self.vertices[first_neighbor].out_degree
                            )
                        except KeyError:
                            pass
        return mat

    def _set_reduce_mapping(self):
        top = 0
        for vid in range(self.nVertices):
            vertex = self.vertices[vid]
            if vertex is None:
                self.vid2newVid_mapping[vid] = 'NotExist'
            elif not vertex.is_connected():
                self.vid2newVid_mapping[vid] = 'Disconnected'
            else:
                self.vid2newVid_mapping[vid] = top
                self.newVid2vid_mapping[top] = vid
                top += 1

    def _reduce(self, verbose=True):
        self._set_reduce_mapping()
        clean_vertices = []
        for newVid, vid in self.newVid2vid_mapping.items():
            new_vertex = Vertex(newVid)
            for out_neighbor in self.fetch_prox(vid, 'out'):
                new_vertex.add_out(self.vid2newVid_mapping[out_neighbor])
            for in_neighbor in self.fetch_prox(vid, 'in'):
                new_vertex.add_in(self.vid2newVid_mapping[in_neighbor])
            clean_vertices.append(new_vertex)
            self.nEdges += new_vertex.out_degree
            if verbose:
                if not newVid % N_VERTEX_VERBOSE:
                    print('%d vertices reduced...' % newVid)
        self.vertices = clean_vertices
        self.nVertices = len(clean_vertices)


if __name__ == '__main__':
    import time
    from sample import sample
    pt = time.time()
    graph = Graph('data\\flickr\\links.txt')
    print('READ TIME: %.2f' % (time.time() - pt))

    k_set = sample(graph, 1000, 'uniform')

    pt = time.time()
    a = graph.calc_matrix(k_set, list(range(graph.nVertices)))
    print('ROW TIME: %.2f' % (time.time() - pt))
    del a

    pt = time.time()
    b = graph.calc_matrix(k_set, list(range(graph.nVertices)))
    print('COL TIME: %.2f' % (time.time() - pt))
    del b

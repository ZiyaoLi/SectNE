import numpy as np


LAMBDA = 0.5
VERBOSE = True
N_EDGE_VERBOSE = 5e6
N_VERTEX_VERBOSE = 5e5


def inverse_index(idx):
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
            self.vertices += [None] * (vid + 1 - self.nVertices)
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
            self.add_edge(vid_out, vid_in, 'undir')

    def fetch_prox(self, vid, typ='out'):
        if typ == 'out':
            return self.vertices[vid].out_prox
        else:
            return self.vertices[vid].in_prox

    def fetch_prox_mat(self, idx_in, idx_out):
        rst = np.zeros((len(idx_in), len(idx_out)))
        for row_id, vid_in in enumerate(idx_in):
            for col_id, vid_out in enumerate(idx_out):
                if vid_out in self.fetch_prox(vid_in, 'out'):
                    rst[row_id, col_id] = 1
        return rst

    def calc_matrix(self, idx_in, idx_out):
        prox_mat = self.fetch_prox_mat(idx_in, idx_out)
        secondary_mat = np.zeros([len(idx_in), len(idx_out)])
        if len(idx_in) <= len(idx_out):
            vid2colid_map = inverse_index(idx_out)
            for row_id, vid in enumerate(idx_in):
                for first_neighbor in self.fetch_prox(vid, 'out'):
                    for second_neighbor in self.fetch_prox(first_neighbor, 'out'):
                        try:
                            col_id = vid2colid_map[second_neighbor]
                            secondary_mat[row_id, col_id] += 1
                        except KeyError:
                            pass
        else:
            vid2rowid_map = inverse_index(idx_in)
            for col_id, vid in enumerate(idx_out):
                for first_neighbor in self.fetch_prox(vid, 'in'):
                    for second_neighbor in self.fetch_prox(first_neighbor, 'in'):
                        try:
                            row_id = vid2rowid_map[second_neighbor]
                            secondary_mat[row_id, col_id] += 1
                        except KeyError:
                            pass
        return prox_mat + LAMBDA * secondary_mat

    def _set_reduce_mapping(self):
        top = 0
        for vid in range(self.nVertices):
            if self.vertices[vid] is not None:
                self.vid2newVid_mapping[vid] = top
                self.newVid2vid_mapping[top] = vid
                top += 1

    def _reduce(self, verbose=True):
        self._set_reduce_mapping()
        clean_vertices = []
        for vid, newVid in self.vid2newVid_mapping.items():
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






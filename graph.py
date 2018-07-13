import numpy as np


class Proximity:

    def __init__(self):
        self.prox = set()
        self.degree = 0

    def __len__(self):
        return self.degree

    def __contains__(self, item):
        return item in self.prox

    def add(self, vid):
        self.degree += 1
        self.prox.add(vid)

    def __mul__(self, other):
        assert isinstance(other, Proximity)
        return len(self.prox.intersection(other.prox))


class VSet:

    def __init__(self, net, indices):
        self.num = len(indices)
        self.len = net.nVertices
        self.vertices = net.fetch_subset(indices)

    def __mul__(self, other):
        assert isinstance(other, VSet)
        rst = np.zeros((self.num, other.num))
        for i, vi in enumerate(self.vertices):
            for j, vj in enumerate(other.vertices):
                rst[i, j] = vi * vj
        return rst


class Graph:

    def __init__(self, filename, sep='\t', typ=0):
        self.vertices = []
        self.nVertices = 0
        self.nEdges = 0
        f = open(filename, 'r')
        s = f.readline()
        while len(s):
            pair = s.strip().split(sep)
            vid1 = int(pair[0])
            vid2 = int(pair[1])
            self.add_edge(vid1, vid2, typ)
            s = f.readline()

    def add_vertex(self, vid):
        if self.nVertices > vid:
            if self.vertices[vid] is None:
                self.vertices[vid] = Proximity()
        else:
            for t in range(vid + 1 - self.nVertices):
                self.vertices.append(None)
            self.vertices[vid] = Proximity()
            self.nVertices = vid + 1

    def add_edge(self, vinid, voutid, typ=0):
        # typ - 0 for direct, 1 for undirect
        self.add_vertex(vinid)
        self.add_vertex(voutid)
        self.vertices[vinid].add(voutid)
        self.nEdges += 1
        if typ == 1:
            self.vertices[voutid].add(vinid)
            self.nEdges += 1

    def fetch_subset(self, indices):
        rst = []
        for t in indices:
            rst.append(self.vertices[t])
        return rst

    def fetch_proxmat(self, indicesin, indicesout):
        rst = np.zeros((len(indicesin), len(indicesout)))
        for i, vinid in enumerate(indicesin):
            for j, voutid in enumerate(indicesout):
                if voutid in self.vertices[vinid]:
                    rst[i, j] = 1
        return rst

    def calc_matrix(self, indicesin, indicesout):
        rst = self.fetch_proxmat(indicesin, indicesout)
        rst += 0.5 * (VSet(self, indicesin) * VSet(self, indicesout))
        return rst

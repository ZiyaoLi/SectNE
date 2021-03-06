import graph
import random
import pandas as pd
import numpy as np

# Louvain Algorithm
# Vincent D. Blondel et al, Fast Unfolding of Communities in Large Networks, Phy.Rev.E, 2008

MAX_ITER = 10
MERGE = (1, 2000)


class Community:
    def __init__(self, comm_id):
        self.id = comm_id
        self.children = set()
        self.out_reaches = {}
        self.in_reaches = {}
        self.out_degree = 0
        self.in_degree = 0

    def to_metanode(self, in_prox, out_prox, vertices):
        return MetaNode((self, in_prox, out_prox, vertices))

    def get_inject(self, meta_id_other, typ):
        if typ == 'to':
            try:
                return self.out_reaches[meta_id_other]
            except KeyError:
                return 0
        else:
            try:
                return self.in_reaches[meta_id_other]
            except KeyError:
                return 0

    def add_inject(self, meta_id_other, amount, typ):
        if typ == 'to':
            try:
                self.out_reaches[meta_id_other] += amount
            except KeyError:
                self.out_reaches[meta_id_other] = amount
            self.out_degree += amount
        else:
            try:
                self.in_reaches[meta_id_other] += amount
            except KeyError:
                self.in_reaches[meta_id_other] = amount
            self.in_degree += amount

    def reduce_inject(self, meta_id_other, amount, typ):
        if typ == 'to':
            t = self.out_reaches[meta_id_other] - amount
            if t == 0:
                self.out_reaches.pop(meta_id_other)
            else:
                self.out_reaches[meta_id_other] = t
            self.out_degree -= amount
        else:
            t = self.in_reaches[meta_id_other] - amount
            if t == 0:
                self.in_reaches.pop(meta_id_other)
            else:
                self.in_reaches[meta_id_other] = t
            self.in_degree -= amount


class MetaNode:
    def __init__(self, param):
        if isinstance(param[0], Community):
            # initialize from a lower-level community
            self.id = param[0].id
            self.vertices = param[3]
            self.community = None
            self.in_degree = param[0].in_degree
            self.out_degree = param[0].out_degree
            self.in_prox = param[1]
            self.out_prox = param[2]
        elif isinstance(param[0], int):
            # initialize from an individual vertex
            self.id = param[0]
            self.vertices = [param[0]]
            self.community = None
            self.in_degree = param[1].in_degree
            self.out_degree = param[1].out_degree
            self.in_prox = dict([(t, 1) for t in param[1].in_prox])
            self.out_prox = dict([(t, 1) for t in param[1].out_prox])

    def get_prox(self, meta_id_other, typ):
        if typ == 'to':
            try:
                return self.out_prox[meta_id_other]
            except KeyError:
                return 0
        else:
            try:
                return self.in_prox[meta_id_other]
            except KeyError:
                return 0


class Louvain:
    def __init__(self, net, rand=True, verbose=True):
        self.meta_nodes = []  # the current community_ids of vertices
        self.volume = net.nEdges
        self.communities = []
        self.n_communities = 0
        self.rand = rand
        self.verbose = verbose
        for vid in range(net.nVertices):
            if verbose:
                if not vid % 5e5:
                    print('%d nodes read in Louvain...' % vid)
            vertex = net.vertices[vid]
            self.meta_nodes.append(MetaNode((vid, vertex)))

    def add_to_community(self, meta_id, comm_id):
        '''
        add one meta node to a community
        :param meta_id: the id of the meta node to be added
        :param comm_id: the id of the community id to be added to
        :return: None
        '''
        meta_node = self.meta_nodes[meta_id]
        assert meta_node.community is None
        comm = self.communities[comm_id]
        # identifying parent-child relationship
        meta_node.community = comm_id
        comm.children.add(meta_id)
        # adding injections
        for out_id, out_degree in meta_node.out_prox.items():
            comm.add_inject(out_id, out_degree, 'to')
        for in_id, in_degree in meta_node.in_prox.items():
            comm.add_inject(in_id, in_degree, 'from')

    def remove_from_community(self, meta_id):
        '''
        remove one meta node from its community
        :param meta_id: the id of the meta node to be removed
        :return: None
        '''
        meta_node = self.meta_nodes[meta_id]
        comm_id = meta_node.community
        assert comm_id is not None
        comm = self.communities[comm_id]
        # discarding parent-child relationship
        meta_node.community = None
        comm.children.remove(meta_id)
        # reducing injections
        for out_id, out_degree in meta_node.out_prox.items():
            comm.reduce_inject(out_id, out_degree, 'to')
        for in_id, in_degree in meta_node.in_prox.items():
            comm.reduce_inject(in_id, in_degree, 'from')

    def adjust_community(self, meta_id, comm_id):
        '''
        move one community to another
        :param meta_id: the to-be-moved meta node id
        :param comm_id: the to-be-moved-to community id
        :return: None
        '''
        # fetch previous parent
        comm_id_prev = self.meta_nodes[meta_id].community
        # if it has a parent, remove it from its parent first.
        if comm_id_prev is not None:
            self.remove_from_community(meta_id)
        self.add_to_community(meta_id, comm_id)

    def loss_modularity(self, meta_id):
        '''
        evaluate the loss of modularity removing a community from its parent.
        :formula:
            reducing all constant factors, the formula is
                2 x reach(c\i, i) - k_i x (k_{c\i}) / m
            =   2 x (reach(c, i) - reach(i, i)) - k_i x (k_c - k_i) / m
        :param meta_id: the community to be removed
        :return: loss of modularity
        '''
        meta_node = self.meta_nodes[meta_id]
        comm_id = meta_node.community
        rst = 0
        if comm_id is not None:
            comm = self.communities[comm_id]
            rst += (comm.get_inject(meta_id, 'to') - meta_node.get_prox(meta_id, 'to')) + \
                   (comm.get_inject(meta_id, 'from') - meta_node.get_prox(meta_id, 'from'))
            rst -= meta_node.out_degree * (comm.out_degree - meta_node.out_degree) / self.volume
        return rst

    def gain_modularity(self, meta_id, comm_id):
        '''
        evaluate the increase of modularity adding a meta node to a community.
        :formula:
            reducing all constant factors, the formula is
                2 x reach(c, i) - k_i x (k_c) / m
        :param meta_id: the meta node to be added
        :param comm_id: the community to be added to
        :return: gain of modularity
        '''
        meta_node = self.meta_nodes[meta_id]
        comm = self.communities[comm_id]
        rst = comm.get_inject(meta_id, 'to') + comm.get_inject(meta_id, 'from')
        rst -= meta_node.out_degree * comm.out_degree / self.volume
        return rst

    def form_modularity(self, meta_id_i, meta_id_j):
        meta_node_i = self.meta_nodes[meta_id_i]
        meta_node_j = self.meta_nodes[meta_id_j]
        return meta_node_i.get_prox(meta_id_j, 'to') + \
            meta_node_i.get_prox(meta_id_j, 'from') - \
            meta_node_i.out_degree * meta_node_j.out_degree / self.volume

    def add_community(self):
        t = self.n_communities
        self.communities.append(Community(self.n_communities))
        self.n_communities += 1
        return t

    def proximity_between_community(self, comm_id, reduce_mapping):
        comm = self.communities[comm_id]
        in_prox = {}
        out_prox = {}
        for meta_id, weight in comm.out_reaches.items():
            comm_id_to = reduce_mapping[self.meta_nodes[meta_id].community]
            try:
                out_prox[comm_id_to] += weight
            except KeyError:
                out_prox[comm_id_to] = weight
        for meta_id, weight in comm.in_reaches.items():
            comm_id_from = reduce_mapping[self.meta_nodes[meta_id].community]
            try:
                in_prox[comm_id_from] += weight
            except KeyError:
                in_prox[comm_id_from] = weight
        return in_prox, out_prox

    def vertices_union(self, comm_id):
        comm = self.communities[comm_id]
        vertices = []
        for meta_id in comm.children:
            vertices += self.meta_nodes[meta_id].vertices
        return vertices

    def first_stage(self):
        # whether there is ops in this function
        amount_increase = 0
        visit_sequence = list(range(len(self.meta_nodes)))
        ite = 0
        while True:
            ite += 1
            # whether there is ops in this loop
            amount_loop_increase = 0
            if self.rand:
                # shuffle the sequence
                random.shuffle(visit_sequence)
            for meta_id_i in visit_sequence:
                meta_node_i = self.meta_nodes[meta_id_i]
                comm_id_i = meta_node_i.community  # current community of node i
                tried_communities = set()
                if comm_id_i is not None:
                    tried_communities.add(comm_id_i)

                max_increase = 0
                max_increase_comm_id = -1
                max_increase_meta_id = -1

                meta_candidates = set(meta_node_i.in_prox.keys()).\
                    union(set(meta_node_i.out_prox.keys()))
                for meta_id_j in meta_candidates:
                    if meta_id_j == meta_id_i:
                        # self-adjustment forbidden
                        continue
                    # iterate through i node's neighbors to find a new community to join
                    meta_node_j = self.meta_nodes[meta_id_j]
                    comm_id_j = meta_node_j.community
                    if comm_id_j is None:
                        # which indicates that j is not assigned to communities yet
                        # try building one community for these two nodes to see the improvement
                        delta_q = self.form_modularity(meta_id_i, meta_id_j) - \
                            self.loss_modularity(meta_id_i)
                        if delta_q > max_increase:
                            max_increase = delta_q
                            max_increase_comm_id = -1
                            max_increase_meta_id = meta_id_j
                    elif comm_id_j not in tried_communities:
                        delta_q = self.gain_modularity(meta_id_i, comm_id_j) - \
                            self.loss_modularity(meta_id_i)
                        tried_communities.add(comm_id_j)
                        if delta_q > max_increase:
                            max_increase = delta_q
                            max_increase_comm_id = comm_id_j

                if max_increase > 0:
                    # some adjustments shall happen (for node i) under this condition
                    # otherwise, just go on looping
                    # first, modify the signs
                    amount_loop_increase += max_increase
                    amount_increase += max_increase
                    # then, do appropriate adjustments
                    if max_increase_comm_id >= 0:
                        # found a current community to join
                        self.adjust_community(meta_id_i, max_increase_comm_id)
                    elif max_increase_meta_id >= 0:
                        # found a new community to join
                        # build a new community of these two nodes (i, j)
                        new_comm_id = self.add_community()
                        self.add_to_community(max_increase_meta_id, new_comm_id)
                        self.adjust_community(meta_id_i, new_comm_id)
                elif comm_id_i is None:
                    # no assigned community found, but node i has no community
                    # build a new community for this node i
                    new_comm_id = self.add_community()
                    self.add_to_community(meta_id_i, new_comm_id)

            if self.verbose:
                print('Looping %d of the first stage. increase=%.4f' % (ite, amount_loop_increase))

            if not amount_loop_increase:
                # indicates no ops in this iteration
                break

        return amount_increase

    def second_stage(self):
        current_id = 0
        comm_id_mapping = {}
        # construct a mapping that reduces empty communities
        for comm_id in range(self.n_communities):
            comm = self.communities[comm_id]
            if len(comm.children) == 0:
                continue
            else:
                comm.id = current_id
                comm_id_mapping[comm_id] = current_id
                current_id += 1

        meta_nodes = []
        for comm_id in range(self.n_communities):
            comm = self.communities[comm_id]
            if len(comm.children) == 0:
                continue
            else:
                in_prox, out_prox = self.proximity_between_community(comm_id, comm_id_mapping)
                vertices_union = self.vertices_union(comm_id)
                meta_nodes.append(comm.to_metanode(in_prox, out_prox, vertices_union))
        self.meta_nodes = meta_nodes
        self.communities = []
        self.n_communities = 0

    def get_groups(self, merge=MERGE):
        groups = []
        merged_groups = []
        for meta_node in self.meta_nodes:
            if len(meta_node.vertices) < merge[0]:
                merged_groups.extend(meta_node.vertices)
                if len(merged_groups) > merge[1]:
                    groups.append(merged_groups)
                    merged_groups = []
            elif len(meta_node.vertices) > merge[1]:
                long_group = meta_node.vertices
                random.shuffle(long_group)
                while len(long_group):
                    groups.append(long_group[:merge[1]])
                    long_group = long_group[merge[1]:]
            else:
                groups.append(meta_node.vertices)
        groups.append(merged_groups)
        return groups

    def execute(self, max_iter=MAX_ITER, merge=MERGE):
        ite = 0
        while ite < max_iter:
            ite += 1
            amount_increase = self.first_stage()
            if self.verbose:
                print("%d iteration finished. first stage increase:%.4f"
                      % (ite, amount_increase))
            if amount_increase > 0:
                self.second_stage()
            else:
                break
        return self.get_groups(merge=merge)


def groups2inv_index(groups, n_vertices, override_set=set()):
    if not isinstance(override_set, set):
        override_set = set(override_set)
    inverse_index = [-1] * n_vertices
    for i in override_set:
        inverse_index[i] = 0
    for group_id, group in enumerate(groups):
        for i in group:
            if inverse_index[i] < 0:
                inverse_index[i] = group_id + 1
    return inverse_index


def pure_override_nodes(groups, inv_index):
    for group_id, group in enumerate(groups):
        t = 0
        while t < len(group):
            if inv_index[group[t]] == 0:
                group.remove(group[t])
            else:
                t += 1


if __name__ == '__main__':
    G = graph.Graph('data\\wiki\\links.txt', sep='\t', typ=1)
    algorithm = Louvain(G)
    communities = algorithm.execute()
    for c in communities:
        print(c)
    print("--------------------")
    print(pd.value_counts([len(t) for t in communities]))

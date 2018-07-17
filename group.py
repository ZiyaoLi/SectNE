import graph
import random
import numpy as np

# Louvain Algorithm
# Vincent D. Blondel et al, Fast Unfolding of Communities in Large Networks, Phy.Rev.E, 2008


class Community:
    def __init__(self, comm_id):
        self.id = comm_id
        self.children = set()
        self.out_reaches = {}
        self.degree = 0

    def to_metanode(self, proximity, vertices):
        return MetaNode((self, proximity, vertices))

    def get_inject(self, meta_id_other):
        try:
            return self.out_reaches[meta_id_other]
        except KeyError:
            return 0

    def add_inject(self, meta_id_other, amount):
        try:
            self.out_reaches[meta_id_other] += amount
        except KeyError:
            self.out_reaches[meta_id_other] = amount
        self.degree += amount

    def reduce_inject(self, meta_id_other, amount):
        t = self.out_reaches[meta_id_other] - amount
        if t == 0:
            self.out_reaches.pop(meta_id_other)
        else:
            self.out_reaches[meta_id_other] = t
        self.degree -= amount


class MetaNode:
    def __init__(self, param):
        if isinstance(param[0], Community):
            # initialize from a lower-level community
            self.id = param[0].id
            self.vertices = param[2]
            self.community = None
            self.degree = param[0].degree
            self.prox = param[1]
        else:
            # initialize from an individual vertex
            self.id = param[0]
            self.vertices = [param[0]]
            self.community = None
            self.degree = len(param[1])
            self.prox = dict([(t, 1) for t in param[1]])

    def get_prox(self, meta_id_other):
        try:
            return self.prox[meta_id_other]
        except KeyError:
            return 0


class Louvain:
    def __init__(self, net):
        self.meta_nodes = []  # the current community_ids of vertices
        self.volume = net.nEdges
        self.communities = []
        self.n_communities = 0
        for vid in range(net.nVertices):
            self.meta_nodes.append(MetaNode((vid, net.fetch_prox(vid))))

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
        for out_id, out_degree in meta_node.prox.items():
            comm.add_inject(out_id, out_degree)

    def remove_from_community(self, meta_id):
        '''
        remove one meta node from its community
        :param meta_id: the id of the meta node to be removed
        :return: None
        '''
        meta_node = self.meta_nodes[meta_id]
        assert meta_node.community is not None
        comm_id = meta_node.community
        comm = self.communities[comm_id]
        # discarding parent-child relationship
        meta_node.community = None
        comm.children.remove(meta_id)
        # re-calculating parent's proximity
        for out_id, out_degree in meta_node.prox.items():
            comm.reduce_inject(out_id, out_degree)

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
            rst += 2 * (comm.get_inject(meta_id) - meta_node.get_prox(meta_id))
            rst -= meta_node.degree * (comm.degree - meta_node.degree) / self.volume
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
        rst = 2 * comm.get_inject(meta_id)
        rst -= meta_node.degree * comm.degree / self.volume
        return rst

    def form_modularity(self, meta_id_i, meta_id_j):
        meta_node_i = self.meta_nodes[meta_id_i]
        meta_node_j = self.meta_nodes[meta_id_j]
        return 2 * meta_node_i.get_prox(meta_id_j) - \
            meta_node_i.degree * meta_node_j.degree / self.volume

    def add_community(self):
        t = self.n_communities
        self.communities.append(Community(self.n_communities))
        self.n_communities += 1
        return t

    def proximity_between_community(self, comm_id, reduce_mapping):
        comm = self.communities[comm_id]
        proximity = {}
        for meta_id, weight in comm.out_reaches.items():
            comm_id_to = reduce_mapping[self.meta_nodes[meta_id].community]
            try:
                proximity[comm_id_to] += weight
            except KeyError:
                proximity[comm_id_to] = weight
        return proximity

    def vertices_union(self, comm_id):
        comm = self.communities[comm_id]
        vertices = []
        for meta_id in comm.children:
            vertices += self.meta_nodes[meta_id].vertices
        return vertices

    def first_stage(self):
        # whether there is ops in this function
        sign_increase = False
        visit_sequence = list(range(len(self.meta_nodes)))
        while True:
            # whether there is ops in this loop
            sign_loop_increase = False
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

                for meta_id_j in meta_node_i.prox.keys():
                    if meta_id_j == meta_id_i:
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
                    sign_loop_increase = True
                    sign_increase = True
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

            if not sign_loop_increase:
                # indicates no ops in this iteration
                break

        return sign_increase

    def second_stage(self):
        current_id = 0
        comm_id_mapping = {}
        for comm_id in range(self.n_communities):
            comm = self.communities[comm_id]
            if comm.degree == 0:
                continue
            else:
                comm.id = current_id
                comm_id_mapping[comm_id] = current_id
                current_id += 1

        meta_nodes = []
        for comm_id in range(self.n_communities):
            comm = self.communities[comm_id]
            if comm.degree == 0:
                continue
            else:
                proximity = self.proximity_between_community(comm_id, comm_id_mapping)
                vertices_union = self.vertices_union(comm_id)
                meta_nodes.append(comm.to_metanode(proximity, vertices_union))
        self.meta_nodes = meta_nodes
        self.communities = []
        self.n_communities = 0

    def get_groups(self):
        groups = []
        for meta_node in self.meta_nodes:
            groups.append(meta_node.vertices)
        return groups

    def execute(self):
        ite = 0
        while True:
            ite += 1
            sign_increase = self.first_stage()
            if sign_increase:
                self.second_stage()
            else:
                break
        return self.get_groups()


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
    G = graph.Graph('wiki.txt', sep='\t', typ=1)
    algorithm = Louvain(G)
    communities = algorithm.execute()
    for c in communities:
        print(c)

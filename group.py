import graph
import random
import numpy as np

# Louvain Algorithm
# Vincent D. Blondel et al, Fast Unfolding of Communities in Large Networks, Phy.Rev.E, 2008


class Community:
    def __init__(self, vid, v_prox):
        self.vertices = {vid}
        self.total_links = 0
        self.total_degree = len(v_prox)
        self.n_vertices = 1

    def add(self, vid, v_prox):
        if isinstance(v_prox, set):
            self.vertices.add(vid)
            self.total_links += self.eval_inject(v_prox)
            self.total_degree += len(v_prox)
            self.n_vertices += 1

    def remove(self, vid, v_prox):
        self.vertices.remove(vid)
        self.total_links -= self.eval_inject(v_prox)
        self.total_degree -= len(v_prox)
        self.n_vertices -= 1

    def eval_inject(self, prox):
        '''
        evaluate the injection amount from (a node with)
            a given proximity set. for undirected graphs,
            this injection amount equals to injection amount
            from a community to a node.
        :param prox: proximity set
        :return: evaluated value
        '''
        return len(self.vertices.intersection(prox))


class Louvain:
    def __init__(self, net):
        self.net = net
        self.communities = []  # a list of communities
        self.vertex_cids = []  # the current community_ids of vertices
        for vid in range(self.net.nVertices):
            self.communities.append(Community(vid, self.net.fetch_prox(vid)))
            self.vertex_cids.append(vid)

    def adjust_community(self, vid, cid, v_prox):
        '''
        move a vertex (vid) to a community (cid)
        :param vid: vertex
        :param cid: community
        :return: None
        '''
        cid_prev = self.vertex_cids[vid]
        self.vertex_cids[vid] = cid
        self.communities[cid_prev].remove(vid, v_prox)
        self.communities[cid].add(vid, v_prox)
        # clear out empty communities
        if self.communities[cid_prev].n_vertices == 0:
            self.communities[cid_prev] = None

    def loss_modularity(self, cid, prox):
        '''
        evaluate the loss of modularity if remove a node
            with given proximity set from the community.
        :formula: (ignore the constants)
            v = n_links_thisNode2communityNodes +
                n_links_communityNodes2thisNode +
                -2 x (degree_thisNode x (degree_communityTotal - degree_thisNode))
            if undirected, this reduces to
            v = n_links_between_thisNode_communityNodes -
                degree_thisNode x (degree_communityTotal - degree_thisNode)
        :param cid: comm
        :param prox: proximity set
        :return: evaluated value
        '''
        comm = self.communities[cid]
        return comm.eval_inject(prox) - \
            len(prox) * (comm.total_degree - len(prox)) / self.net.nEdges / 2

    def gain_modularity(self, cid, prox):
        '''
        very similar with loss_modularity().
        '''
        comm = self.communities[cid]
        return comm.eval_inject(prox) - \
            len(prox) * comm.total_degree / self.net.nEdges / 2

    def first_stage(self):
        # whether there is ops in this function
        sign_increase = False
        visit_sequence = list(range(self.net.nVertices))
        while True:
            # whether there is ops in this loop
            sign_loop_increase = False
            # shuffle the sequence
            # random.shuffle(visit_sequence)
            for i_vid in visit_sequence:
                i_cid = self.vertex_cids[i_vid]
                i_prox = self.net.fetch_prox(i_vid)
                tried_cids = {i_cid}

                max_increase = 0
                max_increase_cid = -1

                for j_vid in self.net.fetch_prox(i_vid):
                    j_cid = self.vertex_cids[j_vid]
                    if j_cid in tried_cids:
                        continue
                    else:
                        delta_q = self.gain_modularity(j_cid, i_prox) - \
                            self.loss_modularity(i_cid, i_prox)
                        tried_cids.add(j_cid)
                        if delta_q > max_increase:
                            max_increase = delta_q
                            max_increase_cid = j_cid

                if max_increase > 0:
                    self.adjust_community(i_vid, max_increase_cid, i_prox)
                    # modify the signs
                    sign_loop_increase = True
                    sign_increase = True

            if not sign_loop_increase:
                # indicates no ops in this iteration
                break

        return sign_increase

    def second_stage(self):
        cid_vertices = {}
        vid_vertex = {}
        for cid,vertices in self.communities.items():
            if len(vertices) == 0:
                continue
            
            new_vertex = Vertex(cid, cid, set())
            for vid in vertices:
                new_vertex.nodes.update(self.vertex_cids[vid]._nodes)
                new_vertex.kin += self.vertex_cids[vid]._kin
                if vid >= self.net.nVertices or self.net.vertices[vid] is None:
                    continue
                for k,v in self.net.vertices[vid].weights.items() :
                    if k in vertices:
                        new_vertex.kin += v / 2.0
            
            cid_vertices[cid] = set([cid])
            vid_vertex[cid] = new_vertex    

        G = graph.Graph() 
        for cid1,vertices1 in self.communities.items():
            if len(vertices1) == 0:
                continue
            for cid2,vertices2 in self.communities.items():
                if cid2<=cid1 or len(vertices2)==0:
                    continue
                edge_weight = 0.0
                for vid in vertices1:
                    if vid >= self.net.nVertices or self.net.vertices[vid] is None:
                        continue
                    for k,v in self.net.vertices[vid].weights.items():
                        if k in vertices2:
                            edge_weight += v                
                if edge_weight != 0:
                    G.add_edge(cid1,cid2,typ=1,w=edge_weight)    

        self.communities = cid_vertices
        self.vertex_cids = vid_vertex
        self.net = G

    def get_communities(self):
        communities = []
        for vertices in self.communities.values():
            if len(vertices) != 0:
                c = set()
                for vid in vertices:
                    c.update(self.vertex_cids[vid]._nodes)
                communities.append(c)
        return communities

    def execute(self):
        ite = 1
        while True:
            ite += 1
            mod_inc = self.first_stage()
            if mod_inc:
                self.second_stage()
            else:
                break
        return self.get_communities()


# 把所有一个节点的社团合并?
# 先分群,再剔除抽出的k个节点?

def group(G, ksample):
    algorithm = Louvain(G)
    communities = algorithm.execute()
    result = []
    one = []
    result.append(ksample)
    kset = set(ksample)
    for c in communities:
        c = c - kset
        clist = list(c)
        if len(clist) > 1:
            result.append(clist)
        elif len(clist) == 1:
            one.append(clist[0])
    result.append(one)
    return result

if __name__ == '__main__':
    G = graph.Graph('.\wiki.txt', sep='\t', typ=1)
    algorithm = Louvain(G)
    communities = algorithm.execute()
    for c in communities:
        print(c)

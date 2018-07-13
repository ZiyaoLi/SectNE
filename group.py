import graph
import random

# Louvain Algorithm
# Vincent D. Blondel et al, Fast Unfolding of Communities in Large Networks, Phy.Rev.E, 2008


class Vertex:
    def __init__(self, vid, cid, nodes, k_in=0):
        self._vid = vid  # node id
        self._cid = cid  # community id
        self._nodes = nodes
        self._kin = k_in  #结点内部的边的权重


class Louvain:
    def __init__(self, net):
        self.net = net
        self.nEdge = net.nEdges
        self._cid_vertices = {} #需维护的关于社区的信息(社区编号,其中包含的结点编号的集合)
        self._vid_vertex = {}   #需维护的关于结点的信息(结点编号，相应的Vertex实例)
        for vid in self.net.get_vertices():
            self._cid_vertices[vid] = set([vid])
            self._vid_vertex[vid] = Vertex(vid, vid, set([vid]))        

    def first_stage(self):
        mod_inc = False  #用于判断算法是否可终止
        visit_sequence = self.net.get_vertices()
        random.shuffle(visit_sequence)        
        while True:
            can_stop = True #第一阶段是否可终止
            for v_vid in visit_sequence:
                v_cid = self._vid_vertex[v_vid]._cid
                k_v = sum(self.net.vertices[v_vid].weights.values()) + self._vid_vertex[v_vid]._kin

                cid_Q = {}
                for w_vid in self.net.vertices[v_vid].prox:
                    w_cid = self._vid_vertex[w_vid]._cid
                    if w_cid in cid_Q:
                        continue
                    else:
                        tot = sum([sum(self.net.vertices[k].weights.values()) + self._vid_vertex[k]._kin for k in self._cid_vertices[w_cid]])
                        if w_cid == v_cid:
                            tot -= k_v
                        k_v_in = sum([v for k,v in self.net.vertices[v_vid].weights.items() if k in self._cid_vertices[w_cid]])
                        delta_Q = k_v_in - k_v * tot / self.nEdge  #由于只需要知道delta_Q的正负，所以少乘了1/(2*self._m)
                        cid_Q[w_cid] = delta_Q                    

                cid,max_delta_Q = sorted(cid_Q.items(),key=lambda item:item[1],reverse=True)[0]
                if max_delta_Q > 0.0 and cid!=v_cid:
                    self._vid_vertex[v_vid]._cid = cid
                    self._cid_vertices[cid].add(v_vid)
                    self._cid_vertices[v_cid].remove(v_vid)
                    can_stop = False
                    mod_inc = True            
            if can_stop:
                break        
        return mod_inc

    def second_stage(self):
        cid_vertices = {}
        vid_vertex = {}
        for cid,vertices in self._cid_vertices.items():
            if len(vertices) == 0:
                continue
            
            new_vertex = Vertex(cid, cid, set())
            for vid in vertices:
                new_vertex._nodes.update(self._vid_vertex[vid]._nodes)
                new_vertex._kin += self._vid_vertex[vid]._kin
                if vid >= self.net.nVertices or self.net.vertices[vid] is None:
                    continue
                for k,v in self.net.vertices[vid].weights.items() :
                    if k in vertices:
                        new_vertex._kin += v/2.0
            
            cid_vertices[cid] = set([cid])
            vid_vertex[cid] = new_vertex    

        G = graph.Graph() 
        for cid1,vertices1 in self._cid_vertices.items():
            if len(vertices1) == 0:
                continue
            for cid2,vertices2 in self._cid_vertices.items():
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

        self._cid_vertices = cid_vertices
        self._vid_vertex = vid_vertex
        self.net = G

    def get_communities(self):
        communities = []
        for vertices in self._cid_vertices.values():
            if len(vertices) != 0:
                c = set()
                for vid in vertices:
                    c.update(self._vid_vertex[vid]._nodes)
                communities.append(c)
        return communities

    def execute(self):
        iter_time = 1
        while True:
            iter_time += 1
            mod_inc = self.first_stage()
            if mod_inc:
                self.second_stage()
            else:
                break
        return self.get_communities()



# 把所有一个节点的社团合并?
# 先分群,再剔除抽出的k个节点?

def Group (G, ksample) :    
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
    G = graph.Graph('.\graph.txt', sep=' ', typ=1)
    algorithm = Louvain(G)
    communities = algorithm.execute()
    for c in communities:
        print(c)
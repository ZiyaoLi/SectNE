
# W'C = M00
# ib -> \bar{i}
# lam -> lambda
# min{ (M0i-M00 B)^2+(Mi0-A M00)^2+(Mii-A' M00 B)^2 +
#      lam*{(M0ib Miib'-M0ib M0ib' A)^2+( Mib0' Mibi-Mibo' Mib0 B)^2} + eta*(A^2+B^2) }
# 继续优化,仅计算对角线即可,按列向量计算减小复杂度

import numpy as np


def CGD(initx, A, grad, maxiter):

    r = -grad
    if r.all()==0 : ##
        return initx ##
    p = r
    epsilon = 1e-6
    x = initx
    for i in range(maxiter) :
        r2 = np.dot(r.T,r)
        Ap = np.dot(A,p)
        alpha = r2/np.dot(p.T,Ap)
        x = x + np.dot(p,np.diag(np.diag(alpha)))
        r = r - np.dot(Ap,np.diag(np.diag(alpha)))
        # print(np.linalg.norm(r))
        if np.linalg.norm(r) <= epsilon :
            break
        beta = np.dot(r.T,r)/r2
        p = r + np.dot(p,np.diag(np.diag(beta)))
    return x

class optimize :

    def __init__ (self, g, sep, lam=0.5, eta=0.3, maxiter=10) :
        self.graph = g
        self.sep = sep
        self.M00 = g.calc_matrix(sep[0],sep[0])
        u,s,v = np.linalg.svd(self.M00)
        self.wt = np.dot(u,np.diag(np.sqrt(s)))
        self.lam = lam
        self.eta = eta
        self.maxiter = maxiter

    def get_ib (self, index) :
        ib = []
        for i in range(1,len(self.sep)) :
            if i == index :
                continue
            ib += self.sep[i]
        return ib

    def train (self, index, embeddings) :
        M0i = self.graph.calc_matrix(self.sep[0],self.sep[index])
        Mii = self.graph.calc_matrix(self.sep[index],self.sep[index])
        ib = self.get_ib(index)
        M0ib = self.graph.calc_matrix(self.sep[0],ib)
        Miib = self.graph.calc_matrix(self.sep[index],ib)
        #init
        k = len(self.sep[0])
        mi = len(self.sep[index])
        A = np.ones((k,mi)) ##
        B = -np.ones((k,mi)) ##
        for ii in range(self.maxiter) :
            # fix B update A
            MB = np.dot(self.M00,B)
            M2 = np.dot(M0ib,M0ib.T)
            # Hessian
            HA = np.dot(self.M00,self.M00)+np.dot(MB,MB.T)+self.lam*np.dot(M2,M2)+self.eta
            bA = np.dot(self.M00,M0i)+np.dot(MB,Mii)+self.lam*np.dot(M2,np.dot(M0ib,Miib.T)) 
            GradA = np.dot(HA,A)-bA
            A = CGD(A, HA, GradA, self.maxiter)

            # fix A update B
            MA = np.dot(self.M00,A)
            MT = np.dot(M0ib,M0ib.T)
            # Hessian
            HB = np.dot(self.M00,self.M00)+np.dot(MA,MA.T)+self.lam*np.dot(MT,MT)+self.eta
            bB = np.dot(self.M00,M0i)+np.dot(MA,Mii)+self.lam*np.dot(MT,np.dot(M0ib,Miib.T))   
            GradB = np.dot(HB,B)-bB
            B = CGD(B, HB, GradB, self.maxiter)

        w = np.dot(self.wt.T,A) 
        for i,v in enumerate(self.sep[index]) :
            embeddings[v]=(w.T)[i].tolist()
        
    def get_embeddings (self) :
        embeddings = {}
        for i, v in enumerate(self.sep[0]) :
            embeddings[v] = self.wt[i].tolist()
        print("{} Blocks in All".format(len(self.sep)))
        for index in range(1,len(self.sep)) :
            self.train(index, embeddings)
            print("Block {} Finished!".format(index))
        return embeddings
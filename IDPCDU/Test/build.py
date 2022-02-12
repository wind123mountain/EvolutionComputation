import numpy as np
from queue import PriorityQueue
import copy

class GraphDomain:
    def __init__(self, path, name = 'NULL', pad = 1):
        self.NAME = name
        self.NUM_NODE = 0
        self.NUM_DOMAIN = 0
        self.START_NODE = -1
        self.END_NODE = -1

        self.adj = {}
        self.distance = {}
        self.pre_node = {}
        self.domain_start_nodes = {}
        
        self.pad = pad
        self.load_data(path)
        self.build_graph()

        self.best = np.inf
 
        
    def load_data(self, path):
        f = open(path, 'r')

        l1 = f.readline()
        l1 = l1.strip()
        l1 = l1.split(' ')
        self.NUM_NODE = int(l1[0])
        self.NUM_DOMAIN = int(l1[1])

        l2 = f.readline()
        l2 = l2.strip()
        l2 = l2.split(' ')
        self.START_NODE = int(l2[0])
        self.END_NODE = int(l2[1])

        # fomat edge: start, end, weight, domain
        edges = f.readlines()

        f.close()

        #setup
        for d in range(self.pad, self.pad+self.NUM_DOMAIN):
            self.adj[d] = {}
            self.distance[d] = {}
            self.domain_start_nodes[d] = []
            self.pre_node[d] = {}
            for v in range(self.pad, self.pad+self.NUM_NODE):
                self.adj[d][v] = list()
                self.distance[d][v] = np.full((self.NUM_NODE+self.pad,), np.inf)
                self.pre_node[d][v] = np.full((self.NUM_NODE+self.pad,), -1)
        
        for e in edges:
            e = e.strip()
            e = e.split(' ')
            #u = e[0], v = e[1], w = e[2], d = e[3]
            self.adj[int(e[3])][int(e[0])].append((int(e[1]), int(e[2])))
            
        

    def Dijkstra(self, domain, start, to_show=False):
        if(to_show):
            dis = np.full((self.NUM_NODE+self.pad,), np.inf)
        else:
            dis = self.distance[domain][start]
        # pre = np.full((self.NUM_NODE+self.pad,), -1)
        pre = self.pre_node[domain][start]
        # dis[start] = 0
        pre[start] = start

        PQ = PriorityQueue()
        # PQ.put((dis[start], start))
        PQ.put((0, start))

        while not PQ.empty():
            curr = PQ.get()
            u = curr[1]
            d = curr[0]

            for e in self.adj[domain][u]:
                if e[1] + d < dis[e[0]]:
                    dis[e[0]] = e[1] + d
                    pre[e[0]] = u
                    PQ.put((dis[e[0]], e[0]))
        
        # return pre
    

    def build_graph(self):
        for d in range(self.pad, self.NUM_DOMAIN+self.pad):
            for v in range(self.pad, self.NUM_NODE+self.pad):
                self.Dijkstra(d, v)
                if len(self.adj[d][v]) != 0:
                    self.domain_start_nodes[d].append(v)

    def Decode(self, indiv):
        if len(indiv) == self.NUM_DOMAIN: return indiv
        result = []
        for i in indiv:
            if i <= self.NUM_DOMAIN:
                result.append(i)
        return np.array(result)
    
    def Cost(self, indiv):
        domains = self.Decode(indiv)
        res = np.inf
        index = 0

        # pre_dis = np.full((self.NUM_NODE+1,), np.inf)
        # pre_dis[self.START_NODE] = 0
        pre_dis = copy.copy(self.distance[domains[0]][self.START_NODE])
        # pre_dis[self.START_NODE] = np.inf
        res = pre_dis[self.END_NODE]

        id = 2
        for d in domains[1:]:
            dis = np.full((self.NUM_NODE+1,), np.inf)
            
            for u in self.domain_start_nodes[d]:
                if (pre_dis[u] != np.inf):             
                    for i in range(1, 1+self.NUM_NODE):
                        if pre_dis[u] + self.distance[d][u][i] < dis[i]:
                            dis[i] = pre_dis[u] + self.distance[d][u][i]

            if res > dis[self.END_NODE]:
                res = dis[self.END_NODE]
                index = id
            if id > 2:
                tmp_best = min(dis)
                if tmp_best > self.best:
                    break

            pre_dis = dis
            id += 1

        self.best = min(self.best, res)
        return res, id
        

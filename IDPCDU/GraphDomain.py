import numpy as np
from queue import PriorityQueue

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
        dis[start] = 0
        pre[start] = start

        PQ = PriorityQueue()
        PQ.put((dis[start], start))

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



###################################### Version 2 #########################################
#                                   Encode Priority                                      #

class EncodePriority(GraphDomain):

    def Decode(self, indiv):
        if len(indiv) == self.NUM_DOMAIN: return indiv
        result = []
        for i in indiv:
            if i <= self.NUM_DOMAIN:
                result.append(i)
        return np.array(result)


    def Cost(self, indiv):
        indiv = self.Decode(indiv)
        domains = np.argsort(-np.array(indiv)) + 1

        dis = np.full((self.NUM_NODE+1,), np.inf)
        dis[self.START_NODE] = 0

        for d in domains:
            for u in self.domain_start_nodes[d]:
                if (dis[u] != np.inf):             
                    for i in range(1, 1+self.NUM_NODE):
                        if dis[u] + self.distance[d][u][i] < dis[i]:
                            dis[i] = dis[u] + self.distance[d][u][i]
        
        return dis[self.END_NODE]


    def show_path(self, indiv):
        # find path                  
        indiv = self.Decode(indiv)
        domains = np.argsort(-np.array(indiv)) + 1

        dis = np.full((self.NUM_NODE+1,), np.inf)
        end_domain = np.full((self.NUM_NODE+1,), -1)            #end domain of shortest path to the node
        pre = np.full((self.NUM_DOMAIN+1, self.NUM_NODE+1, 2), -1)
        dis[self.START_NODE] = 0
        end_domain[self.START_NODE] = 0

        for d in domains:
            for u in self.domain_start_nodes[d]:
                if (dis[u] != np.inf):             
                    for i in range(1, 1+self.NUM_NODE):
                        if dis[u] + self.distance[d][u][i] < dis[i]:
                            dis[i] = dis[u] + self.distance[d][u][i]
                            end_domain[i] = d
                            pre[d][i][0] = end_domain[u]
                            pre[d][i][1] = u

        # show path
        res = []
        curr = (end_domain[self.END_NODE], self.END_NODE)

        while curr[1] != self.START_NODE:
            domain = curr[0]
            start_node = pre[curr[0]][curr[1]][1]
            path = []

            pre_node_in_domain = self.pre_node[domain][start_node]
            curr_node = curr[1]
            while(pre_node_in_domain[curr_node] != curr_node):
                path.append(curr_node)
                curr_node = pre_node_in_domain[curr_node]
            path.append(start_node)

            res.append({domain: path[::-1]})

            curr = pre[curr[0]][curr[1]]
            
        if(dis[self.END_NODE] != np.inf):
            print(res)
        else:
            print('Error path: Not find path to end node!')
        return res, dis[self.END_NODE]
        
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



class AlgorithmV1(GraphDomain):


##################################### Version 1 ########################################
#                                 Encoding Version 1                                   #
# Gene has  two parts:
#    Priority of the Domain
#    End node of the domain

    def Decode(self, indiv):
        if len(indiv[0]) == self.NUM_DOMAIN: return indiv

        result = [[], []]
        for i in range(len(indiv[0])):
            if indiv[0][i] <= self.NUM_DOMAIN:
                result[0].append(indiv[0][i])
                result[1].append(indiv[1][i])
        return np.array(result)

    def Cost(self, indiv):
        indiv = self.Decode(indiv)

        path = []
        weight = np.inf
        domains = np.argsort(-np.array(indiv[0])) + 1
        curr_domain = domains[0]
        end_domains = []   #(domain, weight, index in path)
        start_node = self.START_NODE

        for next_domain in domains[1:]:
            start_node_set = []
            for v in self.domain_start_nodes[next_domain]:
                if self.distance[curr_domain][start_node][v] != np.inf:
                    start_node_set.append(v)

            dis_to_end = self.distance[curr_domain][start_node][self.END_NODE]
            if  dis_to_end != np.inf:
                end_domains.append(curr_domain, weight+dis_to_end, len(path))

            if len(start_node_set) == 0:
                if len(end_domains) == 0:
                    continue
                else:
                    break

            end_node = start_node_set[int(indiv[1][curr_domain-1])%len(start_node_set)]
            weight += self.distance[curr_domain][start_node][end_node]
            path.append((curr_domain, start_node, end_node))
            curr_domain = next_domain
            start_node = end_node

        dis_to_end = self.distance[curr_domain][start_node][self.END_NODE]
        if  dis_to_end != np.inf :
            end_domains.append(curr_domain, weight+dis_to_end, len(path))
        
        if len(end_domains) == 0:
            return path, np.inf
        else:
            endDomain = min(end_domains, key=lambda tup: tup[1])
            path[endDomain[2]] = (endDomain[0], path[endDomain[2]][1], self.END_NODE)
            
            return path[:endDomain[2]+1], endDomain[1]


    def show_path(self, indiv):
        path, weight = self.Cost(indiv)

        res = {}
        for pi in path:
            path_i = []
            domain = pi[0]
            start = pi[1]
            end = pi[2]
            pre = self.pre_node[domain][start]
            curr = end
            while(pre[curr] != curr):
                path_i.append(curr)
                curr = pre[curr]
            path_i.append(start)
            res[domain] = path_i[::-1]

        if(weight != np.inf):
            print(res)
        else:
            print('Error path: Not find path to end node!')
        return res, weight



class EncodePriority(GraphDomain):

##################################### Version 2 ########################################
#                                 Encode Priority                                      #


    def Decode(self, indiv):
        if len(indiv) == self.NUM_DOMAIN: return indiv
        result = []
        for i in indiv:
            if i <= self.NUM_DOMAIN:
                result.append(i)
        return np.array(result)


    def Cost(self, indiv):
        domains = self.Decode(indiv)

        dis = np.full((self.NUM_NODE+1,), np.inf)
        dis[self.START_NODE] = 0

        for d in domains:
            PQ = PriorityQueue()
            for u in self.domain_start_nodes[d]:
                if (dis[u] != np.inf):
                    PQ.put((dis[u], u))

            while not PQ.empty():
                top = PQ.get()
                u = top[1]  
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
        
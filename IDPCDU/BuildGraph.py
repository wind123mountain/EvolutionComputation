import numpy as np

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
        self.build_graph_Floyd_Warshall()
 
        self.best = np.inf
        self.indiv_best = []
        
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
                self.distance[d][v][v] = 0
                self.pre_node[d][v] = np.full((self.NUM_NODE+self.pad,), -1)
        
        for e in edges:
            e = e.strip()
            e = e.split(' ')
            #u = e[0], v = e[1], w = e[2], d = e[3]
            self.adj[int(e[3])][int(e[0])].append((int(e[1]), int(e[2])))
            self.distance[int(e[3])][int(e[0])][int(e[1])] = min(self.distance[int(e[3])][int(e[0])][int(e[1])], int(e[2]))
            
    def Floyd_Warshall(self, domain):
        dis = self.distance[domain]
        for k in range(self.pad, self.pad+self.NUM_NODE):
            if len(self.adj[domain][k]) != 0:
                self.domain_start_nodes[domain].append(k)
            for i in range(self.pad, self.pad+self.NUM_NODE):
                for j in range(self.pad, self.pad+self.NUM_NODE):
                    dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])
    
    def build_graph_Floyd_Warshall(self):
        for d in range(self.pad, self.NUM_DOMAIN+self.pad):
            self.Floyd_Warshall(d)


class AlgorithmV1(GraphDomain):


##################################### Version 1 ########################################
#                                 Encoding Version 1                                   #
# Gene has  two parts:
#    Priority of the Domain
#    End node of the domain

    def Decode(self, indiv):
        tmp = np.argsort(indiv[0]) + 1
        if len(indiv[0]) == self.NUM_DOMAIN: return indiv

        result = [[], []]
        for i in tmp:
            i = int(i)
            if i <= self.NUM_DOMAIN:
                result[0].append(indiv[0][i])
                result[1].append(indiv[1][i])
        return np.array(result)


    def Cost(self, ind):
        indiv = self.Decode(ind)
        # print('decode',indiv)

        # path = []
        weight = 0
        domains = np.argsort(-np.array(indiv[0])) + 1
        curr_domain = domains[0]
        end_domains = []   #(domain, weight, index in path)
        start_node = self.START_NODE

        # print('domains',domains)

        for next_domain in domains[1:]:
            start_node_set = []
            
            for v in self.domain_start_nodes[next_domain]:
                if self.distance[curr_domain][start_node][v] != np.inf:
                    start_node_set.append(v)

            dis_to_end = self.distance[curr_domain][start_node][self.END_NODE]
            if  dis_to_end != np.inf:
                end_domains.append([curr_domain, weight+dis_to_end])

            if len(start_node_set) == 0:
                if len(end_domains) == 0:
                    continue
                else:
                    break

            end_node = start_node_set[int(indiv[1][curr_domain-1])%len(start_node_set)]
            weight += self.distance[curr_domain][start_node][end_node]
            curr_domain = next_domain
            start_node = end_node

        dis_to_end = self.distance[curr_domain][start_node][self.END_NODE]
        if  dis_to_end != np.inf :
            end_domains.append([curr_domain, weight+dis_to_end])
        
        if len(end_domains) == 0:
            if self.best >= np.inf:
                self.best = np.inf
                self.indiv_best = ind
            return np.inf
        else:
            endDomain = min(end_domains, key=lambda tup: tup[1])
            if self.best >= endDomain[1]:
                self.best = endDomain[1]
                self.indiv_best = ind
            return endDomain[1]
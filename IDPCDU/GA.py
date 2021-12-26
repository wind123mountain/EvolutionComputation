import numpy as np
import random
from build import GraphDomain

class GA:

    def __init__(self, size_pop, num_gens, Pc, Pm, task:GraphDomain):
        self.mark = set()
        self.size_pop = size_pop
        self.num_gens = num_gens
        self.Pc = Pc
        self.Pm = Pm 
        self.task = task 


    def check_generated(self, individual, index):
        individual = self.task.Decode(individual)
        individual = individual[:index]
        pre_size = len(self.mark)
        self.mark.add(tuple(individual))
        if(pre_size < len(self.mark)): return False
        
        return True

    def GeneratorPopulaion(self, sizePop):
        Populaion = []
        dims = self.task.NUM_DOMAIN
        for i in range(sizePop):
            pi = np.random.permutation(range(1, dims+1))
            f_pi, index = self.task.Cost(pi)
            while f_pi == self.task.best and self.check_generated(pi, index):
                pi = np.random.permutation(range(1, dims+1))
                f_pi, index = self.task.Cost(pi)
            
            Populaion.append((pi, f_pi))

        return Populaion

    def chooseParents(self, pop):
        n = len(pop)
        index1 = np.random.randint(n)
        index2 = np.random.randint(n)
        while index1 == index2:
            index2 = np.random.randint(n)
        
        index3 = np.random.randint(n)
        index4 = np.random.randint(n)
        while index3 == index4:
            index4 = np.random.randint(n)
        
        if pop[index1][1] < pop[index2][1]:
            p1 = pop[index1][0]
        else:
            p1 = pop[index2][0]
        
        if pop[index3][1] < pop[index4][1]:
            p2 = pop[index3][0]
        else:
            p2 = pop[index4][0]
        
        return p1, p2

    def cross_pmx(self, p1, p2):
        n = len(p1)
        l = int(n/3)
        
        index1 = np.random.randint(l)
        index2 = np.random.randint(l+l, 3*l)
        # while index1 == index2:
        #    index2 = np.random.randint(1, n-1)
    
        # if index1 > index2:
        #     temp = index1
        #     index1 = index2
        #     index2 = temp

        o1 = np.array(p2)
        o2 = np.array(p1)

        pos1 = np.full(n+1, -1, dtype=int)
        pos2 = np.full(n+1, -1, dtype=int)
        for i in range(index1, index2+1):
            pos1[o1[i]] = i
            pos2[o2[i]] = i
            
        for i in range(n):
            if index1 <= i and i <= index2: continue

            id1 = i
            while True:
                if pos1[p1[id1]] == -1:
                    o1[i] = p1[id1]
                    break
                id1 = pos1[p1[id1]]
            
            id2 = i
            while True:
                if pos2[p2[id2]] == -1:
                    o2[i] = p2[id2]
                    break
                id2 = pos2[p2[id2]]
        
        return o1, o2

    def paradox_mutation(self, indiv):
        n = len(indiv)
        index1 = 0
        index2 = np.random.randint(1, n-1)
    
        if index1 > index2:
            temp = index1
            index1 = index2
            index2 = temp

        while index1 < index2:
            temp = indiv[index1]
            indiv[index1] = indiv[index2]
            indiv[index2] = temp
            index1 += 1
            index2 -= 1

        return indiv


    def mutation_swap(self, individual):
        n = len(individual)
        res = np.array(individual)

        index1 = np.random.randint(5)
        index2 = np.random.randint(n)
        while index1 == index2:
            index2 = np.random.randint(n)

        temp = res[index1]
        res[index1] = res[index2]
        res[index2] = temp

        return res

    def selection(self, pre_pop, offs_pop, rate_sel = 0.1):  
        n = len(pre_pop)
        random.shuffle(offs_pop)
        next_pop = pre_pop[0:10] + offs_pop[0:n-10]
        next_pop.sort(key=lambda tup: tup[1])

        return next_pop

    def run(self):
        population = self.GeneratorPopulaion(self.size_pop)
        population.sort(key=lambda tup: tup[1])
        logg = {}

        for t in range(self.num_gens):
            offs_pop = []
            while len(offs_pop) < self.size_pop:
                p1, p2 = self.chooseParents(population)

                rand = np.random.random_sample()
                o1, o2 = [], []
                if rand <= self.Pc:
                    o1, o2 = self.cross_pmx(p1, p2)
                else:
                    o1 = self.paradox_mutation(p1)
                    o2 = self.paradox_mutation(p2)
                if np.random.random_sample() < self.Pm:
                    o1 = self.mutation_swap(o1)
                    o2 = self.mutation_swap(o2)
                
                f_o1, id1 = self.task.Cost(o1)
                if not(f_o1 == self.task.best and self.check_generated(o1, id1)):
                    offs_pop.append((o1, f_o1))

                f_o2, id2 = self.task.Cost(o2)
                if not(f_o2 == self.task.best and self.check_generated(o2, id2)):
                    offs_pop.append((o2, f_o2))
            
            population = self.selection(population, offs_pop)
            logg[t] = population[0][1]
        
        return logg

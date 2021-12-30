import numpy as np
import random
import math

class SR_MFEA:
    def __init__(self, Tasks, NUM_TASK, MAX_DIM , sizePop, TH,Pa,Pb,epochs):
        self.Tasks = Tasks
        self.NUM_TASK = NUM_TASK
        self.MAX_DIM  = MAX_DIM
        self.sizePop = sizePop
        self.n = sizePop
        self.k = self.NUM_TASK
        self.m = int(self.n/self.k)
        self.TH = TH
        self.Pa = Pa
        self.Pb = Pb
        self.epochs = epochs

        self.a1, self.b1, self.a2, self.b2 = self.heSo()

    def GeneratorIndiv(self):
        indiv = np.random.sample(self.MAX_DIM)

        fac_cost = []
        for task in self.Tasks:
            fac_cost.append(task.Cost(indiv))
        
        return indiv, fac_cost

    def Generate_Eval(self):
        population = []
        pop_cost = []
        for i in range(self.sizePop):
            pi, f_cost = self.GeneratorIndiv()
            population.append(pi)
            pop_cost.append(f_cost)
        population,pop_cost = np.asarray(population),np.asarray(pop_cost)
        rank = np.argsort(pop_cost,axis = 0) + 1

        return population,pop_cost,rank

    def heSo(self):
        # a1 + b1 = 1
        # a1*m+b1 = TH
        # a2*m + b2= TH
        # a2*n + b2 = 0
        a1 = (1-self.TH)/(1-self.m)
        b1 = 1 - a1
        a2 = self.TH/(self.m-self.n)
        b2 = (self.TH-a2*self.m)
        return a1,b1,a2,b2

    def fm(self,r):
        if r >= 1 and r <self.m:
            return self.a1*r+self.b1
        elif r == self.m:
            return self.TH
        elif r >= self.m+1:
            return self.a2*r+self.b2

    def abilitiVector(self,rank):
        abiVector =[]
        for x in range(len(rank)):
            df =[]
            for item in rank[x]:
                df.append(self.fm(item))
            abiVector.append(df)
        return np.array(abiVector)

    def cross_SBX(self, p1, p2, nc = 2):
        u = np.random.random_sample()
        if u < 0.5:
            beta = math.pow(2*u, 1/(nc+1))
        else:
            beta = math.pow(0.5/(1-u), 1/(nc+1))

        c1 = 0.5*((1+beta)*p1 + (1-beta)*p2)
        c2 = 0.5*((1-beta)*p1 + (1+beta)*p2)

        return c1, c2

    def Diff_Mutation(self,y1,y2,y3,y4):
        F = np.random.random_sample()
        return y1 + F * (y4 - y1 + y2 - y3)
    
    def paradox_mutation(self,indiv):
        n = self.MAX_DIM
        index1 = 0
        index2 = np.random.randint(1, n)

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

    def Offspring_Generation(self,pop,abiVector,task):
        n = pop.shape[0]
        offs_gen = []
        abi_gen = []
        for indiv in range(int(self.m/2)):
            index1 = np.random.randint(n)
            while True:
                index2 = np.random.randint(n)
                if index1 != index2: break
            p1,p2 = pop[index1],pop[index2]

            if np.random.random_sample() < self.Pa:
                c1,c2 = self.cross_SBX(p1,p2)

                if np.random.random_sample() < self.Pb:
                    c1 = self.Diff_Mutation(c1, p1, p2, self.Tasks[task].indvi_best)
                    c2 = self.Diff_Mutation(c2, p2, p1, self.Tasks[task].indvi_best)
            else:
                c1 = self.paradox_mutation(p1)
                c2 = self.paradox_mutation(p2)


            if np.random.random_sample() < 0.5:
                a1 = abiVector[index1]
                a2 = abiVector[index2]
            else:
                a1 = abiVector[index2]
                a2 = abiVector[index1]
            
            offs_gen.append(c1)
            offs_gen.append(c2)
            abi_gen.append(a1)
            abi_gen.append(a2)
        return np.array(offs_gen), np.array(abi_gen) 

    def Select_Eval(self,offs_gen,abi_gen,task):
        size_gen = offs_gen.shape[0]
        tmp = np.arange(size_gen)
        random.shuffle(tmp)
        
        new_gen = []
        new_fc = []
        for x in tmp:
            indiv = offs_gen[x]
            abi_x = abi_gen[x]
            fc = []
            for js in range(self.NUM_TASK):
                if js == task or np.random.random_sample() <= abi_x[js]:
                    cost = self.Tasks[js].Cost(indiv)
                else:
                    cost = np.inf
                fc.append(cost)
            new_fc.append(fc)
            new_gen.append(indiv)
        return np.array(new_gen),np.array(new_fc)

    def updateAbility(self,pop_cost):
        rank = np.argsort(-pop_cost,axis = 0) + 1
        abiVector = self.abilitiVector(rank)
        return abiVector

    def SREMTO(self):
        population,pop_cost,rank = self.Generate_Eval()

        abiVector = self.abilitiVector(rank)
        
        log = []

        for epoch in range(self.epochs):
            off_gen = []
            off_fc = []
            for task in range(self.k):
                index = abiVector[:,task].argsort()[-(self.m):]
                group = [population[index],pop_cost[index],rank[index],abiVector[index]]
                
                offs_gen,abi_gen = self.Offspring_Generation(group[0],group[3],task)
                
                new_gen,new_fc = self.Select_Eval(offs_gen,abi_gen,task,)
                
                off_gen.append(new_gen)
                off_fc.append(new_fc)
            
            
            off_gen = np.concatenate(off_gen)
            off_fc = np.concatenate(off_fc)
    
            
            population = np.concatenate([population,off_gen],axis=0)
            pop_cost = np.concatenate([pop_cost,off_fc],axis=0)
            
            #update
            tmp_pop = []
            tmp_cost = []
            for task in range(self.k):
                index = pop_cost[:,task].argsort()[:self.m]
                tmp_pop.append(population[index])
                tmp_cost.append(pop_cost[index])
            population = np.concatenate(tmp_pop)
            pop_cost = np.concatenate(tmp_cost)
            abiVector = self.updateAbility(pop_cost)
            
            best = []
            for task in range(self.k):
                best.append(pop_cost[:,task].min())
            log.append(best)
            
        return np.array(log)

            

#TODO: Update with new format
__author__ = 'Alejandro CORREA BAHNSEN'
__version__ = '2.0-dev'

import numpy as np


class GAbin:
    # Same as in BA software
    # Algorithm to evaluate a fitnessFuction using binary genetic algorithms

    def __init__(self, K, fitnessFunction, iters, NC=10, MP=-1, CS=3, fargs=[]):
        # K = Number of elements
        #NC= Population size
        #CS= Number of elite
        #MP= Mutation probability
        #Range = matrix of range of each element, first row low range and second row high range
        #fitnessFunction = function to minimize, first parameter the vector of solution to evaluate,
        #    after the parameters in the fargs list

        """

        :param K:
        :param fitnessFunction:
        :param iters:
        :param NC:
        :param MP:
        :param CS:
        :param fargs:
        """
        self.K = int(K)
        self.NC = int(NC)

        if MP < 0:
            self.MP = 1.0 / K
        else:
            self.MP = MP

        self.iters = iters
        self.fitnessFunction = fitnessFunction
        self.fargs = fargs

        self.CS = CS

        #Initial random population
        self.pop = np.random.binomial(1, 0.5, size=(self.NC, self.K))

    def fit(self):

        self.hist = np.zeros((self.iters, self.K + 1))
        self.full_hist = np.zeros((self.iters * self.NC, self.K + 1))

        for i in range(self.iters):

            # Evaluate fitness function
            cost = self.fitnessFunction(self.pop, *self.fargs)

            cost_sort = np.argsort(cost)

            print i, 1 - cost[cost_sort[0]]

            #Save best result
            self.hist[i] = np.hstack((self.pop[cost_sort[0]], 1 - cost[cost_sort[0]]))

            # Save all results
            self.full_hist[i * self.NC:(i + 1) * self.NC] = np.hstack((self.pop, 1 - cost[:, np.newaxis]))

            #Elitims
            new_pop = self.pop[cost_sort[0:self.CS]]

            #Select Parents
            NumParents = self.NC - self.CS

            #Cumulative probability of selection as parent
            zcost = (cost - np.average(cost)) / np.std(cost)
            from scipy.stats import norm

            pzcost = 1 - norm.cdf(zcost)
            pcost = np.cumsum(pzcost / sum(pzcost))

            #Select parents & match
            rand_parents = np.random.rand(NumParents, 2)
            parents = np.zeros(rand_parents.shape, dtype=np.int)
            for e1 in range(NumParents):
                for e2 in range(2):
                    if (np.nonzero(pcost < rand_parents[e1, e2])[0] + 1).sum() > 0:
                        parents[e1, e2] = max(np.nonzero(pcost < rand_parents[e1, e2])[0]) + 1

                #Continious
                #rand_match=np.random.rand(self.K)
                #child=self.pop[parents[e1,0]]*rand_match+(1-rand_match)*self.pop[parents[e1,1]]
                #new_pop=np.vstack((new_pop,child))

                #Binary
                #random sigle point matching
                rand_match = int(np.random.rand() * self.K)
                child = np.hstack((self.pop[parents[e1, 0], 0:rand_match], self.pop[parents[e1, 1], rand_match:]))
                new_pop = np.vstack((new_pop, child))

            #Mutate
            num_mutations = int(round(self.K * NumParents * self.MP, 0))
            l_mutations = np.array([np.random.random_integers(NumParents, size=num_mutations) + self.CS - 1,
                                    np.random.random_integers(self.K, size=num_mutations) - 1])

            for e3 in range(int(num_mutations)):
                new_pop[l_mutations[0, e3], l_mutations[1, e3]] = (new_pop[
                                                                       l_mutations[0, e3], l_mutations[1, e3]] - 1) ** 2

            # replace replicates with random
            for e1 in range(new_pop.shape[0] - 1):
                for e2 in range(e1 + 1, new_pop.shape[0]):
                    if np.array_equal(new_pop[e1], new_pop[e2]):
                        new_pop[e2] = np.random.binomial(1, np.random.rand(1)[0] * 0.4 + 0.3, size=(new_pop.shape[1]))

            self.pop = new_pop

        self.x = self.hist[-1, :-1]
        self.best_cost = self.hist[-1, -1]


# Implementation based on \cite{Haupt2004} Haupt, R., & Haupt, S. (2004). Practical genetic algorithms (Second Edi.). New Jersey: John Wiley & Sons, Inc. Retrieved from http://books.google.com/books?hl=en&lr=&id=k0jFfsmbtZIC&oi=fnd&pg=PR11&dq=PRACTICAL+GENETIC+ALGORITHMS&ots=PP5QNHJEr9&sig=PHA7Z1cFCKoQLJLPcg_RBoSWEbA
#Initial parameters base on \cite{Marslan} Marslan, S. (2009). Machine Learning: An Algorithmic Perspective. New Jersey, USA: CRC Press. Retrieved from http://www-ist.massey.ac.nz/smarsland/MLbook.html

import numpy as np, datetime as dt

# TODO: Create only one GA function for both binary and continious cases

class GAcont:
    # Same as Cetrel
    def __init__(self, K, fitnessFunction, iters, NC=100, MP=-1, CS=10, range=None, print_iter=True, fargs=[]):
        #K = Number of elements
        #NC= Population size
        #CS= Number of elite
        #MP= Mutation probability
        #Range = matrix of range of each element, first row low range and second row high range
        #fitnessFunction = function to minimize, first parameter the vector of solution to evaluate,
        #    after the parameters in the fargs list

        self.K = K
        self.NC = NC
        self.print_iter = print_iter

        if MP < 0:
            self.MP = 1.0 / K
        else:
            self.MP = MP

        self.iters = iters
        self.fitnessFunction = fitnessFunction
        self.fargs = fargs

        self.CS = CS
        if range == None:
            #Range (-1,1)
            self.range1 = np.vstack(((-1) * np.ones((1, K)), np.ones((1, K))))
        else:
            self.range1 = range
        #Initial random population
        self.pop = (self.range1[1,] - self.range1[0,]) * np.random.rand(NC, self.K) + self.range1[0,]


    def evaluate(self):
        t0 = dt.datetime.now()
        self.hist = np.zeros((self.iters, self.K + 1))
        self.full_hist = np.zeros((self.iters * self.NC, self.K + 1))

        for i in range(self.iters):

            #Evaluate fitness function
            cost = self.fitnessFunction(self.pop, *self.fargs)

            cost_sort = np.argsort(cost)

            #Save best result
            self.hist[i] = np.hstack((self.pop[cost_sort[0]], cost[cost_sort[0]]))

            # Save all results
            self.full_hist[i * self.NC:(i + 1) * self.NC] = np.hstack((self.pop, cost[:, np.newaxis]))

            if self.print_iter:
                print "GA iter " + str(i) + " of " + str(self.iters) + " - secs " + str(
                    (dt.datetime.now() - t0).seconds) + ' - best = ' + str(min(cost))

            #Elitims
            new_pop = self.pop[cost_sort[0:self.CS]]

            #Select Parents
            NumParents = int(self.NC - self.CS)

            #Cumulative probability of selection as parent
            zcost = (cost - np.average(cost)) / np.std(cost)
            from scipy.stats import norm

            pzcost = 1 - norm.cdf(zcost)
            pcost = np.cumsum(pzcost / sum(pzcost))

            #Select parents & match
            rand_parents = np.random.rand(NumParents, 2)
            parents = np.zeros(rand_parents.shape, dtype=np.int)
            for e1 in range(NumParents):
                for e2 in range(2):
                    if (np.nonzero(pcost < rand_parents[e1, e2])[0] + 1).sum() > 0:
                        parents[e1, e2] = max(np.nonzero(pcost < rand_parents[e1, e2])[0]) + 1

                rand_match = np.random.rand(self.K)
                child = self.pop[parents[e1, 0]] * rand_match + (1 - rand_match) * self.pop[parents[e1, 1]]
                new_pop = np.vstack((new_pop, child))

            #Mutate
            num_mutations = int(round(self.K * NumParents * self.MP, 0))
            l_mutations = np.array([np.random.random_integers(NumParents, size=num_mutations) + self.CS - 1,
                                    np.random.random_integers(self.K, size=num_mutations) - 1])

            for e3 in range(int(num_mutations)):
                new_pop[l_mutations[0, e3], l_mutations[1, e3]] = (self.range1[1, l_mutations[1, e3]] - self.range1[
                    0, l_mutations[1, e3]]) * np.random.rand() + self.range1[0, l_mutations[1, e3]]

            self.pop = new_pop

        self.x = self.hist[-1, :-1]
        self.best_cost = self.hist[-1, -1]

        ##Test using Haupt Book example
        #def cost(pop,a,b,c):
        #    return pop[:,0]*np.sin(4*pop[:,0])+1.1*pop[:,1]*np.sin(2*pop[:,1])+a+b+c.sum()
        #
        #ga=None
        #c=np.array([[10,10],[10,10]])
        #ga=ga_cont(2,cost,10,100,CS=5,MP=0.3,range=np.vstack((0*np.ones((1,2)),10*np.ones((1,2)))),fargs=[0,0,c])
        #ga.evaluate()
        ##import matplotlib.pyplot as plt
        ##plt.plot(ga.hist[:,2])
        ##plt.show()
        #print ga.x
        #print ga.best_cost

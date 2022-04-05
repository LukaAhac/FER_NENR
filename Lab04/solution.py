import numpy as np


LB = -4
UB = 4
NOE = 5

#Function defined up to the parameters
def function(x, y, params):

    if len(params) != NOE:

        raise Exception("Wrong number of parameters!")

    return np.sin(params[0]+params[1]*x) + params[2]*np.cos(x*(params[3]+y))/(1+np.exp((x-params[4])**2))

#Structure for saving possible solutions
class Chromosome():

    #Constructor - If genes are not provided, randomly select them from unifrom distribution [-4.4), use data to calculate how good the solution is
    def __init__(self, genes = None):

        if genes is None:

            self.genes = np.random.uniform(LB, UB, NOE)

        else:

            if(len(genes) != NOE):
                raise Exception("Wrong genes length!")

            else:

                self.genes = genes

        self.value = None

    #Evaluate the example by calculating arithmetic mean of absolute error on the examples
    def evaluate(self,data):

        arithmetic_mean = np.array([])

        for line in data:

            x,y,value = line.split("\t")
            x = float(x)
            y = float(y)
            value = float(value)
            arithmetic_mean = np.append(arithmetic_mean, (value - function(x,y,self.genes))**2)

        arithmetic_mean = np.mean(arithmetic_mean)
        self.value = arithmetic_mean

    #Reporoduction by unifrom crossover method
    @staticmethod
    def reproduction(parent1, parent2):

        parents = [parent1,parent2]

        new_gene = np.array([])
        for index in range(5):
            new_gene = np.append(new_gene, parents[int(np.random.randint(0,2))].genes[index])


        return Chromosome(genes = new_gene)

    #Perform mutation
    def mutation(self, probability):

        for index in range(5):
            number = np.random.uniform()

            if number < probability:

                self.genes[index] = np.random.uniform(LB,UB)


class Population():

    def __init__(self,populationSize, data):

        self.members = []
        self.data = data
        self.populationSize = populationSize

        for _ in range(populationSize):

            self.members.append(Chromosome())

    def evaluateMembers(self):

        for index in range(self.populationSize):
            self.members[index].evaluate(self.data)

        k = 0

        #???
        for index in range(self.populationSize):
            k += self.members[index].value

        self.k = k

    def add(self,chromosome):
        self.populationSize += 1
        self.members.append(chromosome)

    def replace(self,replaced,replacement):
        self.members = [x if x != replaced else replacement for x in self.members]


    def findBest(self):

        best = 0
        for index in range(self.populationSize):
            if self.members[index].value < self.members[best].value:
                best = index

        return self.members[best]

    def selection(self,n):

        parents = []

        for _ in range(n):
            chosen = 0
            limit = np.random.uniform(0,self.k)
            upperLimit = 1 / self.members[chosen].value

            while limit > upperLimit and chosen < self.populationSize-1:
                chosen += 1
                upperLimit += 1 / self.members[chosen].value

            parents.append(self.members[chosen])

        return parents

    def choseRandomly(self,n):
        return np.random.choice(self.members, n, replace = False)


class GeneticAlgo():

    def __init__(self,popSize,mutationProbability,iterations,elite,data):

        self.popSize = popSize
        self.mp = mutationProbability
        self.iterations = iterations
        self.elite = elite
        self.data = data


    def newBest(self,iteration,population,best):
        populationBest = population.findBest()
        if populationBest.value < best.value:
            best = populationBest
            print("Iteration: {}".format(iteration))
            print("New best is {}, with value: {}".format(best.genes,best.value))

        return best

    def run(self):

        population = Population(self.popSize,self.data)
        population.evaluateMembers()
        best = population.findBest()
        print("Iteration: 0")
        print("Best is {}, with value: {}".format(best.genes,best.value))

        for iterationIndex in range(self.iterations):

            newPopulation = Population(0,self.data)

            if self.elite == True:
                newPopulation.add(best)

            for _ in range(self.popSize-newPopulation.populationSize):

                parents = population.selection(2)

                child = Chromosome.reproduction(parents[0],parents[1])

                child.mutation(self.mp)

                newPopulation.add(child)

            population = newPopulation

            population.evaluateMembers()

            best = self.newBest(iterationIndex,population,best)


        return best


class TournamentSelection():

    def __init__(self,popSize,mutationProbability,iterations,data):

        self.popSize = popSize
        self.mp = mutationProbability
        self.iterations = iterations
        self.data = data

    def newBest(self,iteration,population,best):
        populationBest = population.findBest()
        if populationBest.value < best.value:
            best = populationBest
            print("Iteration: {}".format(iteration))
            print("New best is {}, with value: {}".format(best.genes,best.value))

        return best    

    def run(self):

        population = Population(self.popSize,self.data)
        population.evaluateMembers()
        best = population.findBest()
        print("Iteration: 0")
        print("Best is {}, with value: {}".format(best.genes,best.value))

        for iterationIndex in range(self.iterations):

            threeElements = population.choseRandomly(3)
            threeElements = sorted(threeElements, key = lambda x: x.value)

            child = Chromosome.reproduction(threeElements[0],threeElements[1])

            child.mutation(self.mp)
            child.evaluate(self.data)

            population.replace(threeElements[2],child)


            best = self.newBest(iterationIndex,population,best)


        return best


which = input("Pokretanje Generacijskog/Eliminacijskog genetskog algoritma - G/E: ")
whichData = input("Data1 ili Data2 - 1/2: ")

if whichData == "1":
    data = open(r"Lab04\zad4-dataset1.txt")
    data = data.readlines()
elif whichData == "2":
    data = open(r"Lab04\zad4-dataset2.txt")
    data = data.readlines()
else:
    exit()

if which == "G":

    algo = GeneticAlgo(50,0.01,2000,True,data)

    result = algo.run()

    print(result.genes)
elif which == "E":

    algo = TournamentSelection(20,0.01,20000,data)

    result = algo.run()

    print(result.genes)


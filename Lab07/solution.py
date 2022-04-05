import numpy as np
from random import uniform
from random import sample
from random import randint
from math import exp
from math import sqrt
import matplotlib.pyplot as plt

first_layer_fun = lambda x1,x2,w1,s1,w2,s2: 1/(1 + (abs(x1-w1)/s1 + (abs(x2-w2)/s2)))
sigmoid = lambda x: 1/(1+np.exp(-x))

class Dataset():
    def __init__(self,path):

        with open(path,"r") as f:
            data =f.readlines()

        self.data = []

        for line in data:
            self.data.append(list(map(float,line.strip().split())))


class neural_netowrk():

    def __init__(self,architecture, regParam = 0.1):

        self.architecture = architecture
        self.regParam = regParam

        layers = architecture.split("x")
        self.layers = [int(x) for x in layers]

        self.number_of_parameters = self.layers[1]*4

        for index in range(1,len(layers)-1):
            self.number_of_parameters += (self.layers[index]+1) * self.layers[index+1]
        
    def calcOuput(self,x1,x2,parameters):

        position = 0

        first_layer = []

        for _ in range(self.layers[1]):

            first_layer.append(first_layer_fun(x1,x2,parameters[position],parameters[position+1],parameters[position+2],parameters[position+3]))

            position += 4

        layer = first_layer

        for index in range(1,len(self.layers)-1):

            next_layer = [0]*self.layers[index+1]

            for index2 in range(self.layers[index+1]):

                for index3 in range(self.layers[index]):

                    next_layer[index2] += layer[index3] * parameters[position]

                    position += 1

                next_layer[index2] += parameters[position]

                position += 1
            
            layer = []

            for x in next_layer:
                layer.append(sigmoid(x))

        return layer

    #Računanje petljama - SPORIJE
    def calcError1(self, parameters, data):

        mse = 0

        for example in data:

            output = self.calcOuput(example[0],example[1],parameters)

            mse += (example[2] - output[0])**2 + (example[3] - output[1])**2 + (example[4] - output[2])**2

        return mse/len(data)

    #Računanje matrično - BRŽE
    def calcError(self,parameters,data):

        data_len = len(data)

        given_outputs = []

        layer_1_output = []

        for example in data:

            position = 0

            given_outputs.append([example[2],example[3],example[4]])

            layer_1_output.append([])

            for _ in range(self.layers[1]):

                layer_1_output[-1].append(first_layer_fun(example[0],example[1],parameters[position],parameters[position+1],parameters[position+2],parameters[position+3]))

                position += 4

        layer_1_output = np.transpose(np.array(layer_1_output))

        given_outputs = np.array(given_outputs).T

        weights = []
        biases = []

        for index in range(2,len(self.layers)):

            weights.append([])
            biases.append([])

            for _ in range(self.layers[index]):

                w = []
                b = []

                for _ in range(self.layers[index-1]):
                    
                    w.append(parameters[position])
                    position += 1

                b = b + [parameters[position]]*data_len

                position += 1

                weights[-1].append(w)
                biases[-1].append(b)

            weights[-1] = np.array(weights[-1])
            biases[-1] = np.array(biases[-1])
        
        next_layer = layer_1_output

        for index in range(len(weights)):

            next_layer = np.matmul(weights[index], next_layer)
            next_layer = next_layer + biases[index]
            next_layer = sigmoid(next_layer)


        return np.sum(np.square(given_outputs - next_layer))/data_len # + self.regParam * sqrt(np.sum(np.square(np.array(parameters[:-1]))))




class Genetic_algorithm():

    def __init__(self,popSize = 50,pm = 0.3,t_i=[2,1,1],pm_i = [0.3,0.3,0.3],distrib = [0.1,1,1],iterations = 100000,nn = None, data = None):

        self.popSize = popSize
        self.nn = nn
        self.data = data
        self.iterations = iterations
        self.pm = pm
        self.t_i = t_i
        self.pm_i = pm_i
        self.distrib = distrib

    def crossover(self,mother,father):

        which = randint(1,3)
        
        #Jednostavna aritmetička rekombinacija
        if which == 3:
            position = randint(0,len(mother))
            new = mother[:position]
            for index in range(position,len(mother)):
                new.append((mother[index]+father[index])/2)
            return new

        #ARITMETIČKO
        if which == 1:
            new = []
            a = uniform(0,1)
            for index in range(len(mother)):
                new.append(a*mother[index]+(1-a)*father[index])
            return new

        #Diskretna rekombinacija
        # elif which == 2:
        #     new = []
        #     for index in range(len(mother)):
        #         if randint(1,2) == 1:
        #             new.append(mother[index])
        #         else:
        #             new.append(father[index])
        #     return new

        #HEURISTIČKO
        elif which == 2:
            new = []
            a = uniform(0,1)
            if mother[-1] > father[-1]:
                for index in range(len(mother)):
                    new.append(a*(father[index]-mother[index])+father[index])
            else:
                for index in range(len(mother)):
                    new.append(a*(mother[index]-father[index])+mother[index])
            return new

        #Potpuna aritmetička rekombinacija
        # if which == 2:
        #     new = []
        #     for index in range(len(mother)):
        #         new.append((mother[index]+father[index])/2)
        #     return new

    def mutation(self,new):

        v_i = [self.t_i[index]/sum(self.t_i) for index in range(len(self.t_i))]

        random = uniform(0,1)

        #M1
        if random < v_i[0]:
            for index in range(len(new)):
                if uniform(0,1) < self.pm_i[0]:
                    new[index] += uniform(0,self.distrib[0])
        #M2
        elif random < v_i[0]+v_i[1]:
            for index in range(len(new)):
                if uniform(0,1) < self.pm_i[1]:
                    new[index] += uniform(0,self.distrib[1])
        #M3
        else:
            for index in range(len(new)):
                if uniform(0,1) < self.pm_i[2]:
                    new[index] = uniform(0,self.distrib[2])
        return new

    def run(self):

        #Stvori populaciju
        self.population = []
        for _ in range(self.popSize):
            self.population.append(list(np.random.uniform(-1,1,size = self.nn.number_of_parameters)))

            # self.population[-1][0] = 0.125
            # self.population[-1][2] = 0.25
            # self.population[-1][4] = 0.625
            # self.population[-1][6] = 0.75
            # self.population[-1][8] = 0.875
            # self.population[-1][10] = 0.25
            # self.population[-1][12] = 0.125
            # self.population[-1][14] = 0.75
            # self.population[-1][16] = 0.375
            # self.population[-1][18] = 0.25
            # self.population[-1][20] = 0.875
            # self.population[-1][22] = 0.75
            # self.population[-1][24] = 0.375
            # self.population[-1][26] = 0.75
            # self.population[-1][28] = 0.625
            # self.population[-1][30] = 0.25

        #Evaluairaj populaciju
        for index in range(len(self.population)):
            self.population[index].append(self.nn.calcError(self.population[index],self.data))

        #ponavljaj dok nije zadovoljen uvjet zaustavljanja
        iterations = 0
        while(iterations <= self.iterations):
        #while True:

            #odaberi slucajno k jedniki
            k_samples = sample(self.population,3)

            #pronađi najlosiju; zatim ju ukloni
            k_samples.sort(key = lambda x: x[-1], reverse = True)
            self.population.remove(k_samples[0])
            
            #nova = krizanje neke preostale 2
            new = self.crossover(k_samples[1],k_samples[2])

            #mutacija
            if uniform(0,1) < self.pm:
                new = self.mutation(new)

            #evaluacija nove
            new[-1] = self.nn.calcError(new,self.data)

            #dodaj novu u populaciju
            self.population.append(new)

            if iterations % 1000 == 0:
                self.population = sorted(self.population, key = lambda x: x[-1])
                print(self.population[0][-1])
                # print("Regularizirano: ",self.population[0][-1])
                # print(self.population[0][-1] - self.nn.regParam * sqrt(np.sum(np.square(np.array(self.population[0][:-1])))))

                file.write(str(iterations))
                file.write(str(self.population[0]))
                file.write("\n")

                if self.population[0][-1] <= 1e-7:
                    break
        

            iterations += 1

            
        
        self.population = sorted(self.population, key = lambda x: x[-1])
        print(self.population[0])
        print(iterations)
        return self.population[0]


if __name__ == "__main__":
    nn = neural_netowrk("2x8x3",regParam = 0.00001)

    data = Dataset(r"Lab07\dataset.txt")
    file = open(r"Lab07\outputs.txt","a")

    genalgo = Genetic_algorithm(popSize = 50,pm = 0.1,t_i=[0.45,0.2,0.35],pm_i = [0.05,0.05,0.03],distrib = [0.01,0.3,0.5],iterations = 20000, nn = nn, data=data.data)

    result = genalgo.run()

    class1_x = []
    class1_y = []
    class2_x = []
    class2_y = []
    class3_x = []
    class3_y = []

    for line in data.data:

        if line[2] == "1":
            class1_x.append(float(line[0]))
            class1_y.append(float(line[1]))
        elif line[3] == "1":
            class2_x.append(float(line[0]))
            class2_y.append(float(line[1]))
        else:
            class3_x.append(float(line[0]))
            class3_y.append(float(line[1]))

    plt.scatter(class1_x,class1_y, label = "Class 1")
    plt.scatter(class2_x,class2_y, label = "Class 2")
    plt.scatter(class3_x,class3_y, label = "Class 3")
    plt.legend()

    for x in range(8):
        plt.scatter(result[x*4],result[x*4+2])

    plt.show()
from random import uniform
from math import exp
import numpy as np
from dataset_generator import generate_data
import matplotlib.pyplot as plt
from drawing import draw3d

#Sigmoida definirana do na parametre
def sigmoid(x,a,b):

    if (-b*(a-x)) > 100:
        return 0

    return 1/(1+exp(-b*(a-x)))

#F debinirana do na parametre
def f(x,y,p,q,r):

    return x*p+y*q+r

class ANFIS():

    def __init__(self,number_of_rules,iterations = 20000):

        #Broj pravila
        self.number_of_rules = number_of_rules
        #Broj iteracija
        self.iterations = iterations

        #Inicijalizacija za prvi dio antecedenta
        self.Alfa_a = np.random.uniform(-1,1,self.number_of_rules)
        self.Alfa_b = np.random.uniform(-1,1,self.number_of_rules)
        #Inicijalizacija za drugi dio antecedenta
        self.Beta_a = np.random.uniform(-1,1,self.number_of_rules)
        self.Beta_b = np.random.uniform(-1,1,self.number_of_rules)
        #Inicijalizacija za konzekvens
        self.p = np.random.uniform(-1,1,self.number_of_rules)
        self.q = np.random.uniform(-1,1,self.number_of_rules)
        self.r = np.random.uniform(-1,1,self.number_of_rules)


    def predict(self,X):

        Y = []

        for example in X:

            sum_of_Wi = 0
            sum_of_Wi_times_fi = 0

            for index2 in range(self.number_of_rules):

                Wi = sigmoid(example[0],self.Alfa_a[index2],self.Alfa_b[index2])*sigmoid(example[1],self.Beta_a[index2],self.Beta_b[index2])

                sum_of_Wi += Wi
                sum_of_Wi_times_fi += Wi*f(example[0],example[1],self.p[index2],self.q[index2],self.r[index2])
                
            output_of_example = sum_of_Wi_times_fi/sum_of_Wi

            Y.append(output_of_example)

        return Y

    def error(self):

        error = 0

        for example in self.dataset:

            sum_of_Wi = 0
            sum_of_Wi_times_fi = 0

            for index2 in range(self.number_of_rules):

                Wi = sigmoid(example[0][0],self.Alfa_a[index2],self.Alfa_b[index2])*sigmoid(example[0][1],self.Beta_a[index2],self.Beta_b[index2])

                sum_of_Wi += Wi
                sum_of_Wi_times_fi += Wi*f(example[0][0],example[0][1],self.p[index2],self.q[index2],self.r[index2])
                
            output_of_example = sum_of_Wi_times_fi/sum_of_Wi

            error += (example[1] - output_of_example)**2

        return error/len(self.dataset)


    def train(self,dataset, batch_size = 1, eta1 = 0.000001, eta2 = 0.001, error = False):

        self.dataset = dataset

        error_matrix = []

        for it in range(1,self.iterations+1):

            for index in range(int(len(dataset)/batch_size)):

                #Djeljenje dataseta po batchevima
                train_on =  dataset[batch_size*index:batch_size*(index+1)]

                #Inicijalizacija vektora pogrešaka
                Alfa_a_error = np.zeros(self.number_of_rules)
                Alfa_b_error = np.zeros(self.number_of_rules)
                Beta_a_error = np.zeros(self.number_of_rules)
                Beta_b_error = np.zeros(self.number_of_rules)
                p_error = np.zeros(self.number_of_rules)
                q_error = np.zeros(self.number_of_rules)
                r_error = np.zeros(self.number_of_rules)

                #Za svaki primjer u batchu, računaj pogrešku
                for example in train_on:
                    
                    #Suma težina
                    sum_of_Wi = 0
                    #Težinska suma funkcija
                    sum_of_Wi_times_fi = 0

                    #Računanje sume težine i težinske sume funkcija
                    for index2 in range(self.number_of_rules):

                        Wi = sigmoid(example[0][0],self.Alfa_a[index2],self.Alfa_b[index2])*sigmoid(example[0][1],self.Beta_a[index2],self.Beta_b[index2])

                        sum_of_Wi += Wi
                        sum_of_Wi_times_fi += Wi*f(example[0][0],example[0][1],self.p[index2],self.q[index2],self.r[index2])

                    #Izlaz za dani primjer
                    output_of_example = sum_of_Wi_times_fi/sum_of_Wi

                    #Iteriraj po parametrima
                    for index2 in range(self.number_of_rules):

                        #Izracunaj Wk
                        Wk = sigmoid(example[0][0],self.Alfa_a[index2],self.Alfa_b[index2])*sigmoid(example[0][1],self.Beta_a[index2],self.Beta_b[index2])
                        #Izracunaj clan koji je zajednicki za p,q i r - radi efikasnosti
                        ts_minus_os_times_wk_times_wi_inverted = (example[1]-output_of_example)*Wk/sum_of_Wi

                        p_error[index2] += ts_minus_os_times_wk_times_wi_inverted*example[0][0]
                        q_error[index2] += ts_minus_os_times_wk_times_wi_inverted*example[0][1]
                        r_error[index2] += ts_minus_os_times_wk_times_wi_inverted

                        #Izracunaj fk
                        fk = f(example[0][0],example[0][1],self.p[index2],self.q[index2],self.r[index2])

                        #Izracunaj Wi*(fk+fi)
                        Wi_times_fk_minus_fi = 0

                        for index3 in range(self.number_of_rules):

                            Wi_times_fk_minus_fi += sigmoid(example[0][0],self.Alfa_a[index3],self.Alfa_b[index3])*sigmoid(example[0][1],self.Beta_a[index3],self.Beta_b[index3])*(fk-f(example[0][0],example[0][1],self.p[index3],self.q[index3],self.r[index3]))

                        #Izracunaj clanove koji je zajednicki za Alfa_a,Alfa_b,Beta_a i Beta_b - radi efikanosti
                        same_for_all = (example[1]-output_of_example)*Wi_times_fk_minus_fi/(sum_of_Wi**2)
                        Alfa = sigmoid(example[0][0],self.Alfa_a[index2],self.Alfa_b[index2])
                        Beta = sigmoid(example[0][1],self.Beta_a[index2],self.Beta_b[index2])

                        Alfa_a_error[index2] += same_for_all*Beta*Alfa*(1-Alfa)*self.Alfa_b[index2]
                        Alfa_b_error[index2] += same_for_all*Beta*Alfa*(1-Alfa)*(self.Alfa_a[index2]-example[0][0])
                        Beta_a_error[index2] += same_for_all*Alfa*Beta*(1-Beta)*self.Beta_b[index2]
                        Beta_b_error[index2] += same_for_all*Alfa*Beta*(1-Beta)*(self.Beta_a[index2]-example[0][1])

                self.Alfa_a = self.Alfa_a + eta1*Alfa_a_error
                self.Alfa_b = self.Alfa_b + eta1*Alfa_b_error
                self.Beta_a = self.Beta_a + eta1*Beta_a_error
                self.Beta_b = self.Beta_b + eta1*Beta_b_error
                self.p = self.p + eta2*p_error
                self.q = self.q + eta2*q_error
                self.r = self.r + eta2*r_error

            #if it == 1 or it % 1000 == 0:

            print("Greška je {}".format(self.error()))

            if error:
                error_matrix.append(self.error())

        return error_matrix


if True:
    trainData = generate_data()

    system = ANFIS(number_of_rules = 4, iterations = 1000)

    error = system.train(trainData,batch_size = 1, eta1 = 0.001,eta2 = 0.01,error = True)


    toPredict = []

    X = [x for x in range(-4,5)]
    Y = [x for x in range(-4,5)]

    for x in X:
        for y in Y:
            toPredict.append([x,y])

    X = []
    Y = []

    for x in toPredict:
        X.append(x[0])
        Y.append(x[1])

    Z = system.predict(toPredict)

    draw3d(X,Y,Z)

    Z_true = []

    for x in trainData:
        Z_true.append(x[1])

    Z_true = np.array(Z_true)
    Z = np.array(Z)

    draw3d(X,Y,Z-Z_true)

    X = np.arange(-20,20,0.01)

    plt.figure()

    for index in range(system.number_of_rules):

        A_Y = []
        B_Y = []

        for x in X:
            A_Y.append(sigmoid(x,system.Alfa_a[index],system.Alfa_b[index]))
            B_Y.append(sigmoid(x,system.Beta_a[index],system.Beta_b[index]))

        plt.subplot(system.number_of_rules,2,(index+1)*2-1)
        plt.plot(X,A_Y)
        plt.subplot(system.number_of_rules,2,(index+1)*2)
        plt.plot(X,B_Y)

    plt.subplots_adjust(wspace = 0.5,hspace = 0.5)

    plt.show()

    plt.figure()
    plt.plot([x for x in range(1,1001)],error)

    plt.show()

# if False:

#     trainData = generate_data()

#     system = ANFIS(number_of_rules = 4, iterations = 10000)

#     error = system.train(trainData,batch_size = len(trainData), eta1 = 0.0001,eta2 = 0.001, error = True)

#     dat2 = open(r"Lab06\gradient_error.txt","w")
#     for x in error:
#         dat2.write("%s\n" % x)


#     system = ANFIS(number_of_rules = 4, iterations = 10000)

#     error = system.train(trainData,batch_size = 1, eta1 = 0.001,eta2 = 0.01, error = True)

#     dat1 = open(r"Lab06\stohastic_error.txt","w")
#     for x in error:
#         dat1.write("%s\n" % x)

#     dat1.close()
#     dat2.close()

# if False:

#     dat1 = open(r"Lab06\stohastic_error.txt","r")    
#     dat2 = open(r"Lab06\gradient_error.txt","r")

#     x = [x for x in range(1,10001)]

#     stoh_error = dat1.readlines()
#     for index in range(len(stoh_error)):
#         stoh_error[index] = float(stoh_error[index].strip())
#     grad_error = dat2.readlines()
#     for index in range(len(grad_error)):
#         grad_error[index] = float(grad_error[index].strip())

#     plt.figure()

#     plt.subplot(1,2,1)
#     plt.title("Stohastički gradijentni")
#     plt.plot(x[:300],stoh_error[:300])
#     plt.subplot(1,2,2)
#     plt.title("Gradijentni")
#     plt.plot(x[:300],grad_error[:300])
#     plt.show()

# if False:
#     trainData = generate_data()

#     system = ANFIS(number_of_rules = 4, iterations = 300)

#     error = system.train(trainData,batch_size = len(trainData), eta1 = 0.01,eta2 = 0.01,error = True)

#     plt.figure()
#     plt.subplot(1,3,1)
#     plt.title("Gradijentni - eta1 = 0.01, eta2 = 0.01")
#     plt.plot([x for x in range(1,301)],error)

#     system = ANFIS(number_of_rules = 4, iterations = 300)

#     error = system.train(trainData,batch_size = len(trainData), eta1 = 0.0001,eta2 = 0.001,error = True)

#     plt.subplot(1,3,2)
#     plt.title("Gradijentni - eta1 = 0.0001, eta2 = 0.001")
#     plt.plot([x for x in range(1,301)],error)

#     system = ANFIS(number_of_rules = 4, iterations = 300)

#     error = system.train(trainData,batch_size = len(trainData), eta1 = 0.000001,eta2 = 0.000001,error = True)

#     plt.subplot(1,3,3)
#     plt.title("Gradijentni - eta1 = 0.000001, eta2 = 0.000001")
#     plt.plot([x for x in range(1,301)],error)

#     plt.show()
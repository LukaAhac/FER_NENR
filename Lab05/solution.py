import numpy as np
from math import exp
from gui_getdata import get_data



sigmoid = lambda x: 1/(1+np.exp(-x))

class neural_netowrk():

    def __init__(self,architecture,stopa_ucenja=0.1):

        self.architecture = architecture
        self.stopa_ucenja = stopa_ucenja

        layers = architecture.split("x")

        self.layers = [int(x) for x in layers]
        self.number_of_layers = len(self.layers)

        self.weights = []
        self.biases = []

        for index in range(len(layers)-1):
            self.weights.append(np.random.uniform(-1e-3,1e-3,size = (self.layers[index+1],self.layers[index])))
            self.biases.append(np.random.uniform(-1e-3,1e-3,size = (self.layers[index+1],1)))

        self.activations = [np.zeros((x,1)) for x in self.layers]
        self.sum_of_weighted_inputs = [np.zeros((x,1)) for x in self.layers]

    #X - matrica primjera, svaki stupac je jedan primjer, svaki red je značajka
    def forward_pass(self,X):

        self.activations[0] = X

        for index in range (0,self.number_of_layers-1):

            self.sum_of_weighted_inputs[index+1] = np.matmul(self.weights[index],self.activations[index])
            self.sum_of_weighted_inputs[index+1] = self.sum_of_weighted_inputs[index+1] - self.biases[index]
            self.activations[index+1] = sigmoid(self.sum_of_weighted_inputs[index+1])


    def backward_pass(self, Y_true):

        EI = []
        EW = []

        EI.append(np.multiply(np.multiply((self.activations[-1]-Y_true),self.activations[-1]),(1-self.activations[-1])))
        EW.append(np.matmul(EI[0],np.transpose(self.activations[-2])))

        for index in range(self.number_of_layers-2):

            EI.append(np.multiply(np.multiply(np.matmul(np.transpose(self.weights[-(1+index)]),EI[index]),self.activations[-(2+index)]),(1-self.activations[-(2+index)])))
            EW.append(np.matmul(EI[index+1],np.transpose(self.activations[-(3+index)])))


        for index in range(0,len(EI)):

            EI[index] = (np.ndarray.sum(EI[index],axis=1)).reshape((-1, 1))



        for index in range(0,len(EI)):
            self.biases[-(1+index)] = self.biases[-(1+index)] + self.stopa_ucenja * EI[index]

        for index in range(0,len(EW)):
            self.weights[-(1+index)] = self.weights[-(1+index)] - self.stopa_ucenja * EW[index]



    def fit(self,x,y,epoches=10000,batch_size = 1):

        for epoche in range(epoches):

            if (epoche+1) % 100 == 0:

                print("Epoha {} -> srednja kvadratna pogreška: {}".format(epoche+1,self.mean_kvadratic_error(x,y)))

            for index in range(int(len(x)/batch_size)):
                self.forward_pass(x[ : , (batch_size*index):(batch_size*(index+1))])
                self.backward_pass(y[ : , (batch_size*index):(batch_size*(index+1))])

    def predict(self,x):

        self.forward_pass(x)

        if self.activations[-1][0] == max(self.activations[-1]):
            print("Prediction: ALFA") 
        elif self.activations[-1][1] == max(self.activations[-1]):
            print("Prediction: BETA") 
        elif self.activations[-1][2] == max(self.activations[-1]):
            print("Prediction: GAMMA") 
        elif self.activations[-1][3] == max(self.activations[-1]):
            print("Prediction: DELTA") 
        elif self.activations[-1][4] == max(self.activations[-1]):
            print("Prediction: EPSILON") 
        else:
            print("FAIL")
            


    def mean_kvadratic_error(self,x,y):

        self.forward_pass(x)

        error = (np.sum(np.square(y-self.activations[-1])))/len(y)

        return error

def preparedata(path = r"Lab05\testData_reorderd.txt"):

    file = open(path,"r")
    lines = file.readlines()
    file.close()

    separated = []

    for x in lines:
        separated.append((x.strip()).split(";"))


    dots = []
    classes = []

    for elem in separated:
        dots.append(elem[:-1])
        classes.append(elem[-1])

    X = []
    Y = []

    for dot in dots:
        prazno = []
        for x in dot:
            splitted = x.split(",")
            prazno.append(float(splitted[0]))
            prazno.append(float(splitted[1]))
        X.append(prazno)

    for elem in classes:
        Y.append([int(x) for x in list(elem)])

    X = np.array(X)
    X = np.transpose(X)

    Y = np.array(Y)
    Y = np.transpose(Y)

    return X,Y


#MAIN

X,Y = preparedata()

net = neural_netowrk("100x50x5",stopa_ucenja = 0.01)   #Mreža 100x50x5 - ni = 0.01, Mreža 100x50x20x5 ni-0.1

net.fit(X,Y,epoches = 3000,batch_size = 100)


while True:
    data = get_data()

    if data is None:
        break

    net.predict(data)

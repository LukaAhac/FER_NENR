import matplotlib.pyplot as plt
import numpy as np

first_layer_fun = lambda x1,x2,w1,s1,w2,s2: 1/(1 + (abs(x1-w1)/s1 + (abs(x2-w2)/s2)))
sigmoid = lambda x: 1/(1+np.exp(-x))


class neural_netowrk():

    def __init__(self,architecture):

        self.architecture = architecture

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

        return np.sum(np.square(given_outputs - next_layer))/data_len



nn = neural_netowrk("2x8x3")

param = [-0.3782084762394054, -1.0015800099033219, 0.8020155811039739, -0.8558585174000775, 0.6249994591995756, -0.04417683787673973, 0.33848259399943126, 0.21332374592244263, 0.14774711888228925, -0.7436152709970298, 0.927414288352012, 2.3786132063940517, 0.5411072640280961, -0.31717296853262916, -0.13916138292326447, -0.17994244477274401, 0.15583759352930726, -0.045039809423624906, -0.034302786244723875, -0.2755186773250411, -0.19292229264772587, -0.28021154310476515, 0.18530094079908227, -0.7309047058435807, 0.34777139198859164, 0.047607214252849514, 0.12030576118863454, -0.16681873560960708, 0.8990639238404214, 0.22760087718167277, 0.18938103088031025, 0.16240804861505417, -3.260351146434033, -0.0002259963546851503, -5.1728472550648394, 92.70870580960344, -7.149871745283132, -0.09330221398646216, -0.4514587860643612, 114.12159236057491, 4.252555544756663, 10.881020837377939, -21.562342583659408, 3.202371410137503, 89.66308013473133, 98.25377749957352, 43.69650004229315, 332.490759117402, -6.429668469516933, 10.696665264379163, 22.42604609869361, 139.17607985792628, -16.05128850054345, -52.48061975517924, 4.219037219288221, 75.66429474907048, -173.6158102629191, -29.71251010183397, 35.52125303221611, 9.777246078447695e-08]

bruno_param = [0.6257662920486536, -0.09317050235684697, 0.7363507233062749, 0.19527476760818244,
0.37152087742543777,0.10079860890660958, 0.7386587955277296, 0.2036337208367274,
0.37621024069020886,0.0808972846446529,0.267409717254095,-0.22098697746880439,
0.12692419024898863,-0.12462317690188526,
0.743714927983187,
0.12759937884571806,0.1251635761156863,
-2.6350590749285696,
0.2804355809043281,
0.43287759848684004,0.6295650135141578,
0.07920529713113802,
0.26973320319620564,
0.1972566571348932,0.8739605610811444,
8.512869091989776,
0.2821203442039153,
-0.5324316129970709,0.8730986422470707,
-0.12778433410566006,
0.739075743598594,
-0.14295848359134458,
64.38663199488212,
-10.668957135963694,
-61.523270972531435,
-41.97378914447521,
21.2176985536655,
-72.2959731348862,
22.280851288651498,
-50.61496366805901,
6.153314452829522,
-34.05781366197828,
-54.52286665612474,
84.68432768635641,
55.40718905145974,
-25.325694172295474,
0.918889622186391,
-6.914130365685562,
57.44707662858251,
-5.452251684819687,
-36.71680926622921,
64.88872571009527,
-17.46942008868537,
-17.342191880970706,
-18.405783729277562,
80.20793479802354,
-8.66242260537452,
-25.398847284494934,
-2.418293899414471]

param3 = [0.125, 0.5289552626824476, 0.25, -0.04033653905748524, 0.625, -0.2270974565998023, 0.75, -0.48269615663160903, 0.875, 0.31876615823551346, 0.25, -0.0015631884543272356, 0.125, -0.18221078002929425, 0.75, -0.14903876056456178, 0.375, 0.8012414903187257, 0.25, 0.8541351946692537, 0.875, 0.7518187376448573, 0.75, 0.5043932257659425, 0.375, 0.23895511516268209, 0.75, -0.7579111414051483, 0.625, 0.13135158393032542, 0.25, 0.5757373521326064, 0.6107730718814808, -0.11346013873576766, 0.4489849467078215, -0.0013585367049404606, 0.8697620799951755, -0.5441769620447419, 0.3520030015527902, -0.9852288439690207, 0.3879001469525729, -0.8013999435819967, -0.10901899792977221, 0.5778933443172287, 0.6874817643621178, 0.2295241511055286, 0.16663123720808426, -0.5195579408962252, 0.56720017924877, -0.8171829178248728, 0.3392880461821861, -0.8212029275466046, -0.7429467672677275, -0.4301950804268031, -0.26884190145634546, -0.8514619105948855, -0.9408108514421218, 0.6830431089538092, 0.32822587650439994, 0.7973953781724057]
data = []


for x in range(8):
    plt.scatter(param3[x*4],param3[x*4+2])

plt.show()

for x in range(301):
    
    for y in range(301):

        data.append([x/300,y/300,nn.calcOuput(x/300,y/300,bruno_param)])


class1_x = []
class1_y = []
class2_x = []
class2_y = []
class3_x = []
class3_y = []

for line in data:


    if line[2][0] > line[2][1] and line[2][0] > line[2][2]:
        class1_x.append(float(line[0]))
        class1_y.append(float(line[1]))
    elif line[2][1] > line[2][0] and line[2][1] > line[2][2]:
        class2_x.append(float(line[0]))
        class2_y.append(float(line[1]))
    else:
        class3_x.append(float(line[0]))
        class3_y.append(float(line[1]))

plt.scatter(class1_x,class1_y, label = "Class 1")
plt.scatter(class2_x,class2_y, label = "Class 2")
plt.scatter(class3_x,class3_y, label = "Class 3")
plt.legend()
plt.show()
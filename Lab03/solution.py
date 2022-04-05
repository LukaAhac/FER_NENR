import sys
import fileinput
from time import sleep

flag = "min"

##################################################################
#                       LFunction class                          #
##################################################################

class LFunction():

    def __init__(self,alpha,betta):

        self.alpha = alpha
        self.betta = betta

    def valueAt(self, x):

        if x <= self.alpha:
            return 1.0
        elif x >= self.betta:
            return 0.0
        else:
            return (self.betta-x)/(self.betta-self.alpha)

##################################################################
#                     GammaFunction class                        #
##################################################################

class GammaFunction():

    def __init__(self,alpha,betta):

        self.alpha = alpha
        self.betta = betta

    def valueAt(self, x):

        if x <= self.alpha:
            return 0.0
        elif x >= self.betta:
            return 1.0
        else:
            return (x-self.alpha)/(self.betta-self.alpha)

##################################################################
#                     LambdaFunction class                       #
##################################################################

class LambdaFunction():

    def __init__(self,alpha,betta,gamma):

        self.alpha = alpha
        self.betta = betta
        self.gamma = gamma

    def valueAt(self, x):

        if x <= self.alpha:
            return 0.0
        elif x >= self.gamma:
            return 0.0
        elif self. betta > x > self.alpha:
            return (x-self.alpha)/(self.betta-self.alpha)
        else:
            return (self.gamma-x)/(self.gamma-self.betta)




class Rule():

    def __init__(self,*rulePart):

        self.rulePart = rulePart


    def zakljuci(self,L_D_LK_DK_V_S):

        vrijednosti_antecedenata = []
        zakljucak = dict()

        for x in self.rulePart[:-1]:
            if x[0] == "L":
                vrijednosti_antecedenata.append(x[1].valueAt(L_D_LK_DK_V_S[0]))
            elif x[0] == "D":
                vrijednosti_antecedenata.append(x[1].valueAt(L_D_LK_DK_V_S[1]))
            elif x[0] == "LK":
                vrijednosti_antecedenata.append(x[1].valueAt(L_D_LK_DK_V_S[2]))
            elif x[0] == "DK":
                vrijednosti_antecedenata.append(x[1].valueAt(L_D_LK_DK_V_S[3]))
            elif x[0] == "V":
                vrijednosti_antecedenata.append(x[1].valueAt(L_D_LK_DK_V_S[4]))
            elif x[0] == "S":
                vrijednosti_antecedenata.append(x[1].valueAt(L_D_LK_DK_V_S[5]))
        
        if flag == "min":
            antecedent = min(vrijednosti_antecedenata)
        else:
            antecedent = 1
            for each in vrijednosti_antecedenata:
                antecedent *= each
        
        for x in range(-90,91):

            if flag == "min":
                value = min(antecedent,self.rulePart[-1][1].valueAt(x))
            else:
                value = antecedent*self.rulePart[-1][1].valueAt(x)

            if value > 0:
                zakljucak[x] = value 

        return zakljucak



# kriticno_blizu = LFunction(30,50)
# blizu = LFunction(50,100)

kriticno_blizu = LFunction(30,50)
blizu = LFunction(50,100)

blago_lijevo = LambdaFunction(40,45,50)
blago_desno = LambdaFunction(-50,-45,-40)

naglo_lijevo = GammaFunction(70,90)
naglo_desno = LFunction(-90,-70)

sporo = LFunction(20,30)
brzo = GammaFunction(40,50)

nula = LFunction(0,1)
dovoljno_daleko = GammaFunction(30,50)

pravila_kormila = []

pravila_kormila.append(Rule(["L",blizu],["K",blago_desno]))
pravila_kormila.append(Rule(["D",blizu],["K",blago_lijevo]))
pravila_kormila.append(Rule(["L",kriticno_blizu],["K",naglo_desno]))
pravila_kormila.append(Rule(["D",kriticno_blizu],["K",naglo_lijevo]))

pravila_kormila.append(Rule(["LK",blizu],["K",blago_desno]))
pravila_kormila.append(Rule(["DK",blizu],["K",blago_lijevo]))
pravila_kormila.append(Rule(["LK",kriticno_blizu],["K",naglo_desno]))
pravila_kormila.append(Rule(["DK",kriticno_blizu],["K",naglo_lijevo]))

pravila_kormila.append(Rule(["S",nula],["D",dovoljno_daleko],["DK",dovoljno_daleko],["K",naglo_desno]))

pravila_brzine = []

pravila_brzine.append(Rule(["V",sporo],["V",brzo]))
pravila_brzine.append(Rule(["V",brzo],["V",sporo]))

#PROVJERA ZAKLJUČKA
if False:
    rule = Rule(["L",blizu],["K",blago_desno])
    zakljucak = rule.zakljuci([75,100,100,100,100,100])
    print(zakljucak)

    CoAu = 0
    CoAd = 0
    for key in zakljucak.keys():
        CoAu += zakljucak[key] * key
        CoAd += zakljucak[key]

    if CoAd == 0:
        K = 0
    else:
        K = round(CoAu/CoAd)
    print(K)

# PROVJERA UNIJE ZAKLJUČAKA
if False:
    zakljucak_kormila = dict()
    for pravilo in pravila_kormila:
        zakljucak = pravilo.zakljuci([30,100,100,100,100,100])

        for key in zakljucak.keys():

            if key in zakljucak_kormila:

                if zakljucak[key] > zakljucak_kormila[key]:
                    zakljucak_kormila[key] = zakljucak[key]
            else:
                zakljucak_kormila[key] = zakljucak[key]


    CoAu = 0
    CoAd = 0
    for key in zakljucak_kormila.keys():
        CoAu += zakljucak_kormila[key] * key
        CoAd += zakljucak_kormila[key]

    if CoAd == 0:
        K = 0
    else:
        K = round(CoAu/CoAd)   
    print(zakljucak_kormila)
    print(K)


#POČETAK MAINA
while True:
    

    line = input()

    zakljucak_kormila = dict()

    if line == "KRAJ":
        break

    L_D_LK_DK_V_S = line.strip().split()
    for index in range(len(L_D_LK_DK_V_S)):
        L_D_LK_DK_V_S[index] = int(L_D_LK_DK_V_S[index])

    for pravilo in pravila_kormila:
        zakljucak = pravilo.zakljuci(L_D_LK_DK_V_S)

        for key in zakljucak.keys():

            if key in zakljucak_kormila:

                if zakljucak[key] > zakljucak_kormila[key]:
                    zakljucak_kormila[key] = zakljucak[key]
            else:
                zakljucak_kormila[key] = zakljucak[key]


    CoAu = 0
    CoAd = 0
    for key in zakljucak_kormila.keys():
        CoAu += zakljucak_kormila[key] * key
        CoAd += zakljucak_kormila[key]

    if CoAd == 0:
        K = 0
    else:
        K = round(CoAu/CoAd)


    zakljucak_V = dict()

    for pravilo in pravila_brzine:
        zakljucak = pravilo.zakljuci(L_D_LK_DK_V_S)

        for key in zakljucak.keys():

            if key in zakljucak_V:

                if zakljucak[key] > zakljucak_V[key]:
                    zakljucak_V[key] = zakljucak[key]
            else:
                zakljucak_V[key] = zakljucak[key]


    CoAu = 0
    CoAd = 0
    for key in zakljucak_V.keys():
        CoAu += zakljucak_V[key] * key
        CoAd += zakljucak_V[key]

    if CoAd == 0:
        V = 0
    else:
        V = round(CoAu/CoAd)



    print("{} {}".format(V,K),flush = True)
    sys.stdout.flush()
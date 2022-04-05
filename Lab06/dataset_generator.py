#Utility to generate input training data

from math import cos
from math import pi

def function(x,y):

    return ((x-1)**2+(y+2)**2-5*x*y+3)* (cos(pi/5))**2


def generate_data():

    xes = [x for x in range(-4,5)]
    yes = [x for x in range(-4,5)]

    z = list()

    for x in xes:
        for y in yes:
            z.append([x,y])

    toRet = [[(x[0],x[1]),function(x[0],x[1])] for x in z]
    
    return toRet

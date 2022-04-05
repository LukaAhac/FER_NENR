#Utility to draw plots

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset_generator import generate_data
import numpy as np


def draw_inputdata():

    data = generate_data()

    X = [x[0][0] for x in data]
    Y = [x[0][1] for x in data]
    Z = [x[1] for x in data]

    draw3d(X,Y,Z)


def draw3d(X,Y,Z):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(X,Y,Z, color='b')
    plt.show()



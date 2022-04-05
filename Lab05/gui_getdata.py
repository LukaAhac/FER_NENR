#Utility function get_data that opens a canvas to draw a symbol for prediction

from tkinter import *
import numpy as np
from math import sqrt

toReturn = None

def get_data():

    canvas_width = 500
    canvas_height = 500
    M = 50

    global toReturn

    xKord = []
    yKord = []

    def paint( event ):
        w.create_line(event.x, event.y, event.x+1, event.y, fill="#476042")
        xKord.append(event.x)
        yKord.append(event.y)

    def reset():
        xKord.clear()
        yKord.clear()
        w.delete("all")


    def done():
        global toReturn

        toReturn = None

        master.destroy()

    def return_data():

        global toReturn
        x = np.array(xKord)
        y = np.array(yKord)
        xKord.clear()
        yKord.clear()
        
        xMean = np.mean(x)
        yMean = np.mean(y)

        x = x - xMean
        y = y - yMean

        xMax = np.amax(np.abs(x))
        yMax = np.amax(np.abs(y))

        m = max(xMax,yMax)

        x = x / m
        y = y / m

        D = 0

        for index in range(0,len(x)-1):
            D += sqrt((x[index]-x[index+1])**2+(y[index]-y[index+1])**2)

        xSampled = []
        ySampled = []
        k = 0
        distance = 0

        for index in range(0,len(x)-1):
            if k*D/(M-1) <= distance:
                xSampled.append(x[index])
                ySampled.append(y[index])
                k += 1
            distance += sqrt((x[index]-x[index+1])**2+(y[index]-y[index+1])**2)

        xSampled.append(x[-1])
        ySampled.append(y[-1])

        X = []
        for index in range(50):
            X.append(xSampled[index])
            X.append(ySampled[index])

        X = np.array([X])
        X = np.transpose(X)

        toReturn = X

        master.destroy()



    master = Tk()
    master.title( "Crtanje" )
    w = Canvas(master, 
            width=canvas_width, 
            height=canvas_height)
    w.pack(expand = YES, fill = BOTH)
    w.bind( "<B1-Motion>", paint )

    BS = Button(master, text ="PREDICT", command = return_data)
    BS.pack()
    BR = Button(master, text ="RESET", command = reset)
    BR.pack()
    BD = Button(master, text ="DONE", command = done)
    BD.pack()
        
    w.mainloop()

    return toReturn

#Utility file to create training data set by drawing symbols into the canvas

from tkinter import *
import numpy as np
from math import sqrt

canvas_width = 500
canvas_height = 500
M = 50
counter = 0

file = open(r"Lab05\data.txt", "w")


xKord = []
yKord = []

def paint( event ):
   w.create_line(event.x, event.y, event.x+1, event.y, fill="#476042")
   xKord.append(event.x)
   yKord.append(event.y)

def save():

    global counter

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

    for index in range(50):
        file.write("{},{};".format(xSampled[index],ySampled[index]))

    if counter < 20:
        file.write("10000\n")
    elif counter < 40:
        file.write("01000\n")
    elif counter < 60:
        file.write("00100\n")
    elif counter < 80:
        file.write("00010\n")
    else:
        file.write("00001\n")

    counter += 1

    w.delete("all")

def reset():
    xKord.clear()
    yKord.clear()
    w.delete("all")

master = Tk()
master.title( "Crtanje" )
w = Canvas(master, 
           width=canvas_width, 
           height=canvas_height)
w.pack(expand = YES, fill = BOTH)
w.bind( "<B1-Motion>", paint )

BS = Button(master, text ="SAVE", command = save)
BS.pack()
BR = Button(master, text ="RESET", command = reset)
BR.pack()
    
w.mainloop()

file.close()
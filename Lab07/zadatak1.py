import numpy as np
import matplotlib.pyplot as plt


y = lambda x,w,s: 1/(1+abs(x-w)/s)



x_axis = list(range(-800,1001))
x_axis = np.array(x_axis)
x_axis = x_axis / 100

y_axis_s1 = []
y_axis_s025 = []
y_axis_s4 = []

for x in x_axis:
    y_axis_s1.append(y(x,2,1))
    y_axis_s025.append(y(x,2,0.25))
    y_axis_s4.append(y(x,2,4))


plt.plot(x_axis,y_axis_s1, label = "S = 1")
plt.plot(x_axis,y_axis_s025, label = "S = 0.25")
plt.plot(x_axis,y_axis_s4, label = "S = 4")
plt.legend()
plt.show()
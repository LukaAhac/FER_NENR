import matplotlib.pyplot as plt

with open(r"Lab07\dataset.txt","r") as f:
    data = f.readlines()


class1_x = []
class1_y = []
class2_x = []
class2_y = []
class3_x = []
class3_y = []

for line in data:

    line = line.strip().split()

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
plt.show()
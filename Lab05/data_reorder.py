#Utility file to reorder training set data in order to be ready for batch learning

toRead = open(r"Lab05\testData.txt","r")
toWrite = open(r"Lab05\testData_reorderd1.txt", "w")

lines = toRead.readlines()


for index1 in range(10):
    for index2 in range(5):

        toWrite.write(lines[index2*20+index1*2])
        toWrite.write(lines[index2*20+index1*2+1])

toRead.close()
toWrite.close()
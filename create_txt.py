import numpy as np
import json
import cv2
import os
import time
import matplotlib.pyplot as plt


def sh(x):
	plt.imshow(x)
	plt.show()
	return x

scale = 1/2
resize_1 = int(864*scale)
resize_2 = int(1296*scale)

f =  open("training_GT_labels_v2.json","r")
parsed_json = json.loads(f.read())
lst = os.listdir("Data_Training/")
images = []
labels = []
asd = ""

for i in range(len(lst)):
    stupidbrokendata = np.asarray(parsed_json[lst[i]][0])
    if stupidbrokendata.shape != (8,):
        print(str(i) + "\t" + lst[i] + ":\t" + str(stupidbrokendata.shape))
    elif stupidbrokendata[0] >= stupidbrokendata[4] or stupidbrokendata[1] >= stupidbrokendata[5]:
        print(str(i) + "backwards" + "\t" + lst[i] + ":\t" + str(stupidbrokendata))
    else:
        asd += "Data_Training/" + lst[i] + ","+ str(parsed_json[lst[i]][0][0]) + ","+ str(parsed_json[lst[i]][0][1]) + ","+  str(parsed_json[lst[i]][0][4]) + "," +  str(parsed_json[lst[i]][0][5]) + "," + "gate" + "\n"
with open('set.csv', 'w') as file:
    file.write(asd)

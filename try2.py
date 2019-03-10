import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import datetime
import os
import random
import csv

resize_1 = 256 #864
resize_2 = 256 #1296
dir = "Data_LeaderboardTesting/"

lst = []
with open('results.txt', 'r') as f:
    reader = csv.reader(f)
    lst = list(reader)

rsz = np.zeros((1,256,256,3))
for i in range(len(lst)):
    print(dir + lst[i][0])
    img = cv2.imread(dir + lst[i][0])
    #print(img)
    rsz[0] = cv2.resize(img, (256, 256),interpolation = cv2.INTER_CUBIC)
    rsz = rsz.astype("uint8")
    print(lst[i][:])
    print(str(lst[i][1]) + "\t" + str(lst[i][3]) + "\t" + str(lst[i][2]) + "\t" + str(lst[i][4]))
    x1 = int(float(lst[i][1]))
    x2 = int(float(lst[i][3]))
    y1 = int(float(lst[i][2]))
    y2 = int(float(lst[i][4]))
    plt.imshow(rsz[0])
    plt.plot([x1,x2],[y1,y1], color='#00ff00', linestyle='-', linewidth=3)
    plt.plot([x2,x2],[y1,y2], color='#00ff00', linestyle='-', linewidth=3)
    plt.plot([x2,x1],[y2,y2], color='#00ff00', linestyle='-', linewidth=3)
    plt.plot([x1,x1],[y2,y1], color='#00ff00', linestyle='-', linewidth=3)
    #plt.plot([abc[0], abc[2]], [abc[1], abc[2]], color='#00ff00', linestyle='-', linewidth=3)
    #plt.plot([abc[1], abc[2]], [abc[], abc[2]], color='#00ff00', linestyle='-', linewidth=3)
    #plt.plot([abc[6], abc[0]], [abc[7], abc[1]], color='#00ff00', linestyle='-', linewidth=3)
    plt.show()


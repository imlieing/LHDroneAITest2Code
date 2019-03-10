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
    x1 = int(float(lst[i][1]))
    x2 = int(float(lst[i][2]))
    x3 = int(float(lst[i][3]))
    x4 = int(float(lst[i][4]))
    y1 = int(float(lst[i][5]))
    y2 = int(float(lst[i][6]))
    y3 = int(float(lst[i][7]))
    y4 = int(float(lst[i][8]))
    """
    x1 = 0
    y1 = 0
    x2 = 255
    y2 = 255
    x3 = 0
    y3 = 0
    x4 = 255
    y4 = 255
    """
    x3 -= 256/2
    y3 -= 256/2
    x4 -= 256/2
    y4 -= 256/2
    theta = np.radians(-90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    mat1 = np.array(((x3),(y3)))
    mat2 = np.array(((x4),(y4)))
    print(mat1)
    print(mat2)
    Rmat1 = np.matmul(R,mat1)
    Rmat2 = np.matmul(R,mat2)
    x3 = Rmat1[0] + 256/2
    y3 = Rmat1[1] + 256/2
    x4 = Rmat2[0] + 256/2
    y4 = Rmat2[1] + 256/2
    plt.imshow(rsz[0])
    plt.plot([x1,x2],[y1,y2], color='#00ff00', linestyle='-', linewidth=3)
    #plt.plot([x2,x3],[y2,y3], color='#00ff00', linestyle='-', linewidth=3)
    plt.plot([x3,x4],[y3,y4], color='#00ff00', linestyle='-', linewidth=3)
    #plt.plot([x4,x1],[y4,y1], color='#00ff00', linestyle='-', linewidth=3)
    #plt.plot([abc[0], abc[2]], [abc[1], abc[2]], color='#00ff00', linestyle='-', linewidth=3)
    #plt.plot([abc[1], abc[2]], [abc[], abc[2]], color='#00ff00', linestyle='-', linewidth=3)
    #plt.plot([abc[6], abc[0]], [abc[7], abc[1]], color='#00ff00', linestyle='-', linewidth=3)
    plt.show()

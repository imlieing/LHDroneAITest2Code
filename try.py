
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU
import keras.backend as K
from keras_retinanet.models import load_model


import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import datetime
import os
import random



resize_1 = 256 #864
resize_2 = 256 #1296
dir = "Data_LeaderboardTesting/"
#f =  open("training_GT_labels_v2.json","r")
#parsed_json = json.loads(f.read())
#custom_objects = retinanet.custom_objects.copy()
#custom_objects.update(keras_resnet.custom_objects)
model = load_model('snapshots/inference_model_15.h5', backbone_name='resnet50')

lst = os.listdir(dir)
rsz = np.zeros((1,256,256,3))
s = ""
for i in range(50):
    img = cv2.imread(dir + lst[i])
    rsz[0] = cv2.resize(img, (256, 256))
    abc = model.predict(rsz)[0]
    print(abc)
    s += lst[i] + "," + abc[1] + "," + abc[2] + "," + abc[3] + "," + abc[4] + "\n"
with open("results.txt", "w") as text_file:
    text_file.write(s)

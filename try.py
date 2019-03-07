
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
for i in range(len(lst)):
    img = cv2.imread(dir + lst[i])
    rsz[0] = cv2.resize(img, (256, 256))
    abc = model.predict(rsz)[0]
    print(abc)
    plt.imshow(rsz[0])
    x1 = abc[0]
    y1 = abc[1]
    x2 = abc[2]
    y2 = abc[3]
    plt.plot([x1, x2], [y1,y1], color='#00ff00', linestyle='-', linewidth=3)
    plt.plot([x2, x2], [y1,y2], color='#00ff00', linestyle='-', linewidth=3)
    plt.plot([x2, x1], [y2,y2], color='#00ff00', linestyle='-', linewidth=3)
    plt.plot([x1, x1], [y2,y1], color='#00ff00', linestyle='-', linewidth=3)
    plt.show()


################################################################################################
###############################################################################################
#################################################################################################
################################################################################################

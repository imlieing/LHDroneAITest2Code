
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import LeakyReLU
import keras.backend as K

import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import datetime
import os
import random

import load_data

def sh(x):
	plt.imshow(x)
	plt.show()
	return x
# gray = cv2.cvtColor(mod[0], cv2.COLOR_BGR2GRAY)
# sh(cv2.dilate(cv2.cornerHarris(cv2.blur(gray,(3,3)),2,3,0.04),None))


scale = 1/2
resize_1 = int(864*scale)
resize_2 = int(1296*scale)
batch_size = 8
dir = "Data_Training_2/"
f =  open("training_GT_labels_v2.json","r")
parsed_json = json.loads(f.read())

def load_multi_image(lst):
	X = np.zeros((len(lst),resize_1,resize_2,1))
	for i in range(len(lst)):
		X[i] = cv2.imread(dir + lst[i])[:,:,0:1] # it is supposed to be 1 channel but its 3 with all the same values...
	return X

def load_multi_label(lst):
	Y = np.zeros((len(lst),8))
	for i in range(len(lst)):
		Y[i] = parsed_json[lst[i]][0]
	return Y


def generate_data(batch_size):
	lst = os.listdir(dir)
	L = int(len(lst)*0.8)
	while True:
		i = 0
		batch_start = 0
		batch_end = batch_size
		b = 0
		while batch_start < L:
			limit = min(batch_end, L)
			X = load_multi_image(lst[batch_start:limit])
			Y = load_multi_label(lst[batch_start:limit])
			
			yield (X,Y) #a tuple with two numpy arrays with batch_size samples

			batch_start += batch_size
			batch_end += batch_size


################################################################################################
###############################################################################################
#################################################################################################
################################################################################################


leak = 0.3
model = Sequential()

model.add(Conv2D(16, kernel_size=(3,3), #orig 32 filters
	#activation=act,
	input_shape=(resize_1,resize_2,1),
	))
model.add(LeakyReLU(alpha=leak))
model.add(Conv2D(32, kernel_size=(3,3) #orig 32 filters
	#activation=act,
	))
model.add(LeakyReLU(alpha=leak))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(LeakyReLU(alpha=leak))
#coordinates
model.add(Dense(8))

def mean_squared_error(y_true, y_pred):
	return K.mean(K.square(y_pred - y_true), axis=-1)

model.compile(loss='mean_squared_error',#'mean_squared_error' iou_loss.iou_loss
	optimizer='Adadelta',#
	metrics=[mean_squared_error])#not 'accuracy' iou_metric.iou_metric

#https://datascience.stackexchange.com/questions/18414/are-there-any-rules-for-choosing-the-size-of-a-mini-batch


#OLD: len(os.listdir("Data_Training_2/")[:int(len(os.listdir("Data_Training_2/") * 0.8))])
history = model.fit_generator(
	generate_data(batch_size),
	steps_per_epoch = int(len(os.listdir("Data_Training_2/")) * 0.8 // batch_size),
	#validation_data = generate_validation("Data_Training_2/", batch_size),
	#validation_steps = int(len(os.listdir("Data_Training_2/")) * 0.2 // batch_size),
	epochs=500,#50
	verbose=1
	)

"""
history = model.fit(images_new, labels,
		batch_size=32,#orig 128
		epochs=500,#50
		verbose=1,
		validation_split = 0.2)
"""
#score = model.evaluate(images_new, labels, verbose=0)
#print('Test loss:', score[0])

fuckfuckfuck = cv2.imread(os.listdir(dir)[0])
abc = model.predict(fuckfuckfuck)[0]
for i1 in range(4):
	abc[2*i1] *= 1296*scale
	abc[2*i1+1] *= 864*scale
print(abc)
imgplot = plt.imshow(images_new[0,:,:,0])
plt.plot([abc[0], abc[2]], [abc[1], abc[3]], color='#00ff00', linestyle='-', linewidth=3)
plt.plot([abc[2], abc[4]], [abc[3], abc[5]], color='#00ff00', linestyle='-', linewidth=3)
plt.plot([abc[4], abc[6]], [abc[5], abc[7]], color='#00ff00', linestyle='-', linewidth=3)
plt.plot([abc[6], abc[0]], [abc[7], abc[1]], color='#00ff00', linestyle='-', linewidth=3)
plt.show()

#s = "my_models/model_"+datetime.datetime.now().strftime("%Y-%m-%d---%H-%M-%S") + ".h5"
s = "my_models/model_junk" + ".h5"
print(s)
model.save(s)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


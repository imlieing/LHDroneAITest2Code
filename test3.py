
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

import load_data

def sh(x):
	plt.imshow(x)
	plt.show()
	return x
# gray = cv2.cvtColor(mod[0], cv2.COLOR_BGR2GRAY)
# sh(cv2.dilate(cv2.cornerHarris(cv2.blur(gray,(3,3)),2,3,0.04),None))


images_base, labels_base = load_data.load_small_clean(1000)
scale = 1/2
resize_1 = int(864*scale)
resize_2 = int(1296*scale)
images = np.zeros((len(images_base),resize_1,resize_2))
labels = labels_base
for i in range(len(images_base)):
	mod = images_base[i,:,:,:]
	asd=cv2.dilate(
			cv2.cornerHarris(
				cv2.blur(
					cv2.cvtColor(
						cv2.resize(
							mod, dsize=(resize_2,resize_1), interpolation=cv2.INTER_CUBIC
						), cv2.COLOR_BGR2GRAY
					),(3,3)
				),2,3,0.04
			),None
			)

	kernel = np.ones((70,70),np.float32)/1
	asd2=cv2.filter2D(asd,-1,kernel)

	asd2_copy = asd2*256
	asd2_copy[asd2_copy < 0] = 0
	#sh(asd2_copy)
	asd2_copy = np.uint8(asd2_copy)
	#sh(asd2_copy)
	#asd2_copy_blurred=sh(cv2.blur(asd2_copy,(3,3)))
	#asd3=sh(cv2.Canny(asd2_copy,5,10))
	images[i] = asd2_copy
	#sh(images[i])
	for i1 in range(4):
		labels[i,2*i1] /= 1296
		labels[i,2*i1+1] /= 864

images_new = np.zeros((images.shape[0],images.shape[1],images.shape[2],1))
images_new[:,:,:,0]=images
print(images_new.shape)
################################################################################################
################################################################################################
################################################################################################
################################################################################################

leak = 0.3
model = Sequential()

model.add(Conv2D(4, kernel_size=(3,3), #orig 32 filters
    #activation=act,
    input_shape=(images_new.shape[1:]),
    ))
model.add(LeakyReLU(alpha=leak))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64))
model.add(LeakyReLU(alpha=leak))
#coordinates
model.add(Dense(8))

def mean_squared_error(y_true, y_pred):    
    return K.mean(K.square(y_pred - y_true), axis=-1)

model.compile(loss='mean_squared_error',#'mean_squared_error' iou_loss.iou_loss
              optimizer='Adadelta',#
              metrics=[mean_squared_error])#not 'accuracy' iou_metric.iou_metric

#https://datascience.stackexchange.com/questions/18414/are-there-any-rules-for-choosing-the-size-of-a-mini-batch

history = model.fit(images_new, labels,
        batch_size=64,#orig 128
        epochs=100,#50
        verbose=1,
        validation_split = 0.2)
score = model.evaluate(images_new, labels, verbose=0)
print('Test loss:', score[0])

abc = model.predict(images_new[0:1,:,:,:])[0]
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



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

#import iou_loss
#import iou_metric

import load_data


np.random.seed(7)
images_base, labels_base = load_data.load_small_clean(5000)
print("loaded")
# resizing labels and images
scale = 8
resize_1 = int(864/scale)
resize_2 = int(1296/scale)
# I DON"T KNOW IF I ACCIDENTALLY FLIPPED OR NOT
images = np.zeros((len(images_base),resize_1,resize_2,3))
#labels = np.zeros((len(labels_base),8))
labels = labels_base
for i in range(len(images)):
        #print(images_base[i,20,20,:])
        #print(images_base.shape)
        #plt.imshow(images_base[i,:,:,:])
        #plt.show()
        images[i] = cv2.resize(images_base[i], dsize=(resize_2,resize_1), interpolation=cv2.INTER_CUBIC)
        #plt.imshow(images[i,:,:,:])
        #plt.show()
        #print(images[i,20,20,:])
        #print(images.shape)
        # labels are clockwise circle from top left corner to bottom left corner
        """
        for i1 in range(4):
                labels[i,i1] = labels_base[i,i1]*resize_1/864
                labels[i,i1+1] = labels_base[i,i1+1]*resize_2/1296
        """
        # I FORGOT TO MAKE FROM 0 to 1
        #WHICH ONE IS WHICH LOL
        for i1 in range(4):
                labels[i,2*i1] /= 1296
                labels[i,2*i1+1] /= 864
images = images.astype(int)
#plt.imshow(images[i,:,:,:])
#plt.show()
print("resized")


leak = 0.3
model = Sequential()
model.add(Conv2D(16, kernel_size=(3,3), #orig 32 filters
    #activation=act,
    input_shape=(resize_1,resize_2,3),
    data_format="channels_last"
    ))
model.add(LeakyReLU(alpha=leak))
model.add(Conv2D(32, (3,3))) # orig 64 filters
model.add(LeakyReLU(alpha=leak))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(LeakyReLU(alpha=leak))
model.add(Dropout(0.5))
model.add(Dense(128))
model.add(LeakyReLU(alpha=leak))
model.add(Dropout(0.1))
model.add(Dense(128))
#coordinates
model.add(Dense(8))
model.add(LeakyReLU(alpha=leak))

def soft_acc(y_true, y_pred):
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

model.compile(loss='mean_squared_error',#'mean_squared_error' iou_loss.iou_loss
              optimizer='Adadelta',#
              metrics=[soft_acc])#not 'accuracy' iou_metric.iou_metric

#https://datascience.stackexchange.com/questions/18414/are-there-any-rules-for-choosing-the-size-of-a-mini-batch

history = model.fit(images, labels,
        batch_size=64,#orig 128
        epochs=100,#22
        verbose=1,
        validation_split = 0.2)
score = model.evaluate(images, labels, verbose=0)
print('Test loss:', score[0])

abc = model.predict(images[0:1,:,:,:])[0]
for i1 in range(4):
        abc[2*i1] *= 1296/8
        abc[2*i1+1] *= 864/8
print(abc)
imgplot = plt.imshow(images[0,:,:,:])
plt.plot([abc[0], abc[2]], [abc[1], abc[3]], color='#00ff00', linestyle='-', linewidth=10)
plt.plot([abc[2], abc[4]], [abc[3], abc[5]], color='#00ff00', linestyle='-', linewidth=10)
plt.plot([abc[4], abc[6]], [abc[5], abc[7]], color='#00ff00', linestyle='-', linewidth=10)
plt.plot([abc[6], abc[0]], [abc[7], abc[1]], color='#00ff00', linestyle='-', linewidth=10)
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
"""
model.fit(images, labels,
        batch_size=128,
        epochs=1,#22
        verbose=1,
        validation_split = 0.2)
"""
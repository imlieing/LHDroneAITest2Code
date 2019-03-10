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
for i in range(len(lst)):
	stupidbrokendata = np.asarray(parsed_json[lst[i]][0])
	if stupidbrokendata.shape != (8,):
		print(str(i) + "\t" + lst[i] + ":\t" + str(stupidbrokendata.shape))
	else:
		img = cv2.imread("Data_Training/" + lst[i])
		#print(parsed_json[lst[i]][0])
		#labels.append(parsed_json[lst[i]][0])
		mod = img.copy()
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
		new_img = asd2_copy
		rsz = cv2.resize(
								img[:,:,1],
								dsize=(resize_2,resize_1),
								interpolation=cv2.INTER_CUBIC
							)
		#print(rsz.shape)
		#print(new_img.shape)
		new_img = new_img + rsz*0.12
		#sh(new_img)
		cv2.imwrite("Data_Training_2/" + lst[i],new_img)
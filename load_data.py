import numpy as np
import json
import cv2
import os
import time
import matplotlib.pyplot as plt


def load():
	f =  open("training_GT_labels_v2.json","r")
	parsed_json = json.loads(f.read())
	lst = os.listdir("Data_Training/")
	images = []
	labels = []
	for i in range(len(lst)):
		images.append(cv2.imread("Data_Training/" + lst[i]))
		labels.append(parsed_json[lst[i]])
		#print(labels[i])
		#print(images[i])
	 # im too lazy to check what this does
	 # but stackoverflow had it and hopefully it does what it seems like it does lol
	images = np.asarray(images)
	labels = np.asarray(labels)
	#print(images.shape())
	#print(labels)
	return images, labels

def load_small(amnt, offset=0):
	f =  open("training_GT_labels_v2.json","r")
	parsed_json = json.loads(f.read())
	lst = os.listdir("Data_Training/")
	images = []
	labels = []
	for i in range(offset,amnt+offset):#len(lst)
		images.append(cv2.imread("Data_Training/" + lst[i]))
		#print(parsed_json[lst[i]][0])
		labels.append(parsed_json[lst[i]][0])
		stupidbrokendata = np.asarray(parsed_json[lst[i]][0])
		if(stupidbrokendata.shape != (8,)):
			print(str(i) + "\t" + lst[i] + ":\t" + str(stupidbrokendata.shape))
		#print(labels[i])
		#print(images[i])
	 # im too lazy to check what this does
	 # but stackoverflow had it and hopefully it does what it seems like it does lol
	images = np.asarray(images)
	labels = np.asarray(labels)
	#print(labels.shape)
	#labels = labels[:,0,:]
	#print(images.shape)
	#print(labels)
	return images, labels

def load_small_clean(amnt, offset=0):
	f =  open("training_GT_labels_v2.json","r")
	parsed_json = json.loads(f.read())
	lst = os.listdir("Data_Training/")
	images = []
	labels = []
	for i in range(offset,amnt+offset):#len(lst)
		stupidbrokendata = np.asarray(parsed_json[lst[i]][0])
		if(stupidbrokendata.shape != (8,)):
			print(str(i) + "\t" + lst[i] + ":\t" + str(stupidbrokendata.shape))
		else:
			images.append(cv2.imread("Data_Training/" + lst[i]))
			#print(parsed_json[lst[i]][0])
			labels.append(parsed_json[lst[i]][0])
		#print(labels[i])
		#print(images[i])
	 # im too lazy to check what this does
	 # but stackoverflow had it and hopefully it does what it seems like it does lol
	images = np.asarray(images)
	labels = np.asarray(labels)
	#print(labels.shape)
	#labels = labels[:,0,:]
	#print(images.shape)
	#print(labels)
	return images, labels

if __name__ == "__main__":
	#for i in range(10):
	#	images, labels = load_small(1000,offset=1000*i)
	#	time.sleep(0.1)
	images, labels = load_small(50)
	print(images.shape)
	print(labels.shape)
	gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
	mod = images[0,:,:,1]
	print(mod.shape)
	#mod0 = images[0,:,:,0]
	#mod2 = images[0,:,:,2]
	#plt.imshow(mod0)
	#plt.show()
	cv2.imshow("mod",mod)
	cv2.waitKey(0)

	#plt.show()
	#plt.imshow(mod2)
	#plt.show()
	cv2.imshow("gray",gray)
	cv2.waitKey(0)

	#plt.show()
	modb = mod - gray
	cv2.imshow("mod - gray",modb)
	cv2.waitKey(0)

	#plt.show()
	
	#print(labels.shape)
	#print("labels: ")
	#print(labels[1,:])
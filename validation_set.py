import os
import numpy as np
from sklearn.model_selection import train_test_split

PATH = "D:/Data/carvana/"
test_size = 0.3


train_path = PATH + "train/train/folder/"
train_path_val = PATH + "train/validation/folder/"
label_path = PATH + "train_masks/train_masks/folder/"
label_path_val = PATH + "train_masks/validation_masks/folder/"

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def moveImage(data, directory):
	for image in data:
		if (not directory in image):
			name = image.split("/")[-1]
			os.rename(image, directory + name)


train_images = np.array(listdir_fullpath(train_path) + listdir_fullpath(train_path_val))
label_images = np.array(listdir_fullpath(label_path) + listdir_fullpath(label_path_val))

num_images = len(train_images)
num_cars = int(num_images / 16)

i_train, i_test = train_test_split(range(num_cars), test_size=test_size)

train_images = train_images.reshape(num_cars, 16)
label_images = label_images.reshape(num_cars, 16)

X_train = train_images[i_train].flatten()
X_test = train_images[i_test].flatten()
y_train = label_images[i_train].flatten()
y_test = label_images[i_test].flatten()

moveImage(X_train, train_path)
moveImage(X_test, train_path_val)
moveImage(y_train, label_path)
moveImage(y_test, label_path_val)
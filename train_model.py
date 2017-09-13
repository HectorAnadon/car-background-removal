import tensorflow as tf
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from u_net import UNet
import numpy as np
import os


# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.89
# set_session(tf.Session(config=config))

PATH = "D:/Data/carvana/"
IMAGE_SIZE = 512
batch_size = 3
net_depth = 2
epochs = 10

optimizer = RMSprop(lr=0.001, decay=0.0000001)
# optimizer = SGD(lr=0.01)
# optimizer = Adam(lr=1e-3, decay=0.995)
# optimizer = Adam(decay=0.01)

# Generator for training
datagen = dict(
	rotation_range=10,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.1,
	rescale=1./255,
	zoom_range=0.2,
	horizontal_flip=True,
	channel_shift_range=0.05,
	fill_mode='nearest'
	)

num_training = len(os.listdir(PATH + "train_masks/train_masks/folder/"))
num_validation = len(os.listdir(PATH + "train_masks/validation_masks/folder/"))

model = UNet((IMAGE_SIZE,IMAGE_SIZE,3), depth=net_depth)
# print(model.summary())


def dice_coef(y_true, y_pred):
    smooth = 1e-5
    
    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))
    
    isct = tf.reduce_sum(y_true * y_pred)
    
    return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))


model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[dice_coef])
checkpoint = ModelCheckpoint('weight.hdf5', monitor='val_dice_coef', verbose=1, save_best_only=True,
								 save_weights_only=True, mode='max')



image_datagen = ImageDataGenerator(**datagen)
mask_datagen = ImageDataGenerator(**datagen)

val_datagen = ImageDataGenerator(rescale=1./255)
val_datagen_mask = ImageDataGenerator(rescale=1./255)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1

image_generator = image_datagen.flow_from_directory(
    PATH + "train/train",
    class_mode=None,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=batch_size,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    PATH + "train_masks/train_masks",
    class_mode=None,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=batch_size,
    color_mode="grayscale", 
    seed=seed)

val_image_generator = val_datagen.flow_from_directory(
    PATH + "train/validation",
    class_mode=None,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=batch_size,
    seed=seed)

val_mask_generator = val_datagen_mask.flow_from_directory(
    PATH + "train_masks/validation_masks",
    class_mode=None,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=batch_size,
    color_mode="grayscale", 
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=num_training/batch_size,
    epochs=epochs,
	verbose=1,
	callbacks=[checkpoint],
	validation_data=val_generator,
	validation_steps=num_validation/batch_size
#   # class_weight=None,
#   # max_queue_size=10,
#   # workers=1,
#   # use_multiprocessing=False,
#   # initial_epoch=0
  	)
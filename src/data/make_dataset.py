from numpy.random import seed
seed(8) #1
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from os import listdir
import pickle
import cv2

##
data_list = listdir( r"C:\Users\saimunikoti\Manifestation\Images-Learning\data\processed\covid-19\four_classes\train")
DATASET_PATH  = r"C:\Users\saimunikoti\Manifestation\Images-Learning\data\processed\covid-19\four_classes\train"
test_dir =  r"C:\Users\saimunikoti\Manifestation\Images-Learning\data\processed\covid-19\four_classes\test"
IMAGE_SIZE    = (150, 150)
NUM_CLASSES   = len(data_list)
BATCH_SIZE    = 10  # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    = 10
LEARNING_RATE = 0.0001

##
datagen = ImageDataGenerator(rescale=1./255, validation_split = 0.2)
datagen_augment = ImageDataGenerator(rescale=1./255, validation_split = 0.2,
                                   rotation_range=50,
                                   featurewise_center = True,
                                   featurewise_std_normalization = True,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.25,
                                   zoom_range=0.1,
                                   zca_whitening = True,
                                   channel_shift_range = 20,
                                   horizontal_flip = True ,
                                   vertical_flip = True ,
                                   fill_mode='constant')

# For multiclass use categorical n for binary use binary
train_generator = datagen.flow_from_directory(DATASET_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset = "training",
                                                  seed=42,
                                                  class_mode="categorical"   #For multiclass use categorical n for binary use binary
                                                  )

valid_generator = datagen.flow_from_directory(DATASET_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset="validation",
                                                  seed=42,
                                                  class_mode="categorical"
                                                  # For multiclass use categorical n for binary use binary
                                                  )

train_generator_augment = datagen_augment.flow_from_directory(DATASET_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset = "training",
                                                  seed=42,
                                                  class_mode="categorical"   #For multiclass use categorical n for binary use binary
                                                  )

valid_generator_augment = datagen_augment.flow_from_directory(DATASET_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset="validation",
                                                  seed=42,
                                                  class_mode="categorical"
                                                  )

## train data

Xagg = np.empty(shape=[0,150,150,3])
yagg = np.empty(shape=[0,4])
batches = 0

for X, y in train_generator:
    # print(X.shape, y.shape)
    Xagg = np.concatenate((Xagg,X), axis=0)
    yagg = np.concatenate((yagg,y), axis=0)
    batches = batches+1

    if batches > len(train_generator):
        break

# rotation image
batches = 0

for X, y in train_generator_augment:
    # print(X.shape, y.shape)
    Xagg = np.concatenate((Xagg,X), axis=0)
    yagg = np.concatenate((yagg,y), axis=0)
    batches = batches+1

    if batches > 20:
        break

filepath = r"C:\Users\saimunikoti\Manifestation\Images-Learning\data\processed"

with open(filepath+'\Xagg.pkl','wb') as f: pickle.dump(Xagg, f)

with open(filepath + '\yagg.pkl','wb') as f: pickle.dump(yagg, f)

yaggnumeric = [np.where(yagg[ind]==1)[0][0] for ind in range(yagg.shape[0]) ]
np.unique(yaggnumeric, return_counts=True)

## valid generator

Xaggval = np.empty(shape=[0,150,150,3])
yaggval = np.empty(shape=[0,4])
batches = 0

for X, y in valid_generator:
    Xaggval = np.concatenate((Xaggval,X), axis=0)
    yaggval = np.concatenate((yaggval,y), axis=0)
    batches = batches+1

    if batches > len(valid_generator):
        break

# rotation image
batches = 0

for X, y in valid_generator_augment:
    # print(X.shape, y.shape)
    Xaggval = np.concatenate((Xaggval,X), axis=0)
    yaggval = np.concatenate((yaggval,y), axis=0)
    batches = batches+1

    if batches > 6:
        break

with open(filepath+'\Xaggval.pkl','wb') as f: pickle.dump(Xaggval, f)
with open(filepath+'\yaggval.pkl','wb') as f: pickle.dump(yaggval, f)

yaggnumeric = [np.where(yaggval[ind]==1)[0][0] for ind in range(yaggval.shape[0]) ]
np.unique(yaggnumeric, return_counts=True)

##
cv2.imshow('img', xval[320,:])
cv2.waitKey(0)
cv2.destroyAllWindows()

## standalone rotation generator

Xagg = np.empty(shape=[0,150,150,3])
yagg = np.empty(shape=[0,4])
batches = 0

for X, y in train_generator_augment:
    # print(X.shape, y.shape)
    Xagg = np.concatenate((Xagg,X), axis=0)
    yagg = np.concatenate((yagg,y), axis=0)
    batches = batches+1

    if batches > len(train_generator_augment)-1:
        break

filepath = r"C:\Users\saimunikoti\Manifestation\Images-Learning\data\processed"

with open(filepath+'\Xagg.pkl','wb') as f: pickle.dump(Xagg, f)

with open(filepath + '\yagg.pkl','wb') as f: pickle.dump(yagg, f)

## plot

indi =  np.random.randint(0,560,1)[0]
cv2.imshow('img', Xaggval[indi,:])
cv2.waitKey(0)
cv2.destroyAllWindows()

##


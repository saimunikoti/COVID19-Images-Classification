from numpy.random import seed
seed(8) #1

import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

import pickle
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#
# import os
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imblearn.keras import balanced_batch_generator
from imblearn.over_sampling import RandomOverSampler
# from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from tensorflow.keras import optimizers
from src.models.BalancedDataGenerator import BalancedDataGenerator
from os import listdir
import cv2

##
data_list = listdir( r"C:\Users\saimunikoti\Manifestation\Images-Learning\data\processed\covid-19\four_classes\train")
DATASET_PATH  = r"C:\Users\saimunikoti\Manifestation\Images-Learning\data\processed\covid-19\four_classes\train"
test_dir =  r"C:\Users\saimunikoti\Manifestation\Images-Learning\data\processed\covid-19\four_classes\test"
IMAGE_SIZE    = (150, 150)
NUM_CLASSES   = len(data_list)
BATCH_SIZE    = 10  # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    = 40
LEARNING_RATE = 0.0001

## Train datagen here is a preprocessor
train_datagen = ImageDataGenerator(rescale=1./255,
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

test_datagen = ImageDataGenerator(rescale=1. / 255)

# For multiclass use categorical n for binary use binary
train_generator = train_datagen.flow_from_directory(DATASET_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset = "training",
                                                  seed=42,
                                                  class_mode="categorical"   #For multiclass use categorical n for binary use binary
                                                  )

valid_generator = train_datagen.flow_from_directory(DATASET_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset="validation",
                                                  seed=42,
                                                  class_mode="categorical"
                                                  # For multiclass use categorical n for binary use binary
                                                  )

eval_generator = test_datagen.flow_from_directory(test_dir,target_size=IMAGE_SIZE,batch_size=1,
                                                  shuffle=False, seed=42, class_mode="categorical")

## aggregate data for loading

filepath = r"C:\Users\saimunikoti\Manifestation\Images-Learning\data\processed"

with open(filepath+'\Xagg.pkl','rb') as f:
    Xagg = pickle.load(f)

with open(filepath+'\yagg.pkl','rb') as f:
    yagg = pickle.load(f)

with open(filepath + '\Xaggval.pkl','rb') as f:
    Xaggval = pickle.load(f)

with open(filepath + '\yaggval.pkl','rb') as f:
    yaggval = pickle.load(f)

## smote
# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=42)
#
# x = Xagg.reshape(Xagg.shape[0], -1)
# xval = Xaggval.reshape(Xaggval.shape[0], -1)
#
# X_smote, y_smote = sm.fit_resample(x, yagg)
# X_smoteval, y_smoteval = sm.fit_resample(xval, yaggval)

# x = X_smote.reshape(X_smote.shape[0], 150, 150, 3)
# xval = X_smoteval.reshape(X_smoteval.shape[0], 150, 150, 3)
#
# yaggnumeric = [np.where(y_smote[ind] == 1)[0][0] for ind in range(y_smote.shape[0]) ]
# np.unique(yaggnumeric, return_counts=True)
#
# yaggnumeric = [np.where(y_smoteval[ind] == 1)[0][0] for ind in range(y_smoteval.shape[0]) ]
# np.unique(yaggnumeric, return_counts=True)
#
# indi =  np.random.randint(0, 2412 ,1)[0]
# cv2.imshow('img', x[indi,:])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##
balanced_gen = BalancedDataGenerator(Xagg, yagg, train_datagen, batch_size=BATCH_SIZE)

balanced_gen_val = BalancedDataGenerator(Xaggval, yaggval, train_datagen, batch_size=BATCH_SIZE)

steps_per_epoch = balanced_gen.steps_per_epoch
steps_per_epoch_val = balanced_gen_val.steps_per_epoch

## testing balanced generator
# y_gen = [balanced_gen.__getitem__(0) for i in range(steps_per_epoch)]
#
# cv2.imshow('img', y_gen[10][0][6,:])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# X = X.reshape(X.shape[0],-1)
#
# train_generator, steps_per_epoch = balanced_batch_generator(X, y, sampler=RandomOverSampler(sampling_strategy='minority'), batch_size=BATCH_SIZE, keep_sparse = True, random_state=42)
#
# my_generator = ((np.reshape(X, (-1, 150, 150, 3)), y) for (X,y) in train_generator)
#
# for Xval, yval in valid_generator:
#     # print(X.shape, y.shape)
#     break
#
# Xval = Xval.reshape(Xval.shape[0], -1)
#
# val_generator, steps_per_epoch_val = balanced_batch_generator(Xval, yval, sampler=RandomOverSampler(sampling_strategy='minority'), batch_size=BATCH_SIZE, keep_sparse = True, random_state=42)
#
# my_generator_val = ((np.reshape(X, (-1, 150, 150, 3)), y) for (X,y) in val_generator)

## model build

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
])
model.summary()

model.compile(loss='categorical_crossentropy',  # for multiclass use categorical_crossentropy
              optimizer=optimizers.Adam(lr=LEARNING_RATE),
              metrics=['acc'])

#FIT MODEL
print(len(train_generator))
print(len(valid_generator))

steps_per_epoch=train_generator.n//train_generator.batch_size
steps_per_epoch_val=valid_generator.n//valid_generator.batch_size

# history = model.fit(x ,y_smote, epochs=100, batch_size=10, validation_data=(xval, y_smoteval))
# history = model.fit(Xagg ,yagg, epochs=100, batch_size=10, validation_data=(Xaggval, yaggval))

history = model.fit_generator(balanced_gen,
                        steps_per_epoch =steps_per_epoch,
                        validation_data =balanced_gen_val,
                        validation_steps =steps_per_epoch_val,
                        epochs = 40 )

## save models

##
eval_generator.reset()
x = model.evaluate_generator(eval_generator,
                           steps = np.ceil(len(eval_generator)),
                           use_multiprocessing = False,
                           verbose = 1,
                           workers=1,
                           )

print('Test loss:' ,  x[0])
print('Test accuracy:', x[1])



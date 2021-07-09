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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imblearn.keras import balanced_batch_generator
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from src.visualization import visualize as vs
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from os import listdir
from src.data import config
##
data_list = listdir( config.datapath + "\\train")
DATASET_PATH  = config.datapath + "\\train"
test_dir =  config.datapath + "\\test"

IMAGE_SIZE    = (150, 150)
NUM_CLASSES   = len(data_list)
BATCH_SIZE    = 10  # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    = 40
LEARNING_RATE = 0.0001

##
#Train datagen here is a preprocessor
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
                                   validation_split = 0.15,
                                   fill_mode='constant')

# eval generator
test_datagen = ImageDataGenerator(rescale=1. / 255)

eval_generator_train = test_datagen.flow_from_directory(DATASET_PATH, target_size=IMAGE_SIZE,batch_size=1,
                                                  shuffle=False, seed=42, class_mode="categorical")

eval_generator = test_datagen.flow_from_directory(test_dir,target_size=IMAGE_SIZE,batch_size=1,
                                                  shuffle=False, seed=42, class_mode="categorical")

# For multiclass use categorical n for binary use binary
train_batches = train_datagen.flow_from_directory(DATASET_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset = "training",
                                                  seed=42,
                                                  class_mode="categorical"   #For multiclass use categorical n for binary use binary
                                                  )

valid_batches = train_datagen.flow_from_directory(DATASET_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset="validation",
                                                  seed=42,
                                                  class_mode="categorical"
                                                  # For multiclass use categorical n for binary use binary
                                                  )

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
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
])
model.summary()

##
model.compile(loss='categorical_crossentropy',  # for multiclass use categorical_crossentropy
              optimizer=optimizers.RMSprop(),
              metrics=['acc'])

#FIT MODEL
print(len(train_batches))
print(len(valid_batches))

STEP_SIZE_TRAIN = train_batches.n//train_batches.batch_size
STEP_SIZE_VALID = valid_batches.n//valid_batches.batch_size

filepath = config.modelpath + "\\2-class-Covid19-CNN_rmsprop_test.h5"

mcp = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')

history = model.fit_generator(train_batches,
                        steps_per_epoch =STEP_SIZE_TRAIN,
                        validation_data = valid_batches,
                        validation_steps = STEP_SIZE_VALID,
                        callbacks=[mcp],
                        epochs = 80 )

# plot training performance
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# def plot_acc_loss(result, epochs):
#     acc = result.history['acc']
#     loss = result.history['loss']
#     val_acc = result.history['val_acc']
#     val_loss = result.history['val_loss']
#     plt.figure(figsize=(15, 5))
#     plt.subplot(121)
#     plt.plot(range(1, epochs), acc[1:], label='Train_acc')
#     plt.plot(range(1, epochs), val_acc[1:], label='Val_acc')
#     plt.title('Accuracy over ' + str(epochs) + ' Epochs', size=15)
#     plt.legend()
#     plt.grid(True)
#     plt.subplot(122)
#     plt.plot(range(1, epochs), loss[1:], label='Train_loss')
#     plt.plot(range(1, epochs), val_loss[1:], label='Val_loss')
#     plt.title('Loss over ' + str(epochs) + ' Epochs', size=15)
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
# plot_acc_loss(result, 100)

## Evalute the trained model on evaluate generator

filepath = config.modelpath + "\\3-class-Covid19-CNN_rmsprop.h5"
model = load_model(filepath)

eval_generator.reset()
x = model.evaluate_generator(eval_generator,
                           steps = np.ceil(len(eval_generator)),
                           use_multiprocessing = False,
                           verbose = 1,
                           workers=1,
                           )
print('Test loss:' ,  x[0])
print('Test accuracy:', x[1])

eval_generator.reset()
eval_generator_train.reset()
names=["covid","normal","pneumonia_bac","pneumonia_vir"]

y_pred = model.predict(eval_generator, steps = np.ceil(len(eval_generator)))
y_pred_all = model.predict(eval_generator_train, steps = np.ceil(len(eval_generator_train)))

predict = np.argmax(y_pred, axis=-1)

classes = eval_generator.classes[eval_generator.index_array]
classes_all = eval_generator_train.classes[eval_generator_train.index_array]

classes = np.reshape(classes, (len(classes), 1))

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
y_test = enc.fit_transform(classes).toarray()

##
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

Resultsdic = {}

Resultsdic['accuracy'] = accuracy_score(classes, predict )
Resultsdic['precision'] = precision_score(classes, predict, average="weighted")
Resultsdic['recall'] = recall_score(classes, predict, average="weighted")
Resultsdic['auc score'] = roc_auc_score(y_test, y_pred, average="weighted", multi_class='ovr')

##
names=["covid","normal","pneumonia-bac", "pneumonia-vir"]
names=["Healthy","infected", "Pneumonia"]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(classes, predict)
vs.plot_confusion_matrix(cm, target_names=names, normalize=False)



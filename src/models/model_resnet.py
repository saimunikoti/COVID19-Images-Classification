from numpy.random import seed
seed(8) #1

import tensorflow
tensorflow.random.set_seed(7)

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
from tensorflow.keras.models import Model ,load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.applications.vgg16 import decode_predictions
# from keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np


from tensorflow.python.keras import models
from tensorflow.python.keras import layers

from tensorflow.keras import optimizers

from os import listdir
##
data_list = listdir( r"C:\Users\sddahale\Desktop\Shweta\CourseWork\CIS830\covid-19\four_classes\train")
DATASET_PATH  = r"C:\Users\sddahale\Desktop\Shweta\CourseWork\CIS830\covid-19\four_classes\train"
test_dir =  r"C:\Users\sddahale\Desktop\Shweta\CourseWork\CIS830\covid-19\four_classes\test"
IMAGE_SIZE    = (150, 150)
NUM_CLASSES   = len(data_list)
BATCH_SIZE    = 10  # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    = 10
LEARNING_RATE = 0.0001

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
                                   validation_split = 0.2,
                                   fill_mode='constant')

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
from tensorflow.keras.applications import ResNet50

conv_base = ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

conv_base.trainable = True

model = models.Sequential()
model.add(conv_base)

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',  # for multiclass use categorical_crossentropy
              optimizer=optimizers.Adam(lr=LEARNING_RATE),
              metrics=['acc'])

#FIT MODEL
print(len(train_batches))
print(len(valid_batches))

STEP_SIZE_TRAIN=train_batches.n//train_batches.batch_size
STEP_SIZE_VALID=valid_batches.n//valid_batches.batch_size

result = model.fit_generator(train_batches,
                        steps_per_epoch =STEP_SIZE_TRAIN,
                        validation_data = valid_batches,
                        validation_steps = STEP_SIZE_VALID,
                        epochs= NUM_EPOCHS,
                       )


## test network
import matplotlib.pyplot as plt

def plot_acc_loss(result, epochs):
    acc = result.history['acc']
    loss = result.history['loss']
    val_acc = result.history['val_acc']
    val_loss = result.history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(range(1, epochs), acc[1:], label='Train_acc')
    plt.plot(range(1, epochs), val_acc[1:], label='Val_acc')
    plt.title('Accuracy over ' + str(epochs) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(1, epochs), loss[1:], label='Train_loss')
    plt.plot(range(1, epochs), val_loss[1:], label='Val_loss')
    plt.title('Loss over ' + str(epochs) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_acc_loss(result, 10)

model.save(r"C:\Users\sddahale\Desktop\Shweta\CourseWork\CIS830\covid-19\4-class-Covid19-Mod-Xception.h5")

## load save model
model = tf.keras.models.load_model(r"C:\Users\sddahale\Desktop\Shweta\CourseWork\CIS830\covid-19\4-class-Covid19-Mod-Resnet50.h5")

## # Create evaluate data generator from test set
#Dont forget shuffle false

test_datagen = ImageDataGenerator(rescale=1. / 255)
eval_generator = test_datagen.flow_from_directory(test_dir,target_size=IMAGE_SIZE,batch_size=1,
                                                  shuffle=False, seed=42, class_mode="categorical")

## Evalute the trained model on evaluate generator
eval_generator.reset()
x = model.evaluate_generator(eval_generator,
                           steps = np.ceil(len(eval_generator)),
                           use_multiprocessing = False,
                           verbose = 1,
                           workers=1,
                           )

print('Test loss:' ,  x[0])
print('Test accuracy:', x[1])

## daat generator

eval_generator.reset()

count = [0, 0, 0, 0]

files = eval_generator.filenames

for i in range(len(files)):
    x, y = eval_generator.next()
    img = x
    predict = model.predict(img)

    p = np.argmax(predict, axis=-1)
    print(str(p[0]) + " " + files[eval_generator.batch_index - 1])
    # print(predict)
    # p=model.predict_classes(img)
    count[p[0]] += 1

# print(str(p[0])+" "+files[i])
print(count)
##
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay

filenames = eval_generator.filenames
nb_samples = len(filenames)
eval_generator.reset()
predict = model.predict_generator(eval_generator,steps = np.ceil(len(eval_generator)))
pp=predict
predict=np.argmax(predict, axis=-1)
classes= eval_generator.classes[eval_generator.index_array]
acc=sum(predict==classes)/len(predict)
names=["covid","normal","pneumonia_bac","pneumonia_vir"]
#print(confusion_matrix(classes,predict))

font = {
'family': 'Times New Roman',
'size': 12
}
plt.rc('font', **font)
cm = confusion_matrix(classes, predict)
print(cm)
print(classification_report(classes,predict, labels = names))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel('Predicted labels \nAccuracy: {:0.2f}'.format(acc*100))
plt.ylabel("True labels")
plt.xticks(classes, [])
plt.yticks(classes, [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()

##

# fpr, tpr, _ = roc_curve(classes, predict)
# roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

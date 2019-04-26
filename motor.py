
#Importing all the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import pylab as plt
import time
import os
from datetime import datetime
import numpy as np
from numpy import genfromtxt
import h5py


import sklearn as sk
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model

# method to create recurrence plot
def rec_plot(s, eps=None, steps=None):
    if eps == None:
        eps = 0.1
    if steps == None:
        steps = 10
    N = s.size
    S = np.repeat(s[None, :], N, axis=0)
    Z = np.floor(np.abs(S-S.T)/eps)
    Z[Z > steps] = steps
    return Z

def get_year_to_timestamp(year):
    return (datetime.strptime(str(year)+'-01-01 00:00:00+00:00', '%Y-%m-%d %H:%M:%S%z').timestamp())


#input y_label to ref_y_label
start = time.time()
try:
    print("load")
    X = np.load('X.npy')
    Y = np.load('Y.npy')
except:
    f = open('y_label', 'r')
    ref_y_label = {}
    for line in f.read().splitlines():
        (key, val) = line.split(":")
        ref_y_label[(key)] = val
    #output dataset
    path = "dataset/"
    dataset = []
    batch = 3
    Y = []
    X = []

    #X=np.array(X)
    for file_name in os.listdir(path):
        raw_data = genfromtxt(path+file_name, delimiter=',', dtype="unicode")
        #del label name
        raw_data = np.delete(raw_data, 0, axis=0)
        temp_x = []
        temp_y = []

        error_date = ref_y_label[file_name]
        error_y = datetime.strptime(error_date, '%Y-%m-%d').year
        error_m = datetime.strptime(error_date, '%Y-%m-%d').month
        error_d = datetime.strptime(error_date, '%Y-%m-%d').day
        for idx, raw in enumerate(raw_data):
            temp_y.append(raw[0])
            year = datetime.strptime(raw[0], '%Y-%m-%d %H:%M:%S%z').year
            data_time = datetime.strptime(
                raw[0], '%Y-%m-%d %H:%M:%S%z').timestamp()
            data_ystart_time = get_year_to_timestamp(year)
            #=data_time-data_ystart_time

            raw[0] = data_time - data_ystart_time
            #print(raw_data[idx][0])

            temp_x.append(raw.astype(float))
            if (idx+1) % batch == 0:
                rec = rec_plot(np.asarray(temp_x).reshape(-1))
                X.append(rec)

                #for Y
                if_error = False
                for temp_y_d in temp_y:
                    temp_y = datetime.strptime(
                        temp_y_d, '%Y-%m-%d %H:%M:%S%z').year
                    temp_m = datetime.strptime(
                        temp_y_d, '%Y-%m-%d %H:%M:%S%z').month
                    temp_d = datetime.strptime(
                        temp_y_d, '%Y-%m-%d %H:%M:%S%z').day
                    if temp_y == error_y and temp_m == error_m and temp_d == error_d:
                        if_error = True
                if if_error:
                    Y.append([1, 0])
                else:
                    Y.append([0, 1])

                temp_x = []
                temp_y = []
                #exit(1)
        raw_data = raw_data.astype(float)
        #print(raw_data.astype(float))
    X = np.array(X)
    Y = np.array(Y)
    np.save('X', X)
    np.save('Y', Y)

end = time.time()
print('Elapsed time:')
print(end - start)
X = X.reshape(X.shape[0], 1, 144, 144)
X = X.astype('float32')
X /= np.amax(X)
# split in train and test set
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=13)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Define CNN model
batch_size = 64
epochs = 2
num_classes = 2
motor_model_dl = Sequential()
motor_model_dl.add(Conv2D(32, kernel_size=(
    3, 3), activation='linear', padding='same', input_shape=(1, 144, 144)))
motor_model_dl.add(LeakyReLU(alpha=0.1))
motor_model_dl.add(MaxPooling2D((2, 2), padding='same'))
motor_model_dl.add(Dropout(0.15))


motor_model_dl.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
motor_model_dl.add(LeakyReLU(alpha=0.1))
motor_model_dl.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
motor_model_dl.add(Dropout(0.25))
motor_model_dl.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
motor_model_dl.add(LeakyReLU(alpha=0.1))
motor_model_dl.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
motor_model_dl.add(Dropout(0.4))
motor_model_dl.add(Flatten())
motor_model_dl.add(Dense(128, activation='linear'))
motor_model_dl.add(LeakyReLU(alpha=0.1))
motor_model_dl.add(Dropout(0.3))
motor_model_dl.add(Dense(num_classes, activation='softmax'))

motor_model_dl.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=keras.optimizers.Adam(), metrics=['accuracy']) 

#reshape to include depth

#convert to float32 and normalize to [0,1]


motor_train = motor_model_dl.fit(x_train, y_train, batch_size=batch_size,
                                 epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# save
motor_model_dl.save('my_model.h5')  # HDF5, pip3 install h5py
# delete
del motor_model_dl

# 從 HDF5 檔案中載入模型
model = load_model('my_model.h5')

# 驗證模型
score = model.evaluate(x_test, y_test, verbose=0)

# 輸出結果
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plotting the curves
accuracy = motor_train.history['acc']
val_accuracy = motor_train.history['val_acc']
loss = motor_train.history['loss']
val_loss = motor_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# # test on AFS
# from afs import models
# # Write a file as model file.
# with open('my_model.h5', 'w') as f:
#     f.write('dummy model')

# # User-define evaluation result
# extra_evaluation = {
#     'confusion_matrix_TP': 0.9,
#     'confusion_matrix_FP': 0.8,
#     'confusion_matrix_TN': 0.7,
#     'confusion_matrix_FN': 0.6,
#     'AUC': 1.0
# }

# # User-define Tags
# tags = {'machine': 'machine01'}

# # Model object
# afs_models = models()

# # Upload the model to repository and the repository name is the same as file name.
# # Accuracy and loss is necessary, but extra_evaluation and tags are optional.
# afs_models.upload_model(
#     model_path='my_model.h5', accuracy=0.9, loss=0.13,
#     extra_evaluation=extra_evaluation, tags=tags, model_repository_name='my_model.h5')

# # Get the latest model info
# model_info = afs_models.get_latest_model_info(
#     model_repository_name='my_model.h5')

# # See the model info
# print(model_info)


"""
Created on Fri Nov 10 12:11:49 2017

@author: Narotam
"""
#%%
from keras import backend as K
import os

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend
set_keras_backend('tensorflow');

import random
import scipy.io as sio
import numpy as np
from keras.callbacks import EarlyStopping

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D
from os import listdir
import timeit
from skimage import io


#%%
def loadImages(path):
    # return array of images
 
    imagesList = listdir(path) 
    ar=np.arange(1,len(imagesList)+1);
    i=0;
    while i<len(imagesList): 
        imagesList[i]=str(ar[i])+".jpg"; 
        i=i+1;
    loadedImages = []
    for image in imagesList:
        img = io.imread(path + image)
        loadedImages.append(img)        
    
    return loadedImages


train_imgs_path = "F:\\Project\\Dataset\\salient_train\\"
val_imgs_path = "F:\\Project\\Dataset\\salient_val\\"
salient_train_Y=sio.loadmat('F:\Project\Dataset\\salient_train.mat');
salient_val_Y=sio.loadmat('F:\Project\Dataset\\salient_val.mat')

imgs_train = loadImages(train_imgs_path)
imgs_val = loadImages(val_imgs_path)
imgs_val=np.array(imgs_val, dtype='float32')
imgs_val=imgs_val/255
imgs_train=np.array(imgs_train, dtype='float32')
imgs_train=imgs_train/255
salient_train_Y=salient_train_Y['sal_val'];
salient_val_Y=salient_val_Y['sal_vals'];

#%%
prob_drop_hidden = 0.2
prob_drop_conv=0.2;
img_rows, img_cols = 64,64  
input_shape = (img_rows, img_cols,3)
# Convolutional model
model = Sequential()

# conv1 layer
model.add(Conv2D(96, (5,5), padding='valid', activation='relu', input_shape=input_shape,use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
model.add(Dropout(prob_drop_conv))

# conv2 layer
model.add(Conv2D(256, (5,5), padding='valid', activation='relu',use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
model.add(Dropout(prob_drop_conv))

# conv3 layer
model.add(Conv2D(384, (3, 3), padding='valid', activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
model.add(Conv2D(384, (3, 3), padding='valid', activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2,2), padding='valid'))
model.add(Flatten())

# fc1 layer
model.add(Dense(4096, activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dropout(prob_drop_hidden))

# fc2 layer
model.add(Dense(1024, activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
opt = optimizers.SGD(lr=0.01)
model.compile(optimizer=opt, loss='mean_squared_error')
model.summary()
earlyStopping=EarlyStopping(monitor='val_loss',patience=50, verbose=0, mode='auto')
model.fit(imgs_train,salient_train_Y, epochs=500, batch_size=1000, shuffle=True,callbacks=[earlyStopping],validation_data=(imgs_val,salient_val_Y));
model.save('F:\\my_model_global.h5');
























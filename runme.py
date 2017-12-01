
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


import random
import scipy.io as sio
import numpy as np
import h5py
from keras.callbacks import EarlyStopping

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from os import listdir
import timeit
from skimage import io
from keras.callbacks import ModelCheckpoint

def generate_batches_from_hdf5_file(filepath, batchsize):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated.
    :return: (ndarray, ndarray) (xs, ys): Yields a tuple which contains a full batch of images and labels.
    """
    dimensions = (batchsize, 51, 51, 3) # 28x28 pixel, one channel
 
    while 1:
        f = h5py.File(filepath, "r")
        filesize = len(f['sal_train'])
        
        s = list(range(148670))
        random.shuffle(s);
        n_entries = 0
        while n_entries < (filesize - batchsize):
            b=s[n_entries : n_entries + batchsize];
            b.sort();
            xs = f['imgs_train'][b,:,:,:];
            xs = np.reshape(xs, dimensions).astype('float32')

            y_values = f['sal_train'][b]
            n_entries += batchsize
            yield (xs, y_values)
        f.close()


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

#path = "F:\\Project\\Dataset\\all_patches\\"
#
#imgs = loadImages(path)
#path = "F:\\Project\\Dataset\\salient_val\\"
#
#imgs_val = loadImages(path)
#imgs_val=np.array(imgs_val, dtype='float32')
#imgs_val=imgs_val/255
#imgs=np.array(imgs, dtype='float32')
#imgs=imgs/255


#%%
prob_drop_hidden = 0.4
prob_drop_conv=0.3;
img_rows, img_cols = 51,51  
input_shape = (img_rows, img_cols,3)
# Convolutional model
model = Sequential()

# conv1 layer
model.add(Conv2D(96, (11, 11), padding='valid', activation='relu', input_shape=input_shape,use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
model.add(Dropout(prob_drop_conv))

# conv2 layer
model.add(Conv2D(256, (5,5), padding='valid', activation='relu',use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Dropout(prob_drop_conv))

# conv3 layer
model.add(Conv2D(384, (3, 3), padding='valid', activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(3,3), padding='valid'))
model.add(Flatten())
#model.add(Dropout(prob_drop_conv))

# fc1 layer
model.add(Dense(2048, activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dropout(prob_drop_hidden))

# fc2 layer
model.add(Dense(512, activation='relu', use_bias=True, kernel_initializer='random_uniform', bias_initializer='zeros'))
model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt, loss='mean_squared_error')
model.summary()
#sal_vals=sio.loadmat('F:\Project\Dataset\\sal_vals.mat');
#sal_vals=sal_vals['sal_vals'];
#sal_vals1=sio.loadmat('F:\Project\Dataset\\sal_vals1.mat')
#sal_vals1=sal_vals1['sal_vals1'];
#sal_train=np.append(sal_vals,sal_vals1,axis=0);
#sal_vals=sio.loadmat('F:\Project\Dataset\\validation.mat');
#sal_vals=sal_vals['sal_vals'];
# Train

#earlyStopping=EarlyStopping(monitor='loss',patience=20, verbose=0, mode='auto')
#history = model.fit(imgs,sal_train, epochs=200, batch_size=1000, shuffle=True,callbacks=[earlyStopping],validation_data=(imgs_val,sal_vals));

i=0;
summ=0;
s = list(range(148670))
nepochs=1;

f = h5py.File('F:\\train.hdf5', 'r')
g = h5py.File('F:\\val_set.h5', 'r')
X=f['imgs_train'][:,:,:,:];
Y=f['sal_train'][:];
val_x=g['imgs_val'][:,:,:,:];
val_y=g['sal_val'][:];
earlyStopping=EarlyStopping(monitor='loss',patience=20, verbose=0, mode='auto')
model.fit(X,Y, epochs=500, batch_size=1000, shuffle=True,callbacks=[earlyStopping],validation_data=(val_x,val_y));
model.save('F:\\my_model1.h5');























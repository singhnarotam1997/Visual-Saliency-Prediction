# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:51:22 2017

@author: Narotam
"""
from keras import backend as K
import os

def set_keras_backend(backend):

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

set_keras_backend('tensorflow')
import random
import scipy.misc
import scipy.io as sio
import numpy as np
import h5py
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from os import listdir
from keras.models import load_model
from skimage import io
from keras.callbacks import ModelCheckpoint

model=load_model(r'/home/clab2/Desktop/cnn/models/my_model1.h5');
images_path1='F:\\Project\\Dataset\\images\\val2\\'
imagesList = listdir(images_path1) 
for image in imagesList :
    a=np.zeros((480,640), dtype=np.double)
    img = io.imread(images_path1+image);
    img=np.array(img, dtype='float32');
    img=img/255;
    im=np.zeros((1,480,640,3),dtype='float32');
    im[0,:,:,:]=img;
    i=25;
    j=25;
    while i<455:
        j=25;
        while j<615:
            img=im[:,i-25:i+26,j-25:j+26,:];
            a[i,j]=model.predict(img,batch_size=1);
            j=j+1;
        i=i+1;

    scipy.misc.imsave('/home/clab2/Desktop/cnn/predicted1/'+image,a); #output folder for first model's images

model=load_model(r'/home/clab2/Desktop/cnn/models/my_model2.h5');
images_path2='/home/clab2/Desktop/cnn/val2/'
imagesList = listdir(images_path2) 
cx=480;
cy=640;
for image in imagesList :
    a=np.zeros((480,640), dtype=np.double)
    img = io.imread(images_path2+image);
    img=np.array(img, dtype='float32');
    img=np.float32(img);
    img=img/255;
    i=0;
    j=0;
    while i<480:
        j=0;
        while j<640:
            im=np.zeros((480*2+1,640*2+1,3),dtype='float32');
            im[:,:,0]=0.463241614681625;
            im[:,:,1]=0.430633775453174;
            im[:,:,2]=0.389898652047589;
            im[cx-i:cx+(-i+480),cy-j:cy+(-j+640),:]=img;
            im1=np.zeros((1,64,64,3),dtype='float32');
            im1[0,:,:,:]=scipy.misc.imresize(im[:,:,:],(64,64));
            im1=im1/255;
            a[i,j]=model.predict(im1,batch_size=1);
            j=j+1;
        i=i+1;

    scipy.misc.imsave('/home/clab2/Desktop/cnn/predicted2/'+image,a); #output folder for second model's images

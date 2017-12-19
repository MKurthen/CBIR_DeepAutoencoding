import os
import pickle
import sys
import timeit

import h5py
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

sys.path.append('./')
sys.path.append('./deep_learning/')

from deep_learning.classes_and_functions import Linear_SdA
from deep_learning.logistic_sgd import load_data
from deep_learning.utils import tile_raster_images
from deep_learning.dA import dA
from deep_learning.SdA import SdA

try:
    import PIL.Image as Image
except ImportError:
    import Image


pretrain_lr=0.1
pretrain_lr_binary = 1e-2

batch_size=128

with h5py.File('./deep_learning_data/flickr_32x32_2.hdf5') as f:
    X = f['X'][:] 

X_std = np.std(X)
X_mean = np.mean(X)
X = (X-X_mean) / X_std

numpy_rng = np.random.RandomState(89677)

print ('... building the model')
# construct the stacked denoising autoencoder class
sda = Linear_SdA(
    numpy_rng=numpy_rng,
    n_ins=32*32*3,
    hidden_layers_sizes=[8192, 4096, 2048, 1024, 512, 256]
)


# load parameters from previous training 
parameters = pickle.load(open('deep_learning_data/sda_parameters.p', 'rb'))
for value, param in zip(parameters, sda.params):
    param.set_value(value)

X_mean = X_mean / 256 
X_std = X_std / 256

abs_diff = 0
rmse = 0
for i in range(100):
    original = (X[i*250:(i+1)*250] * X_std) + X_mean 
    #original = (X[i*250:(i+1)*250]) 
    reconstructed = (sda.prop_through(X[i*250:(i+1)*250].reshape((250,-1)))[-1] * X_std )+ X_mean 
    #reconstructed = (sda.prop_through(X[i*250:(i+1)*250].reshape((250,-1)))[-1]) 
    difference = original - reconstructed
    abs_diff += np.mean(np.abs(difference)) 
    rmse += np.sqrt(np.mean(difference**2))
print('absolute mean difference: {}'.format( abs_diff / 100))
print('RMSE: {}'.format( rmse / 100))

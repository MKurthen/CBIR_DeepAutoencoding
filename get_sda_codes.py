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


with h5py.File('./deep_learning_data/flickr_32x32.hdf5') as f:
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

print(sda.sigmoid_layers)

# load parameters from previous training 
parameters = pickle.load(open('deep_learning_data/sda_parameters.p', 'rb'))
for value, param in zip(parameters, sda.params):
    param.set_value(value)

code = T.round(sda.sigmoid_layers[-1].output)
get_code = theano.function([sda.x], [code])

with h5py.File('deep_learning_data/codes_sda_full.hdf5', 'a') as f:
    #ds = f.create_dataset('X', shape = (25000, 256), dtype = 'int')
    # process in 100 batches  250 images
    ds = f['X']
    for i in range(100):
        if i % 10 == 0:
            print('processing batch {}/100'.format(i))

        codes = get_code(X[i*250:(i+1)*250].reshape((-1,3072)) )[0]
        ds[i*250:(i+1)*250] = codes

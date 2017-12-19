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


pretrain_lr_linear = 5e-2
pretrain_lr_binary = 5e-3

batch_size=128

with h5py.File('deep_learning_data/flickr_32x32_varpatches.hdf5', 'r') as f:
    X = f['X'][:] 

X_std = np.std(X)
X_mean = np.mean(X)
X = (X-X_mean) / X_std

train_set_x = X 
valid_set_x = X[1925000:1975000]
test_set_x = X[1975000:]

train_set_x = theano.shared(np.asarray(train_set_x,
                                               dtype=theano.config.floatX),
                                 borrow=True)

valid_set_x = theano.shared(np.asarray(valid_set_x,
                                               dtype=theano.config.floatX),
                                 borrow=True)

test_set_x = theano.shared(np.asarray(test_set_x,
                                               dtype=theano.config.floatX),
                                 borrow=True)

datasets = (train_set_x, valid_set_x, test_set_x)

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

numpy_rng = np.random.RandomState(89677)
print ('... building the model')

# construct the stacked denoising autoencoder class
sda = Linear_SdA(
    numpy_rng=numpy_rng,
    n_ins=112*3,
    hidden_layers_sizes=[1024, 512, 256, 128, 64, 28]
)

pretraining_epochs = 20

print ('... getting the pretraining functions')
pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                            batch_size=batch_size)
print ('... pre-training the model')
start_time = timeit.default_timer()
## Pre-train layer-wise
corruption_levels = [.1, .1, .1, .1, .1, .1, .1, .1, .1]
for i in range(0,sda.n_layers):
    lr = pretrain_lr_linear if i == 0 else pretrain_lr_binary
    # go through pretraining epochs
    for epoch in range(pretraining_epochs):
        # go through the training set
        c = []
        for batch_index in range(n_train_batches):
            c.append(pretraining_fns[i](index=batch_index,
                     corruption=corruption_levels[i],
                     lr=lr ))
        print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c)))

# save the parameters
parameters = [param.get_value() for param in sda.params]
pickle.dump(parameters, open('deep_learning_data/sda_parameters_varpatch.p', 'wb'))


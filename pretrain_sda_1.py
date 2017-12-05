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

with h5py.File('../deep_learning_files/flickr_32x32.hdf5') as f:
    X = f['X'][:] 

X_std = np.std(X)
X_mean = np.mean(X)
X = (X-X_mean) / X_std

train_set_x = X[:24000] 
valid_set_x = X[24000:24500]
test_set_x = X[24500:25000]

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
    n_ins=32*32*3,
    hidden_layers_sizes=[8192, 4096, 2048, 1024, 512, 256]
)

pretraining_epochs = 500
pretrain_lr=0.1

print ('... getting the pretraining functions')
pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                            batch_size=batch_size)
print ('... pre-training the linear-binary layer')
start_time = timeit.default_timer()
## Pre-train layer-wise
corruption_levels = [.3, .2, .1, .05, .025, .01]
for i in range(0,1):
    # go through pretraining epochs
    for epoch in range(pretraining_epochs):
        # go through the training set
        c = []
        for batch_index in range(n_train_batches):
            c.append(pretraining_fns[i](index=batch_index,
                     corruption=corruption_levels[i],
                     lr=pretrain_lr))
        print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, np.mean(c) ))
        if epoch % 10 == 0:
            with open('sda_training_log.txt', 'a') as f:
                f.write('epoch {}: cost: {}'.format(epoch, np.mean(c)))

# save the parameters
parameters = [param.get_value() for param in sda.params]
pickle.dump(parameters, open('sda_parameters.p', 'wb'))


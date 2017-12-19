import collections
import os
import pickle
import sys

import bitstring
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


batch_size=128

with h5py.File('deep_learning_data/flickr_32x32_varpatches.hdf5', 'r') as f:
    X = f['X'][:] 

X_std = np.std(X)
X_mean = np.mean(X)
X = (X-X_mean) / X_std

train_set_x = X 
valid_set_x = X[1925000:1975000]
test_set_x = X[1975000:]
numpy_rng = np.random.RandomState(89677)

sda = Linear_SdA(
    numpy_rng=numpy_rng,
    n_ins=112*3,
    hidden_layers_sizes=[1024, 512, 256, 128, 64, 28]
)

# load parameters from previous training 
parameters = pickle.load(open('deep_learning_data/sda_parameters_varpatch.p', 'rb'))
for value, param in zip(parameters, sda.params):
    param.set_value(value)

get_code = theano.function(
            [sda.x],
            [sda.sigmoid_layers[-1].output])

hashing_table = collections.defaultdict(list)
#with h5py.File('./deep_learning_data/varpatch_codes_sda.hdf5') as f:
#    ds = f.create_dataset('X', shape=(25000, 81), dtype='uint')

for i in range(81):
    if i % 9 == 0:
        print(i)
    codes = np.round(get_code(X[i*25000:(i+1)*25000].reshape(-1,336))).reshape(-1,28)

    with h5py.File('./deep_learning_data/varpatch_codes_sda.hdf5') as f:
        #ds = f.create_dataset('X', shape=(25000, 81), dtype='uint')
        ds = f['X']
        for j in range(25000):
            ds[j,i] = bitstring.Bits(codes[j]).uint

    for j in range(25000):
        code = codes[j]
        code_uint = bitstring.Bits(code).uint
        # the way defaultdict behaves, if code_uint already exists as a key, the image index j is appended to the correspoding list,
        #   if not, a new key:value pair (code_uint:[j]) is created
        hashing_table[code_uint].append(j)

    #with h5py.File('./deep_learning_data/varpatch_hashing_table_sda.hdf5') as f:
    #    dtype = h5py.special_dtype(vlen=np.dtype('int32'))
    #    #ds = f.create_dataset('X', (2**28,), dtype=dtype)
    #    ds = f['X']
    #    for j in range(25000):
    #        code = codes[j]
    #        if i == 0
    #        ds[bitstring.Bits(code).uint] = np.append(ds[bitstring.Bits(code).uint], j)
pickle.dump(hashing_table, open('./deep_learning_data/sda_varpatch_hashing_table.p', 'wb'))

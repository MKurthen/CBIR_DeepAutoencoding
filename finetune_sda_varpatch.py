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

# load parameters from previous training 
parameters = pickle.load(open('deep_learning_data/sda_parameters_varpatch.p', 'rb'))
for value, param in zip(parameters, sda.params):
    param.set_value(value)

# FINETUNING THE MODEL #
finetune_lr = 5e-3
training_epochs = 1000 
patience = 1000 * n_train_batches  # look as this many examples regardless
patience_increase = 2.  # wait this much longer when a new best is
                        # found
improvement_threshold = 0.995  # a relative improvement of this much is
                               # considered significant


# get the training, validation and testing function for the model
print ('... getting the finetuning functions')
train_fn, validate_model, test_model = sda.build_finetune_functions(
    datasets=datasets,
    batch_size=batch_size,
    learning_rate=finetune_lr
)

print ('... finetuning the model')
# early-stopping parameters

validation_frequency = min(n_train_batches, patience / 2)
    # go through this many minibatche before checking the network
    # on the validation set; in this case we check every epoch

best_validation_loss = np.inf
test_score = 0.
start_time = timeit.default_timer()

done_looping = False
epoch = 0

while (epoch < training_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):
        minibatch_avg_cost = train_fn(minibatch_index)
        iter = (epoch - 1) * n_train_batches + minibatch_index

        if (iter + 1) % validation_frequency == 0:
            validation_losses = validate_model()
            this_validation_loss = np.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation RMSE of%f' %
                  (epoch, minibatch_index + 1, n_train_batches,
                   this_validation_loss ))

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if (
                    this_validation_loss < best_validation_loss *
                    improvement_threshold
                ):
                    patience = max(patience, iter * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the test set
                test_losses = test_model()
                test_score = np.mean(test_losses)
                print(('epoch %i, minibatch %i/%i, test RMSE of '
                       'best model %f') %
                      (epoch, minibatch_index + 1, n_train_batches,
                       test_score ))

        if patience <= iter:
            done_looping = True
            break
    if epoch % 20 == 0:
        # save the parameters everyy 100 epochs
        parameters = [param.get_value() for param in sda.params]
        pickle.dump(parameters, open('deep_learning_data/sda_parameters_intermediate_varpatch.p', 'wb'))


# save the parameters
parameters = [param.get_value() for param in sda.params]
pickle.dump(parameters, open('deep_learning_data/sda_parameters_varpatch.p', 'wb'))

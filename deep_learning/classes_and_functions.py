import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from deep_learning.logistic_sgd import load_data
from deep_learning.utils import tile_raster_images
from deep_learning.dA import dA
from deep_learning.SdA import SdA

try:
    import PIL.Image as Image
except ImportError:
    import Image
	
class HiddenLayer(object):
    def __init__(self, rng, input,  n_in, n_out, W=None, b=None, v=None,decode_input = None,
                 activation=T.tanh, decode_own_output = False, decode_activation = T.tanh):

        self.input = input
        self.activation = activation
        self.decode_activation = decode_activation
        
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            
        if v is None:
            v_values = numpy.zeros((n_in,), dtype=theano.config.floatX)
            v = theano.shared(value=v_values, name='v', borrow=True)

        self.W = W
        self.b = b
        self.v = v

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        
        #output backwards for decoding
        if decode_own_output:
            self.decode_input = self.output
        else:
            self.decode_input = decode_input
            
        if self.decode_input is not None:
            lin_decode_output = T.dot(self.decode_input, self.W.T) + self.v
            self.decode_output = (
                lin_decode_output if activation is None
                else activation(lin_decode_output)
        )
        # parameters of the model
        self.params = [self.W, self.b, self.v]
    
    def set_decode_input(self, decode_input, ):
        self.decode_input = decode_input
        lin_decode_output = T.dot(self.decode_input, self.W.T) + self.v
        self.decode_output = (
            lin_decode_output if self.decode_activation is None
            else self.decode_activation(lin_decode_output)
        )


class Linear_dA(dA):

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """

        if theano.tensor.gt(corruption_level, 0 ):
            return input + self.theano_rng.normal(size = input.shape, avg = 0,
                std = corruption_level, dtype = theano.config.floatX)
        else:
            return input
    



    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.dot(hidden, self.W_prime) + self.b_prime

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        RMSE =  T.sqrt(T.mean((self.x - z)**2, axis=1) )
       
        cost = T.mean(RMSE)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)
    
    def get_reconstruction_function(self):
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        return theano.function([self.x], [z])
    
class Linear_SdA(SdA):
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1]
    ):
       

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] label

        # start-snippet-2
        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x

                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid,
                                        decode_activation=None)
            else:
                layer_input = self.sigmoid_layers[-1].output

                sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                            input=layer_input,
                                            #decode_input = layer_decode_input,
                                            n_in=input_size,
                                            n_out=hidden_layers_sizes[i],
                                            activation=T.nnet.sigmoid,
                                            decode_activation=T.nnet.sigmoid)
                
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            if i == 0:
                dA_layer = Linear_dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b,
                          bvis = sigmoid_layer.v)
            else:
                dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b,
                          bvis = sigmoid_layer.v)
                
            self.dA_layers.append(dA_layer)
        # end-snippet-2

        #construct the decoding chain in top down order   
        for i in range(self.n_layers):
            if i == 0:
                self.sigmoid_layers[self.n_layers -1].set_decode_input(
                    self.sigmoid_layers[self.n_layers -1].output)
            else:
                self.sigmoid_layers[self.n_layers -1 - i].set_decode_input(
                    self.sigmoid_layers[self.n_layers - i].decode_output)

        #RMSE
        self.finetune_cost = T.sqrt(T.mean((self.sigmoid_layers[0].decode_output - self.x)**2))
        
        #define prop-through function
            
        output = self.sigmoid_layers[0].decode_output
        self.prop_through = theano.function([self.x], output) 
        
        

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        train_set_x = datasets[0]
        valid_set_x = valid_set_y = datasets[1]
        test_set_x  = test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.finetune_cost,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
                
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.finetune_cost,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score
    
    def build_reconstruction_functions_no_round(self):
        reconstruction_functions = []
        for i in range(self.n_layers):
            code = self.sigmoid_layers[i].output
            reconstruction_function = theano.function(
                [self.x],
                [self.sigmoid_layers[0].decode_output],
                givens = {self.sigmoid_layers[i].decode_input: code} )
            reconstruction_functions.append(reconstruction_function)
        return reconstruction_functions
            
    def build_reconstruction_functions(self):
        reconstruction_functions = []
        for i in range(self.n_layers):
            code = self.sigmoid_layers[i].output
            code = T.round(code)
            reconstruction_function = theano.function(
                [self.x],
                [self.sigmoid_layers[0].decode_output],
                givens = {self.sigmoid_layers[i].decode_input: code} )
            reconstruction_functions.append(reconstruction_function)
        return reconstruction_functions
            
                        
                    
                    

def print_and_log(string, logfile = '/home/ubuntu/ipe/ipe/logfile_sda1'):
    print(string)
    with open(logfile, 'a') as f:
        f.write(string + '\n')
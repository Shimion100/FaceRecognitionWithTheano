from __future__ import print_function

import numpy
import copy

# theano
import theano
import theano.tensor as T

# other classes
from ConvPoolLayer import LeNetConvPoolLayer
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression
from LoadData import LoadData


class CpuModel(object):

    def __init__(self):
        print("The default Init----------------------")
        # other params
        self.learning_rate = 0.01
        self.data_set = None
        self.n_kernels = None
        self.batch_size = 1
        self.test_set_x = None
        self.test_set_y = None
        self.predict_model = None
        self.predict_result = None

        # Layers
        self.layer0 = LeNetConvPoolLayer()
        self.layer1 = LeNetConvPoolLayer()
        self.layer2 = LeNetConvPoolLayer()
        self.layer3 = LeNetConvPoolLayer()
        self.layer4 = HiddenLayer()
        self.layer5 = LogisticRegression()

    def switch_data_set(self, data_set=''):
        # Read the data
        load = LoadData()
        datasets = load.load_data(data_set)
        self.test_set_x, self.test_set_y = datasets[2]

    def copy_params_init(self, params, learning_rate, data_set='', nkerns=[64, 128, 192, 192], batch_size=1):

        # Init the params
        self.learning_rate = learning_rate
        self.data_set = data_set
        self.n_kernels = nkerns
        self.batch_size = batch_size

        # Read the data
        load = LoadData()
        datasets = load.load_data(data_set)
        self.test_set_x, self.test_set_y = datasets[2]

        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y')

        # Building model
        print('... building the model')
        layer0_input = x.reshape((batch_size, 1, 126, 126))

        self.layer0.copy_model(
            input=layer0_input,
            image_shape=(batch_size, 1, 126, 126),
            filter_shape=(nkerns[0], 1, 7, 7),
            poolsize=(2, 2),
            w=theano.shared(numpy.asarray(copy.deepcopy(params['array_0']), dtype=theano.config.floatX), borrow=True),
            b=theano.shared(numpy.asarray(copy.deepcopy(params['array_4']), dtype=theano.config.floatX), borrow=True)
        )

        self.layer1.copy_model(
            input=self.layer0.output,
            image_shape=(batch_size, nkerns[0], 60, 60),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2),
            w=theano.shared(numpy.asarray(copy.deepcopy(params['array_1']), dtype=theano.config.floatX), borrow=True),
            b=theano.shared(numpy.asarray(copy.deepcopy(params['array_5']), dtype=theano.config.floatX), borrow=True)
        )

        self.layer2.copy_model(
            input=self.layer1.output,
            image_shape=(batch_size, nkerns[1], 28, 28),
            filter_shape=(nkerns[2], nkerns[1], 5, 5),
            poolsize=(2, 2),
            w=theano.shared(numpy.asarray(copy.deepcopy(params['array_2']), dtype=theano.config.floatX), borrow=True),
            b=theano.shared(numpy.asarray(copy.deepcopy(params['array_6']), dtype=theano.config.floatX), borrow=True)
        )

        self.layer3.copy_model(
            input=self.layer2.output,
            image_shape=(batch_size, nkerns[2], 12, 12),
            filter_shape=(nkerns[3], nkerns[2], 5, 5),
            poolsize=(2, 2),
            w=theano.shared(numpy.asarray(copy.deepcopy(params['array_3']), dtype=theano.config.floatX), borrow=True),
            b=theano.shared(numpy.asarray(copy.deepcopy(params['array_7']), dtype=theano.config.floatX), borrow=True)
        )

        layer4_input = self.layer3.output.flatten(2)

        self.layer4.copy_init(
            input=layer4_input,
            n_in=nkerns[3] * 4 * 4,
            n_out=1024,
            W=theano.shared(numpy.asarray(copy.deepcopy(params['W']), dtype=theano.config.floatX), name='W',
                            borrow=True),
            b=theano.shared(numpy.asarray(copy.deepcopy(params['b']), dtype=theano.config.floatX), name='b',
                            borrow=True),
            activation=T.tanh
        )

        self.layer5 = LogisticRegression()
        self.layer5.copy_init(
            input=self.layer4.output,
            n_in=1024,
            n_out=17,
            W=theano.shared(numpy.asarray(copy.deepcopy(params['W_2']), dtype=theano.config.floatX), borrow=True),
            b=theano.shared(numpy.asarray(copy.deepcopy(params['b_2']), dtype=theano.config.floatX), borrow=True)
        )

        self.predict_model = theano.function(
            [index],
            self.layer5.y_pred,
            givens={
                x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
            }
        )
        a_index = 0
        self.predict_result = self.predict_model(a_index)



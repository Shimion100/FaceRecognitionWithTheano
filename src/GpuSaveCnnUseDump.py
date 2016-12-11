from __future__ import print_function

# primary
import os
import sys
import timeit
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


class CnnModel(object):

    def __init__(self):
        print("The default Init----------------------")
        # other params
        self.learning_rate = 0.01
        self.data_set = None
        self.n_kernels = None
        self.batch_size = 1
        self.test_set_x = None
        self.test_set_y = None
        self.train_set_x = None
        self.train_set_y = None
        self.valid_set_x = None
        self.valid_set_y = None
        self.predict_model = None
        self.predict_result = None
        self.train_model = None
        self.test_model = None
        self.validate_model = None

        # Layers
        self.layer0 = LeNetConvPoolLayer()
        self.layer1 = LeNetConvPoolLayer()
        self.layer2 = LeNetConvPoolLayer()
        self.layer3 = LeNetConvPoolLayer()
        self.layer4 = HiddenLayer()
        self.layer5 = LogisticRegression()

    def copy_init(self, from_model, learning_rate=0.1, data_set='', n_kerns=[64, 128, 192, 192], batch_size=1):
        print("copy from a model------------------")
        # Init the params
        self.learning_rate = learning_rate
        print("learning_rate----------------------%", learning_rate)
        self.data_set = data_set
        self.n_kernels = n_kerns
        self.batch_size = batch_size
        load = LoadData()
        data_sets = load.load_data(data_set)
        self.test_set_x, self.test_set_y = data_sets[2]

        # Read the data
        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y')

        # Building model
        print('... building the model')

        layer0_input = x.reshape((batch_size, 1, 126, 126))

        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)

        self.layer0 = LeNetConvPoolLayer()

        self.layer0.copy_model(
                         input=layer0_input,
                         image_shape=(batch_size, 1, 126, 126),
                         filter_shape=(n_kerns[0], 1, 7, 7),
                         poolsize=(2, 2),
                         w=copy.deepcopy(from_model.layer0.W),
                         b=copy.deepcopy(from_model.layer0.b)

        )
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        self.layer1 = LeNetConvPoolLayer()
        self.layer1.copy_model(
            input=self.layer0.output,
            image_shape=(batch_size, n_kerns[0], 60, 60),
            filter_shape=(n_kerns[1], n_kerns[0], 5, 5),
            poolsize=(2, 2),
            w=copy.deepcopy(from_model.layer1.W),
            b=copy.deepcopy(from_model.layer1.b)
        )

        self.layer2 = LeNetConvPoolLayer()
        self.layer2.copy_model(
            input=self.layer1.output,
            image_shape=(batch_size, n_kerns[1], 28, 28),
            filter_shape=(n_kerns[2], n_kerns[1], 5, 5),
            poolsize=(2, 2),
            w=copy.deepcopy(from_model.layer2.W),
            b=copy.deepcopy(from_model.layer2.b)
        )

        self.layer3 = LeNetConvPoolLayer()
        self.layer3.copy_model(
            input=self.layer2.output,
            image_shape=(batch_size, n_kerns[2], 12, 12),
            filter_shape=(n_kerns[3], n_kerns[2], 5, 5),
            poolsize=(2, 2),
            w=copy.deepcopy(from_model.layer3.W),
            b=copy.deepcopy(from_model.layer3.b)
        )

        layer4_input = self.layer3.output.flatten(2)
        self.layer4 = HiddenLayer()

        self.layer4.copyInit(
            input=layer4_input,
            n_in=n_kerns[3] * 4 * 4,
            n_out=1024,
            W=copy.deepcopy(from_model.layer4.W),
            b=copy.deepcopy(from_model.layer4.b),
            activation=T.tanh
        )

        self.layer5 = LogisticRegression()
        self.layer5.copyInit(input=self.layer4.output, n_in=1024, n_out=17,
                             W=copy.deepcopy(from_model.layer5.W),
                             b=copy.deepcopy(from_model.layer5.b))

        self.predict_model = theano.function(
            [index],
            self.layer5.y_pred,
            givens={
                x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
            }
        )
        a_index = 0
        self.predict_result = self.predict_model(a_index)
        # End of building model

    def original_init(self, learning_rate, data_set='', nkerns=[64, 128, 192, 192], batch_size=1):

        # Init the params
        self.learning_rate = learning_rate
        print("learning_rate----------------------", learning_rate)
        self.data_set = data_set
        self.n_kernels = nkerns
        self.batch_size = batch_size

        # Read the data
        load = LoadData()
        datasets = load.load_data(data_set)
        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]

        rng = numpy.random.RandomState(23455)
        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y')

        # Building model
        print('... building the model')
        layer0_input = x.reshape((batch_size, 1, 126, 126))

        self.layer0.cov_pool_init(
            rng,
            input=layer0_input,
            image_shape=(batch_size, 1, 126, 126),
            filter_shape=(nkerns[0], 1, 7, 7),
            poolsize=(2, 2)
        )

        self.layer1.cov_pool_init(
            rng,
            input=self.layer0.output,
            image_shape=(batch_size, nkerns[0], 60, 60),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )

        self.layer2.cov_pool_init(
            rng,
            input=self.layer1.output,
            image_shape=(batch_size, nkerns[1], 28, 28),
            filter_shape=(nkerns[2], nkerns[1], 5, 5),
            poolsize=(2, 2)
        )

        self.layer3.cov_pool_init(
            rng,
            input=self.layer2.output,
            image_shape=(batch_size, nkerns[2], 12, 12),
            filter_shape=(nkerns[3], nkerns[2], 5, 5),
            poolsize=(2, 2)
        )

        layer4_input = self.layer3.output.flatten(2)

        self.layer4 = HiddenLayer()
        self.layer4.original_init(
            rng,
            input=layer4_input,
            n_in=nkerns[3] * 4 * 4,
            n_out=1024,
            activation=T.tanh
        )

        self.layer5 = LogisticRegression()
        self.layer5.original_init(input=self.layer4.output, n_in=1024, n_out=17)

        cost = self.layer5.negative_log_likelihood(y)
        params = self.layer5.params + self.layer4.params + self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params
        grads = T.grad(cost, params)

        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
            ]

        self.train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: self.train_set_x[index * batch_size: (index + 1) * batch_size],
                y: self.train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.test_model = theano.function(
            [index],
            self.layer5.errors(y),
            givens={
                x: self.test_set_x[index * batch_size: (index + 1) * batch_size],
                y: self.test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.predict_model = theano.function(
            [index],
            self.layer5.y_pred,
            givens={
                x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
            }
        )

        self.validate_model = theano.function(
            [index],
            self.layer5.errors(y),
            givens={
                x: self.valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: self.valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )


        # End of building model

    def train(self, n_epochs):
        print("n_epochs-----------------------%", n_epochs)
        # Prepare the dataset
        n_train_batches = self.train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = self.test_set_x.get_value(borrow=True).shape[0]
        n_train_batches //= self.batch_size
        n_valid_batches //= self.batch_size
        n_test_batches //= self.batch_size

        print("---self.batch_size--------n_train_batches-------", self.batch_size, n_train_batches)

        ###############
        # TRAIN MODEL #
        ###############
        print('... training')
        # early-stopping parameters
        patience = 5000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
        # found
        improvement_threshold = 0.995  # a relative improvement of this much is
        # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
        # go through this many
        # minibatche before checking the network
        # on the validation set; in this case we
        # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                cost_ij = self.train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * \
                                improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            self.test_model(i)
                            for i in range(n_test_batches)
                            ]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    def save_parameters(self):
        ws = []
        bs = []
        ws.append(self.layer0.W)
        ws.append(self.layer1.W)
        ws.append(self.layer2.W)
        ws.append(self.layer3.W)
        ws.append(self.layer4.W)
        ws.append(self.layer5.W)

        bs.append(self.layer0.b)
        bs.append(self.layer1.b)
        bs.append(self.layer2.b)
        bs.append(self.layer3.b)
        bs.append(self.layer4.b)
        bs.append(self.layer5.b)
        w = ws
        b = bs
        params = [w, b]
        from theano.misc.pkl_utils import dump
        with open('best_model.pkl', 'wb') as f:
            dump(params, f)

"""
    Main method
"""

if __name__ == '__main__':
    model = CnnModel()
    model.original_init(0.002, 'file-126.pkl.gz', [64, 128, 192, 192], 435)
    model.train(2000)
    model.save_parameters()

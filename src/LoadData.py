import numpy
import theano
import os
import six.moves.cPickle as pickle
import theano.tensor as T
import gzip


class LoadData(object):
    def __init__(self):
        # type: () -> object
        print("LoadData init...")

    def load_data(self, data_set):

        data_dir, data_file = os.path.split(data_set)
        if data_dir == "" and not os.path.isfile(data_set):
            # Check if dataset is in the data directory.
            new_path = os.path.join(
                os.path.split(__file__)[0],
                "..",
                "data",
                data_set
            )
            if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
                data_set = new_path

        print('... loading data')

        # Load the dataset
        with gzip.open(data_set, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)

        def shared_dataset(data_xy, borrow=True):

            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            return shared_x, T.cast(shared_y, 'int32')

        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval
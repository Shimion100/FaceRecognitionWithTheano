from __future__ import print_function
import numpy
# theano
import theano
import theano.tensor as T

# other classes
from CpuModel import CpuModel
from BinAndCrop import BinAndCropClass
from createDataSet import initDataSet


class PredictOnCPU(object):
    def __init__(self):
        self.model = CpuModel()

    def load_params(self):
        params = numpy.load('best_model.pkl')
        return params

    def load_model(self):
        params = self.load_params()
        self.model.copy_params_init(params, 0.002, 'face.pkl.gz', [64, 128, 192, 192], 1)

    def switch_data_set(self, my_data_set=''):
        bin_crop = BinAndCropClass()
        bin_crop.bin()
        initDataSet()

        # This is needed.
        self.model.switch_data_set('face.pkl.gz')

    def predict(self):
        a_index = T.lscalar()
        predict_the_model = theano.function(
            inputs=[a_index],
            outputs=theano.shared(self.model.predict_result),
            on_unused_input='ignore'
        )

        print("Start----------------------------------")
        predicted_values = predict_the_model(20)
        print(predicted_values)
        print("End----------------------------------")
        name = ''
        if predicted_values == 0:
            name = 'cjr'
        elif predicted_values == 1:
            name = 'dbw'
        elif predicted_values == 2:
            name = 'gwd'
        elif predicted_values == 3:
            name = 'hzz'
        elif predicted_values == 4:
            name = 'jxd'
        elif predicted_values == 5:
            name = 'wy'
        elif predicted_values == 6:
            name = 'll'
        elif predicted_values == 7:
            name = 'qhx'
        elif predicted_values == 8:
            name = 'wwl'
        elif predicted_values == 9:
            name = 'xcl'
        elif predicted_values == 10:
            name = 'zsy'
        elif predicted_values == 11:
            name = 'zx'
        else:
            name = ''

        print(name)

        if predicted_values > 0:
            return True
        else:
            return False

"""
    Main method
"""
if __name__ == '__main__':

    # First you should have a object of PredictOnCPU, then invoke the load_model method
    predictModel = PredictOnCPU()
    predictModel.load_model()

    # In each predict, you should invoke load_data_set and predict
    predictModel.switch_data_set()
    predictModel.predict()

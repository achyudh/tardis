from keras.layers import Average
from keras.models import Model

class Ensemble:
    def __init__(self, models):
        self.models = models

    # TODO: add other methods
    def avg_voting(self, input):
        outputs = [model.outputs[0] for model in self.models]
        target = Average()(outputs)
        model = Model(input, target, name='avg_ensemble')
        return model

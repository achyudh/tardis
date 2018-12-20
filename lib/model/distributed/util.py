from keras.engine.topology import Layer


class EncoderSlice(Layer):
    def __init__(self, input_split_index, **kwargs):
        self.input_split_index = input_split_index
        super(EncoderSlice, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EncoderSlice, self).build(input_shape)

    def call(self, x):
        return x[:, :self.input_split_index]

    def compute_output_shape(self, input_shape):
        return input_shape
        # return input_shape[0], self.input_split_index, input_shape[2]

    def get_config(self):
        config = {
            'input_split_index': self.input_split_index
        }
        base_config = super(EncoderSlice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DecoderSlice(Layer):
    def __init__(self, input_split_index, **kwargs):
        self.input_split_index = input_split_index
        super(DecoderSlice, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DecoderSlice, self).build(input_shape)

    def call(self, x):
        return x[:, self.input_split_index:]

    def compute_output_shape(self, input_shape):
        return input_shape
        # return input_shape[0], input_shape[1] - self.input_split_index, input_shape[2]

    def get_config(self):
        config = {
            'input_split_index': self.input_split_index
        }
        base_config = super(DecoderSlice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
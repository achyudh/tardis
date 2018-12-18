import numpy as np
from keras.utils import Sequence


class WMTSequence(Sequence):

    def __init__(self, encoder_input_data, decoder_input_data, decoder_target_data, config):
        self.batch_size = config.batch_size
        self.target_vocab_size = config.target_vocab_size
        self.x_encoder = encoder_input_data
        self.x_decoder = decoder_input_data
        self.y = decoder_target_data

    def __len__(self):
        return int(np.ceil(len(self.x_encoder) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x_encoder = self.x_encoder[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x_decoder = self.x_decoder[idx * self.batch_size:(idx + 1) * self.batch_size]
        raw_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = np.zeros((raw_y.shape[0], raw_y.shape[1], self.target_vocab_size), dtype=np.int64)

        for i in range(raw_y.shape[0]):
            for j in range(raw_y.shape[1]):
                batch_y[i, j, int(raw_y[i, j])] = 1

        return [batch_x_encoder, batch_x_decoder], batch_y

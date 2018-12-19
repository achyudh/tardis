import codecs
import os
import time

import dill
import keras
import numpy as np
from keras.callbacks import LearningRateScheduler
from tqdm import tqdm


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=dict()):
        self.times = []

    def on_epoch_begin(self, epoch, logs=dict()):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=dict()):
        self.times.append(time.time() - self.epoch_time_start)


def lr_scheduler(initial_lr, decay_factor):
    def schedule(epoch):
        if epoch and epoch < 5:
            return initial_lr
        else:
            # Decay after first 5 epochs
            # TODO: Add step size
            return initial_lr * (decay_factor ** epoch)

    return LearningRateScheduler(schedule, verbose=1)


def load_weights(model, path):
    model.load_weights(path)


def embedding_matrix(path, vocab, embed_dim=300):
    print("Loading embedding matrix from:", os.path.basename(path))

    if os.path.isfile(path + '.pkl'):
        with open(path + '.pkl', 'rb') as pkl_file:
            embed_index = dill.load(pkl_file)
    else:
        embed_index = dict()
        file_sizes = {'wiki.en.vec': 2519428, 'wiki.de.vec': 2275261, 'wiki.fr.vec': 1152450, 'wiki.vi.vec': 292169}

        with codecs.open(path, encoding='utf-8') as embedding_file:
            for line in tqdm(embedding_file, total=file_sizes[os.path.basename(path)]):
                values = line.rstrip().rsplit(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embed_index[word] = coefs

        with open(path + '.pkl', 'wb') as pkl_file:
            dill.dump(embed_index, pkl_file)

    embed_matrix = np.zeros((len(vocab), embed_dim))
    for word, i in vocab.items():
        embedding_vector = embed_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            embed_matrix[i] = embedding_vector
    return embed_matrix

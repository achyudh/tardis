import os
import random

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.initializers import RandomUniform
from keras.layers import Input, LSTM, GRU, Embedding, Dense, Average
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from lib.model.metrics import bleu_score
from lib.model.util import lr_scheduler, TimeHistory


class Seq2Seq:
    def __init__(self, config):
        self.config = config
        recurrent_unit = self.config.recurrent_unit.lower()

        if self.config.cpu:
            devices = list('/cpu:' + str(x) for x in (0, 0))
        else:
            devices = list('/gpu:' + x for x in config.devices)

        with tf.device(devices[0]):
            initial_weights = RandomUniform(minval=-0.08, maxval=0.08, seed=config.seed)
            encoder_inputs = Input(shape=(None,))
            encoder_embedding = Embedding(config.source_vocab_size, config.embedding_dim,
                                          weights=[config.source_embedding_map],
                                          trainable=False)
            encoder_embedded = encoder_embedding(encoder_inputs)

            if recurrent_unit == 'lstm':
                encoder = LSTM(self.config.hidden_dim, return_state=True, return_sequences=True,
                               recurrent_initializer=initial_weights)(encoder_embedded)
                for i in range(1, self.config.num_encoder_layers):
                    encoder = LSTM(self.config.hidden_dim, return_state=True, return_sequences=True)(encoder)
                _, state_h, state_c = encoder
                encoder_states = [state_h, state_c]
            else:
                encoder = GRU(self.config.hidden_dim, return_state=True, return_sequences=True,
                              recurrent_initializer=initial_weights)(encoder_embedded)
                for i in range(1, self.config.num_encoder_layers):
                    encoder = GRU(self.config.hidden_dim, return_state=True, return_sequences=True)(encoder)
                _, state_h = encoder
                encoder_states = [state_h]

        with tf.device(devices[1]):
            decoder_inputs = Input(shape=(None,))
            decoder_embedding = Embedding(config.target_vocab_size, config.embedding_dim,
                                          weights=[config.target_embedding_map],
                                          trainable=False)
            decoder_embedded = decoder_embedding(decoder_inputs)

            if recurrent_unit.lower() == 'lstm':
                decoder = LSTM(self.config.hidden_dim, return_state=True, return_sequences=True)(decoder_embedded, initial_state=encoder_states)
                for i in range(1, self.config.num_decoder_layers):
                    decoder = LSTM(self.config.hidden_dim, return_state=True, return_sequences=True)(decoder)
                decoder_outputs, decoder_states = decoder[0], decoder[1:]
            else:
                decoder = GRU(self.config.hidden_dim, return_state=True, return_sequences=True)(decoder_embedded, initial_state=encoder_states)
                for i in range(1, self.config.num_decoder_layers):
                    decoder = GRU(self.config.hidden_dim, return_state=True, return_sequences=True)(decoder)
                decoder_outputs, decoder_states = decoder[0], decoder[1]

            decoder_dense = Dense(config.target_vocab_size, activation='softmax')
            decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        optimizer = Adam(lr=config.lr, clipnorm=25.)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
        print(self.model.summary())

    def train(self, encoder_train_input, decoder_train_input, decoder_train_target):
        checkpoint_filename = \
            '%s_el%d_dl%d_ds%d_sv%d_tv%d_ep{epoch:02d}.hdf5' % (self.config.recurrent_unit.lower(),
                                                                self.config.num_encoder_layers,
                                                                self.config.num_decoder_layers,
                                                                self.config.dataset_size,
                                                                self.config.source_vocab_size,
                                                                self.config.target_vocab_size)
        time_history_callback = TimeHistory()
        callbacks = [time_history_callback,
                     lr_scheduler(initial_lr=self.config.lr, decay_factor=self.config.decay),
                     ModelCheckpoint(os.path.join('data', 'checkpoints', self.config.dataset, checkpoint_filename),
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=False,
                                     save_weights_only=True,
                                     mode='auto',
                                     period=1)]
        self.model.fit([encoder_train_input, decoder_train_input], decoder_train_target,
                       batch_size=self.config.batch_size,
                       epochs=self.config.epochs,
                       validation_split=0.20,
                       callbacks=callbacks)
        print("Training time (in seconds):", time_history_callback.times)

    def train_generator(self, training_generator, validation_generator):
        checkpoint_filename = \
            '%s_el%d_dl%d_ds%d_sv%d_tv%d_ep{epoch:02d}.hdf5' % (self.config.recurrent_unit.lower(),
                                                                self.config.num_encoder_layers,
                                                                self.config.num_decoder_layers,
                                                                self.config.dataset_size,
                                                                self.config.source_vocab_size,
                                                                self.config.target_vocab_size)
        time_history_callback = TimeHistory()
        callbacks = [time_history_callback,
                     lr_scheduler(initial_lr=self.config.lr, decay_factor=self.config.decay),
                     ModelCheckpoint(os.path.join('data', 'checkpoints', self.config.dataset, checkpoint_filename),
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=False,
                                     save_weights_only=True,
                                     mode='auto',
                                     period=1)]
        self.model.fit_generator(training_generator,
                                 epochs=self.config.epochs,
                                 callbacks=callbacks,
                                 validation_data=validation_generator)
        print("Training time (in seconds):", time_history_callback.times)

    def predict(self, encoder_predict_input):
        beam_size = self.config.beam_size
        max_target_len = encoder_predict_input.shape[0]
        k_beam = [(0, [0] * max_target_len)]

        for i in range(max_target_len):
            all_hypotheses = []
            for prob, hyp in k_beam:
                predicted = self.model.predict([encoder_predict_input, np.array(hyp)])
                new_hypotheses = predicted[i, 0, :].argsort(axis=-1)[-beam_size:]
                for next_hyp in new_hypotheses:
                    all_hypotheses.append((
                            sum(np.log(predicted[j, 0, hyp[j + 1]]) for j in range(i)) + np.log(predicted[i, 0, next_hyp]),
                            list(hyp[:(i + 1)]) + [next_hyp] + ([0] * (encoder_predict_input.shape[0] - i - 1))
                        ))
            k_beam = sorted(all_hypotheses, key=lambda x: x[0])[-beam_size:]  # Sort by probability
        return k_beam[-1][1]  # Pick hypothesis with highest probability

    def evaluate(self, encoder_predict_input, decoder_predict_target):
        y_pred = np.apply_along_axis(self.predict, 1, encoder_predict_input)
        print("BLEU Score:", bleu_score(decoder_predict_target, y_pred))
        # An error in the sacrebleu library prevents multi_bleu_score from working on WMT '14 EN-DE test split
        # print("Multi-BLEU Score", multi_bleu_score(y_pred, self.config.target_vocab, self.config.dataset))


class EnsembleSeq2Seq:
    def __init__(self, config):
        self.config = config

        encoder_inputs = Input(shape=(None,))
        decoder_inputs = Input(shape=(None,))

        models = []
        for i in range(config.num_models):
            config.seed = random.randint(1, 1000)
            ind_model = Seq2Seq(config).model
            models.append(ind_model)

        ind_outputs = [ind_model([encoder_inputs, decoder_inputs]) for ind_model in models]
        model_output = Average()(ind_outputs)

        self.model = Model([encoder_inputs, decoder_inputs], model_output)
        optimizer = Adam(lr=config.lr, clipnorm=25.)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
        print(self.model.summary())

    def train_generator(self, training_generator, validation_generator):
        checkpoint_filename = \
            'ensemble_%s_el%d_dl%d_ds%d_sv%d_tv%d_ep{epoch:02d}.hdf5' % (self.config.recurrent_unit.lower(),
                                                                self.config.num_encoder_layers,
                                                                self.config.num_decoder_layers,
                                                                self.config.dataset_size,
                                                                self.config.source_vocab_size,
                                                                self.config.target_vocab_size)
        time_history_callback = TimeHistory()
        callbacks = [time_history_callback,
                     lr_scheduler(initial_lr=self.config.lr, decay_factor=self.config.decay),
                     ModelCheckpoint(os.path.join('data', 'checkpoints', self.config.dataset, checkpoint_filename),
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=False,
                                     save_weights_only=True,
                                     mode='auto',
                                     period=1)]
        self.model.fit_generator(training_generator,
                                 epochs=self.config.epochs,
                                 callbacks=callbacks,
                                 validation_data=validation_generator)
        print("Training time (in seconds):", time_history_callback.times)

    def predict(self, encoder_predict_input):
        beam_size = self.config.beam_size
        max_target_len = encoder_predict_input.shape[0]
        k_beam = [(0, [0] * max_target_len)]

        for i in range(max_target_len):
            all_hypotheses = []
            for prob, hyp in k_beam:
                predicted = self.model.predict([encoder_predict_input, np.array(hyp)])
                new_hypotheses = predicted[i, 0, :].argsort(axis=-1)[-beam_size:]
                for next_hyp in new_hypotheses:
                    all_hypotheses.append((
                            sum(np.log(predicted[j, 0, hyp[j + 1]]) for j in range(i)) + np.log(predicted[i, 0, next_hyp]),
                            list(hyp[:(i + 1)]) + [next_hyp] + ([0] * (encoder_predict_input.shape[0] - i - 1))
                        ))
            k_beam = sorted(all_hypotheses, key=lambda x: x[0])[-beam_size:]  # Sort by probability
        return k_beam[-1][1]  # Pick hypothesis with highest probability

    def evaluate(self, encoder_predict_input, decoder_predict_target):
        y_pred = np.apply_along_axis(self.predict, 1, encoder_predict_input)
        print("BLEU Score:", bleu_score(decoder_predict_target, y_pred))
        # An error in the sacrebleu library prevents multi_bleu_score from working on WMT '14 EN-DE test split
        # print("Multi-BLEU Score", multi_bleu_score(y_pred, self.config.target_vocab, self.config.dataset))

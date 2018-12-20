import random

import numpy as np
from keras.initializers import RandomUniform
from keras.layers import Input, LSTM, GRU, Embedding, Dense, Average
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.generic_utils import get_custom_objects

from lib.model.distributed.util import EncoderSlice, DecoderSlice
from lib.model.metrics import bleu_score


class Seq2Seq:
    def __init__(self, config):
        self.config = config
        recurrent_unit = self.config.recurrent_unit.lower()
        get_custom_objects().update({'EncoderSlice': EncoderSlice, 'DecoderSlice': DecoderSlice})

        initial_weights = RandomUniform(minval=-0.08, maxval=0.08, seed=config.seed)
        stacked_input = Input(shape=(None,))
        
        # encoder_input = Lambda(lambda x: x[:, config.input_split_index:])(stacked_input)
        encoder_input = EncoderSlice(config.input_split_index)(stacked_input)
        encoder_embedding = Embedding(config.source_vocab_size, config.embedding_dim,
                                      weights=[config.source_embedding_map],
                                      trainable=False)
        encoder_embedded = encoder_embedding(encoder_input)

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

        # decoder_input = Lambda(lambda x: x[:, config.input_split_index:])(stacked_input)
        decoder_input = DecoderSlice(config.input_split_index)(stacked_input)
        decoder_embedding = Embedding(config.target_vocab_size, config.embedding_dim,
                                      weights=[config.target_embedding_map],
                                      trainable=False)
        decoder_embedded = decoder_embedding(decoder_input)

        if recurrent_unit.lower() == 'lstm':
            decoder = LSTM(self.config.hidden_dim, return_state=True, return_sequences=True)(decoder_embedded, initial_state=encoder_states)
            for i in range(1, self.config.num_decoder_layers):
                decoder = LSTM(self.config.hidden_dim, return_state=True, return_sequences=True)(decoder)
            decoder_output, decoder_state = decoder[0], decoder[1:]
        else:
            decoder = GRU(self.config.hidden_dim, return_state=True, return_sequences=True)(decoder_embedded, initial_state=encoder_states)
            for i in range(1, self.config.num_decoder_layers):
                decoder = GRU(self.config.hidden_dim, return_state=True, return_sequences=True)(decoder)
            decoder_output, decoder_state = decoder[0], decoder[1]

        decoder_dense = Dense(config.target_vocab_size, activation='softmax')
        decoder_output = decoder_dense(decoder_output)

        self.model = Model(stacked_input, decoder_output)
        optimizer = Adam(lr=config.lr, clipnorm=25.)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
        print(self.model.summary())

    def predict(self, encoder_predict_input):
        beam_size = self.config.beam_size
        max_target_len = encoder_predict_input.shape[0]
        k_beam = [(0, [0] * max_target_len)]

        for i in range(max_target_len):
            all_hypotheses = []
            for prob, hyp in k_beam:
                train_input = np.hstack((encoder_predict_input, np.array(hyp)))
                train_input = np.expand_dims(train_input, axis=0)

                predicted = self.model.predict(train_input)
                predicted = np.squeeze(predicted, axis=0)

                new_hypotheses = predicted[i, :].argsort(axis=-1)[-beam_size:]
                for next_hyp in new_hypotheses:
                    all_hypotheses.append((
                            sum(np.log(predicted[j, hyp[j + 1]]) for j in range(i)) + np.log(predicted[i, next_hyp]),
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

        models = []
        model_inputs = Input(shape=(None,))

        for i in range(config.num_models):
            config.seed = random.randint(1, 1000)
            ind_model = Seq2Seq(config).model
            models.append(ind_model)

        ind_outputs = [ind_model(model_inputs) for ind_model in models]
        model_output = Average()(ind_outputs)

        self.model = Model(model_inputs, model_output)
        optimizer = Adam(lr=config.lr, clipnorm=25.)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
        print(self.model.summary())

    def predict(self, encoder_predict_input):
        beam_size = self.config.beam_size
        max_target_len = encoder_predict_input.shape[0]
        k_beam = [(0, [0] * max_target_len)]

        for i in range(max_target_len):
            all_hypotheses = []
            for prob, hyp in k_beam:
                train_input = np.hstack((encoder_predict_input, np.array(hyp)))
                train_input = np.expand_dims(train_input, axis=0)

                predicted = self.model.predict(train_input)
                predicted = np.squeeze(predicted, axis=0)

                new_hypotheses = predicted[i, :].argsort(axis=-1)[-beam_size:]
                for next_hyp in new_hypotheses:
                    all_hypotheses.append((
                            sum(np.log(predicted[j, hyp[j + 1]]) for j in range(i)) + np.log(predicted[i, next_hyp]),
                            list(hyp[:(i + 1)]) + [next_hyp] + ([0] * (encoder_predict_input.shape[0] - i - 1))
                        ))
            k_beam = sorted(all_hypotheses, key=lambda x: x[0])[-beam_size:]  # Sort by probability
        return k_beam[-1][1]  # Pick hypothesis with highest probability

    def evaluate(self, encoder_predict_input, decoder_predict_target):
        y_pred = np.apply_along_axis(self.predict, 1, encoder_predict_input)
        print("BLEU Score:", bleu_score(decoder_predict_target, y_pred))
        # An error in the sacrebleu library prevents multi_bleu_score from working on WMT '14 EN-DE test split
        # print("Multi-BLEU Score", multi_bleu_score(y_pred, self.config.target_vocab, self.config.dataset))

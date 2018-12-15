from keras.layers import Input, LSTM, Embedding, Dense, Activation, Bidirectional, Concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.initializers import RandomUniform
from keras.callbacks import LearningRateScheduler

from lib.model.metrics import bleu_score, multi_bleu_score
from lib.model.util import lr_scheduler

class Seq2Seq:
    def __init__(self, config):
        self.config = config

        # Encoder
        initial_weights = RandomUniform(minval=-0.08, maxval=0.08, seed=config.seed)
        encoder_inputs = Input(shape=(None, ))
        encoder_embedding = Embedding(config.source_vocab_size, config.embedding_dim,
                                      weights=[config.source_embedding_map], trainable=False)
        encoder_embedded = encoder_embedding(encoder_inputs)
        encoder = Bidirectional(LSTM(config.hidden_dim, return_state=True, return_sequences=True, recurrent_initializer=initial_weights), merge_mode='concat')(encoder_embedded)
        for i in range(1, config.num_layers):
            encoder = Bidirectional(LSTM(config.hidden_dim, return_state=True, return_sequences=True), merge_mode='concat')(encoder)
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder
        encoder_states = [Concatenate()([forward_h, backward_h]), Concatenate()([forward_c, backward_c])]

        # Decoder
        decoder_inputs = Input(shape=(None, ))
        decoder_embedding = Embedding(config.target_vocab_size, config.embedding_dim,
                                      weights=[config.target_embedding_map], trainable=False)
        decoder_embedded = decoder_embedding(decoder_inputs)
        decoder = LSTM(config.hidden_dim * 2, return_state=True, return_sequences=True)(decoder_embedded, initial_state=encoder_states)  # Accepts concatenated encoder states as input
        for i in range(1, config.num_layers):
            decoder = LSTM(config.hidden_dim * 2, return_state=True, return_sequences=True)(decoder) # Use the final encoder state as context
        decoder_outputs, _, _ = decoder
        decoder_dense = Dense(config.target_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Input: Source and target sentence, Output: Predicted translation
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        optimizer = SGD(lr=config.lr, momentum=0.0, clipnorm=25.)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
        print(self.model.summary())

    def train(self, encoder_train_input, decoder_train_input, decoder_train_target):
        callbacks = [lr_scheduler(initial_lr=self.config.lr, decay_factor=self.config.decay)]
        self.model.fit([encoder_train_input, decoder_train_input], decoder_train_target,
                       batch_size=self.config.batch_size,
                       epochs=self.config.epochs,
                       validation_split=0.20,
                       callbacks=callbacks)

    def train_generator(self, training_generator, validation_generator):
        callbacks = [lr_scheduler(initial_lr=self.config.lr, decay_factor=self.config.decay)]
        self.model.fit_generator(training_generator, epochs=self.config.epochs, callbacks=callbacks,
                                 validation_data=validation_generator)

    def predict(self, encoder_predict_input, decoder_predict_input):
        return self.model.predict([encoder_predict_input, decoder_predict_input])

    def evaluate(self, encoder_predict_input, decoder_predict_input, decoder_train_target):
        y_pred = self.model.predict([encoder_predict_input, decoder_predict_input])
        print("BLEU Score:", bleu_score(y_pred, decoder_train_target, self.config.target_vocab))
        # An error in the sacrebleu library prevents multi_bleu_score from working on WMT '14 EN-DE test split
        # print("BLEU Score", multi_bleu_score(y_pred, self.config.target_vocab, self.config.dataset))


class TinySeq2Seq:
    def __init__(self, config):
        pass

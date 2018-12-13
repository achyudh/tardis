from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.initializers import RandomUniform
from keras.callbacks import LearningRateScheduler

from lib.model.metrics import bleu_score, multi_bleu_score


class Seq2Seq:
    def __init__(self, config):
        self.config = config

        # Encoder
        initial_weights = RandomUniform(minval=-0.08, maxval=0.08, seed=config.seed)
        encoder_inputs = Input(shape=(None, ))
        encoder_embedding = Embedding(config.source_vocab_size, config.embedding_dim)
        encoder_embedded = encoder_embedding(encoder_inputs)
        encoder = LSTM(config.hidden_dim, return_state=True, return_sequences=True, recurrent_initializer=initial_weights)(encoder_embedded)
        for i in range(1, config.num_layers):
            encoder = LSTM(config.hidden_dim, return_state=True, return_sequences=True)(encoder)
        _, state_h, state_c = encoder
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(None, ))
        decoder_embedding = Embedding(config.target_vocab_size, config.embedding_dim)
        decoder_embedded = decoder_embedding(decoder_inputs)
        decoder = LSTM(config.hidden_dim, return_state=True, return_sequences=True)(decoder_embedded, initial_state=encoder_states)
        for i in range(1, config.num_layers):
            decoder = LSTM(config.hidden_dim, return_state=True, return_sequences=True)(decoder) # Use the final encoder state as context
        decoder_outputs, _, _ = decoder
        decoder_dense = Dense(config.target_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Input: Source and target sentence, Output: Predicted translation
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        optimizer = SGD(lr=config.lr, momentum=0.0, clipnorm=25.)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
        print(self.model.summary())

    def train(self, encoder_train_input, decoder_train_input, decoder_train_target):
        def lr_scheduler(initial_lr, decay_factor):
            def schedule(epoch):
                if epoch and epoch < 5:
                    return initial_lr
                else: # decay after first 5 epochs
                    return initial_lr * (decay_factor ** epoch)  # TODO: add step size

            return LearningRateScheduler(schedule, verbose=1)

        callbacks = [
            lr_scheduler(initial_lr=self.config.lr, decay_factor=self.config.decay)
            ]

        self.model.fit([encoder_train_input, decoder_train_input], decoder_train_target,
                       batch_size=self.config.batch_size,
                       epochs=self.config.epochs,
                       validation_split=0.20,
                       callbacks=callbacks)

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

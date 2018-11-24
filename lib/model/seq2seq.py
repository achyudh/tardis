from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model


class Seq2Seq:
    def __init__(self, config):
        # TODO: Add multiple layers
        self.config = config

        # Encoder
        encoder_inputs = Input(shape=(None, ))
        encoder_embedding = Embedding(config.source_vocab_size, config.embedding_dim)
        encoder = LSTM(config.hidden_dim, return_state=True)
        encoder_embedded = encoder_embedding(encoder_inputs)
        _, state_h, state_c = encoder(encoder_embedded)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(None, ))
        decoder_embedding = Embedding(config.target_vocab_size, config.embedding_dim)
        decoder = LSTM(config.hidden_dim, return_sequences=True, return_state=True)
        decoder_dense = Dense(config.target_vocab_size, activation='softmax')
        decoder_embedded = decoder_embedding(decoder_inputs)
        decoder_outputs, _, _ = decoder(decoder_embedded, initial_state=encoder_states)  # Use the final encoder state as context
        decoder_outputs = decoder_dense(decoder_outputs)

        # Input: Source and target sentence, Output: Predicted translation
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # TODO: Change optimizer
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        print(self.model.summary())

    def train(self, encoder_train_input, decoder_train_input, decoder_train_target):
        self.model.fit([encoder_train_input, decoder_train_input], decoder_train_target, batch_size=self.config.batch_size,
                       epochs=self.config.epochs, validation_split=0.20)

    def predict(self, encoder_predict_input, decoder_predict_input):
        return self.model.predict([encoder_predict_input, decoder_predict_input])


class TinySeq2Seq:
    def __init__(self, config):
        pass

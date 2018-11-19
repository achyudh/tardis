from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense
from keras.utils import plot_model

class Seq2Seq:
    def __init__(self, encoder_input_data, decoder_input_data, decoder_target_data, source_vocab_size, target_vocab_size, config):
        # TODO: add multiple layers
        embedding_dim = config.embedding_dim
        hidden_dim = config.hidden_dim

        batch_size = config.batch_size
        num_epochs = config.epochs

        # encoder
        encoder_inputs = Input(shape=(None, ))
        encoder_embedding = Embedding(source_vocab_size, embedding_dim)
        encoder = LSTM(hidden_dim, return_state=True)

        encoder_embedded = encoder_embedding(encoder_inputs)
        _, state_h, state_c = encoder(encoder_embedded)
        encoder_states = [state_h, state_c]

        # decoder
        decoder_inputs = Input(shape=(None, ))
        decoder_embedding = Embedding(target_vocab_size, embedding_dim)
        decoder = LSTM(hidden_dim, return_sequences=True, return_state=True)
        decoder_dense = Dense(target_vocab_size, activation='softmax')

        decoder_embedded = decoder_embedding(decoder_inputs)
        decoder_outputs, _, _ = decoder(decoder_embedded, initial_state=encoder_states) # use the final encoder state as context
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs) # input: source and target sentence, output: predicted translation

        # TODO: change optimizer
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
        print(model.summary())

        # train
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.20)

class TinySeq2Seq:
    def __init__(self, config):
        pass

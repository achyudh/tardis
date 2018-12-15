import numpy as np
import nltk

from tqdm import tqdm


def preprocess(source_data, target_data):
    # TODO: Preprocess in one pass
    # Convert to lowercase characters
    source_data = source_data.swifter.apply(lambda x: x.str.lower())
    target_data = target_data.swifter.apply(lambda x: x.str.lower())

    # Add SOS and EOS tokens
    target_data = target_data.swifter.apply(lambda x: 'SOS ' + x + ' EOS')

    # Remove punctuation and digits
    source_data = source_data.swifter.apply(lambda x: x.str.replace('[^a-zA-Z\s]', ''))
    target_data = target_data.swifter.apply(lambda x: x.str.replace('[^a-zA-Z\s]', ''))

    source_data = source_data.values.flatten()
    target_data = target_data.values.flatten()
    return source_data, target_data


def replace_unknown(line, vocab):
    words = list()
    for word in nltk.word_tokenize(line):
        if word in vocab:
            words.append(word)
        else:
            words.append('UNK')
    return ' '.join(words)


def build_indices(source_data, target_data, source_vocab, target_vocab, one_hot):
    max_target_len = max(len(nltk.word_tokenize(sent)) for sent in target_data)
    max_source_len = max(len(nltk.word_tokenize(sent)) for sent in source_data)
    target_vocab_size = len(target_vocab)
    num_instances = len(source_data)

    encoder_input_data = np.zeros((num_instances, max_source_len), dtype=np.float64)
    decoder_input_data = np.zeros((num_instances, max_target_len), dtype=np.float64)
    if one_hot:
        decoder_target_data = np.zeros((num_instances, max_target_len, target_vocab_size), dtype=np.float64)
    else:
        decoder_target_data = np.zeros((num_instances, max_target_len), dtype=np.float64)

    # Convert words to ids
    for i, (source_sent, target_sent) in tqdm(enumerate(zip(source_data, target_data)), total=len(source_data)):
        for j, word in enumerate(nltk.word_tokenize(source_sent)):
            encoder_input_data[i, j] = source_vocab[word]
        for j, word in enumerate(nltk.word_tokenize(target_sent)):
            decoder_input_data[i, j] = target_vocab[word]
            if j > 0:
                if one_hot:
                    decoder_target_data[i, j - 1, target_vocab[word]] = 1
                else:
                    decoder_target_data[i, j - 1] = target_vocab[word]
    return encoder_input_data, decoder_input_data, decoder_target_data


def reverse_indexing(indexed_data, vocab, ravel=False):
    reversed_data = list()
    indexed_data = np.argmax(indexed_data, axis=-1)
    # TODO: Use dict comprehension instead
    word_idx = dict([(id, word) for word, id in vocab.items()])
    for indexed_line in indexed_data:
        if ravel:
            reversed_data.append([' '.join((word_idx[x] for x in indexed_line[1:len(indexed_line)-1]))])
        else:
            reversed_data.append(' '.join((word_idx[x] for x in indexed_line[1:len(indexed_line)-1])))
    return reversed_data

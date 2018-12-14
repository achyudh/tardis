import numpy as np
import nltk


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

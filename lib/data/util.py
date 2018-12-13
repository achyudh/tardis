import numpy as np


def preprocess(source_data, target_data, source_vocab=None, target_vocab=None, replace_unk=False):
    # TODO: Preprocess in one pass
    # Convert to lowercase characters
    source_data = source_data.apply(lambda x: x.str.lower())
    target_data = target_data.apply(lambda x: x.str.lower())

    # Add SOS and EOS tokens
    target_data = target_data.apply(lambda x: 'SOS ' + x + ' EOS')

    # Remove punctuation and digits
    source_data = source_data.apply(lambda x: x.str.replace('[^a-zA-Z\s]', ''))
    target_data = target_data.apply(lambda x: x.str.replace('[^a-zA-Z\s]', ''))

    if replace_unk:
        if source_vocab is not None:
            source_vocab = set(source_vocab)
            source_data = [replace_unknown(x, source_vocab) for x in source_data.values.flatten()]
        if target_vocab is not None:
            target_vocab = set(target_vocab)
            target_data = [replace_unknown(x, target_vocab) for x in target_data.values.flatten()]
    else:
        source_data = source_data.values.flatten()
        target_data = target_data.values.flatten()
    return source_data, target_data


def replace_unknown(line, vocab):
    # TODO: Tokenize using NLTK
    words = list()
    for word in line.split():
        if word in vocab:
            words.append(word)
        else:
            words.append('UNK')
    return ' '.join(words)


def reverse_indexing(indexed_data, vocab, ravel=False):
    reversed_data = list()
    indexed_data = np.argmax(indexed_data, axis=-1)
    # TODO: Use dict comprehension instead
    word_idx = dict([(id, word) for id, word in enumerate(vocab)])
    for indexed_line in indexed_data:
        if ravel:
            reversed_data.append([' '.join((word_idx[x] for x in indexed_line[1:len(indexed_line)-1]))])
        else:
            reversed_data.append(' '.join((word_idx[x] for x in indexed_line[1:len(indexed_line)-1])))
    return reversed_data

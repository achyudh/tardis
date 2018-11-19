import string

import pandas as pd

import numpy as np

def load_data(path, source_lang, target_lang):
    data = pd.read_table(path, names=[source_lang, target_lang])

    print(data.shape)

    print(data.sample(10))

    data = preprocess(data, source_lang, target_lang)

    print(data.sample(10))

    num_instances = len(data[source_lang])

    # create source vocabulary
    all_source_words = set()
    for source_line in data[source_lang]:
        for word in source_line.split():
            if word not in all_source_words:
                all_source_words.add(word)

    source_len_list = [len(sent.split(' ')) for sent in data[source_lang]]
    max_source_len = np.max(source_len_list)

    # create target vocabulary
    all_target_words = set()
    for target_line in data[target_lang]:
        for word in target_line.split():
            if word not in all_target_words:
                all_target_words.add(word)

    target_len_list = [len(sent.split(' ')) for sent in data[target_lang]]
    max_target_len = np.max(target_len_list)

    source_words = sorted(list(all_source_words))
    target_words = sorted(list(all_target_words))
    source_vocab_size = len(source_words)
    target_vocab_size = len(target_words)

    source_word_idx = dict([(word, id) for id, word in enumerate(source_words)])
    target_word_idx = dict([(word, id) for id, word in enumerate(target_words)]) # EOS: 0, SOS: 1

    # TODO: pickle vocab

    encoder_input_data = np.zeros((num_instances, max_source_len), dtype=np.float64)
    decoder_input_data = np.zeros((num_instances, max_target_len), dtype=np.float64)
    decoder_target_data = np.zeros((num_instances, max_target_len, target_vocab_size), dtype=np.float64)

    # convert words to ids
    for i, (source_sent, target_sent) in enumerate(zip(data[source_lang], data[target_lang])):
        for j, word in enumerate(source_sent.split()):
            encoder_input_data[i, j] = source_word_idx[word]
        for j, word in enumerate(target_sent.split()):
            decoder_input_data[i, j] = target_word_idx[word]
            # TODO: shift target by 1

    return encoder_input_data, decoder_input_data, decoder_target_data, source_vocab_size, target_vocab_size

def preprocess(data, source_lang, target_lang):
    # TODO: do all in one pass?

    # convert to lowercase characters
    data[source_lang] = data[source_lang].apply(lambda x : x.lower())
    data[target_lang] = data[target_lang].apply(lambda x : x.lower())

    # add SOS and EOS tokens
    data[target_lang] = data[target_lang].apply(lambda x : 'SOS ' + x + ' EOS ')

    # remove punctuation and digits
    punctuation_list = string.punctuation
    digit_list = string.digits
    exclude_list = punctuation_list + digit_list
    data[source_lang] = data[source_lang].apply(lambda x : ''.join(ch for ch in x if ch not in exclude_list))
    data[target_lang] = data[target_lang].apply(lambda x : ''.join(ch for ch in x if ch not in exclude_list))

    return data

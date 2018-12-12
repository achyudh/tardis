import numpy as np
import pandas as pd
import os

from lib.data.util import preprocess


def en_de(path, reverse=False, splits='train'):
    if reverse:
        source_lang, target_lang = 'de', 'en'
    else:
        source_lang, target_lang = 'en', 'de'

    if splits.lower() == 'train':
        # TODO: Replace one-got encoding with continuous vector representations to prevent memory errors
        source_data = pd.read_table(os.path.join(path, 'en_de', 'train.%s' % source_lang)).head(n=4000)
        target_data = pd.read_table(os.path.join(path, 'en_de', 'train.%s' % target_lang)).head(n=4000)
    elif splits.lower() == 'test':
        source_data = pd.read_table(os.path.join(path, 'en_de', 'test15.%s' % source_lang))
        target_data = pd.read_table(os.path.join(path, 'en_de', 'test15.%s' % target_lang))
    else:
        raise Exception("Unsupported dataset splits")

    print("Dataset size:", source_data.shape, target_data.shape)

    source_data, target_data = preprocess(source_data, target_data)
    num_instances = len(source_data)

    # Create source vocabulary
    all_source_words = set()
    for source_line in source_data:
        for word in source_line.split():
            if word not in all_source_words:
                all_source_words.add(word)

    source_len_list = [len(sent.split(' ')) for sent in source_data]
    max_source_len = np.max(source_len_list)

    # Create target vocabulary
    all_target_words = set()
    for target_line in target_data:
        for word in target_line.split():
            if word not in all_target_words:
                all_target_words.add(word)

    target_len_list = [len(sent.split(' ')) for sent in target_data]
    max_target_len = np.max(target_len_list)

    source_words = sorted(list(all_source_words))
    target_words = sorted(list(all_target_words))
    source_vocab_size = len(source_words)
    target_vocab_size = len(target_words)

    source_word_idx = dict([(word, id) for id, word in enumerate(source_words)])
    target_word_idx = dict([(word, id) for id, word in enumerate(target_words)])  # EOS: 0, SOS: 1

    # TODO: Pickle vocab
    encoder_input_data = np.zeros((num_instances, max_source_len), dtype=np.float64)
    decoder_input_data = np.zeros((num_instances, max_target_len), dtype=np.float64)
    decoder_target_data = np.zeros((num_instances, max_target_len, target_vocab_size), dtype=np.float64)

    # Convert words to ids
    for i, (source_sent, target_sent) in enumerate(zip(source_data, target_data)):
        for j, word in enumerate(source_sent.split()):
            encoder_input_data[i, j] = source_word_idx[word]
        for j, word in enumerate(target_sent.split()):
            decoder_input_data[i, j] = target_word_idx[word]
            if j > 0:
                decoder_target_data[i, j - 1, target_word_idx[word]] = 1

    return encoder_input_data, decoder_input_data, decoder_target_data, source_vocab_size, target_vocab_size

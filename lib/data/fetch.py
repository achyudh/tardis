import numpy as np
import pandas as pd
import nltk
import os
import swifter

from lib.data.util import preprocess, replace_unknown
from lib.data import vocab
from tqdm import tqdm


def en_de(path, source_vocab=None, target_vocab=None, reverse=False, replace_unk=True, one_hot=False, splits='train'):
    if reverse:
        source_lang, target_lang = 'de', 'en'
    else:
        source_lang, target_lang = 'en', 'de'

    if splits.lower() == 'train':
        source_data = pd.read_table(os.path.join(path, 'en_de', 'train.%s' % source_lang))
        target_data = pd.read_table(os.path.join(path, 'en_de', 'train.%s' % target_lang))
    elif splits.lower() == 'test':
        source_data = pd.read_table(os.path.join(path, 'en_de', 'test15.%s' % source_lang))
        target_data = pd.read_table(os.path.join(path, 'en_de', 'test15.%s' % target_lang))
    else:
        raise Exception("Unsupported dataset splits")

    source_data, target_data = preprocess(source_data, target_data)
    if source_vocab is None:
        # Create source vocabulary
        source_vocab = vocab.build(source_data, max_size=15000)
        print("Source vocabulary size:", len(source_vocab))

    if target_vocab is None:
        # Create target vocabulary
        target_vocab = vocab.build(target_data, max_size=15000)
        print("Target vocabulary size:", len(target_vocab))

    if replace_unk:
        source_data = [replace_unknown(x, source_vocab) for x in source_data]
        target_data = [replace_unknown(x, target_vocab) for x in target_data]

    max_target_len = max(len(nltk.word_tokenize(sent)) for sent in target_data)
    max_source_len = max(len(nltk.word_tokenize(sent)) for sent in source_data)
    target_vocab_size = len(target_vocab)

    # TODO: Pickle vocab
    num_instances = len(source_data)
    print("Source", splits, "split size:", len(source_data))
    print("Target", splits, "split size:", len(target_data))
    encoder_input_data = np.zeros((num_instances, max_source_len), dtype=np.float64)
    decoder_input_data = np.zeros((num_instances, max_target_len), dtype=np.float64)
    if one_hot:
        decoder_target_data = np.zeros((num_instances, max_target_len, target_vocab_size), dtype=np.float64)
    else:
        decoder_target_data = np.zeros((num_instances, max_target_len), dtype=np.float64)

    # Convert words to ids
    print("Converting words to indices for", splits, "split...", len(source_data))
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
    return encoder_input_data, decoder_input_data, decoder_target_data, source_vocab, target_vocab

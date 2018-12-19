import os
import re

import dill
import nltk
import swifter
import numpy as np
import pandas as pd
from tqdm import tqdm


def preprocess(source_data, target_data):
    # Convert to lowercase characters
    source_data = source_data.swifter.apply(lambda x: x.str.lower())
    target_data = target_data.swifter.apply(lambda x: x.str.lower())

    # Remove punctuation
    # Note: Does not remove underscores
    source_data = source_data.swifter.apply(lambda x: x.str.replace(r'[^\w\s]', ''))
    target_data = target_data.swifter.apply(lambda x: x.str.replace(r'[^\w\s]', ''))

    # Add SOS and EOS tokens
    target_data = target_data.swifter.apply(lambda x: 'SOS ' + x + ' EOS')

    source_data = source_data.values.flatten()
    target_data = target_data.values.flatten()
    return source_data, target_data


def load_dataset(source_data_path, target_data_path, dataset_size=None):
    if os.path.isfile(source_data_path + '.pkl') and os.path.isfile(target_data_path + '.pkl'):
        with open(source_data_path + '.pkl', 'rb') as pkl_file:
            source_data = dill.load(pkl_file)
        with open(target_data_path + '.pkl', 'rb') as pkl_file:
            target_data = dill.load(pkl_file)
    else:
        source_data = pd.read_table(source_data_path)
        target_data = pd.read_table(target_data_path)
        source_data, target_data = preprocess(source_data, target_data)
        with open(source_data_path + '.pkl', 'wb') as pkl_file:
            dill.dump(source_data, pkl_file)
        with open(target_data_path + '.pkl', 'wb') as pkl_file:
            dill.dump(target_data, pkl_file)

    if dataset_size is None or dataset_size <= 0:
        return source_data, target_data
    else:
        return source_data[:dataset_size], target_data[:dataset_size]


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

    encoder_input_data = np.zeros((num_instances, max_source_len), dtype=np.int64)
    decoder_input_data = np.zeros((num_instances, max_target_len), dtype=np.int64)
    if one_hot:
        decoder_target_data = np.zeros((num_instances, max_target_len, target_vocab_size), dtype=np.int64)
    else:
        decoder_target_data = np.zeros((num_instances, max_target_len), dtype=np.int64)

    # Convert words to ids
    for i, (source_sent, target_sent) in tqdm(enumerate(zip(source_data, target_data)), total=len(source_data)):
        for j, word in enumerate(reversed(nltk.word_tokenize(source_sent))):
            encoder_input_data[i, j] = source_vocab[word]
        for j, word in enumerate(nltk.word_tokenize(target_sent)):
            decoder_input_data[i, j] = target_vocab[word]
            if j > 0:
                if one_hot:
                    decoder_target_data[i, j - 1, target_vocab[word]] = 1
                else:
                    decoder_target_data[i, j - 1] = target_vocab[word]
    return encoder_input_data, decoder_input_data, decoder_target_data


def trim_sentences(sentences):
    trimmed_sentences = list()
    for sentence in sentences:
        trim_index = sentence.find('EOS')
        if trim_index != -1:
            trimmed_sentences.append(sentence[:trim_index])
        else:
            trimmed_sentences.append(sentence)
    return trimmed_sentences


def reverse_index(indexed_data, vocab, ravel=False):
    reversed_data = list()
    word_idx = {id: word for word, id in vocab.items()}
    for indexed_line in indexed_data:
        if ravel:
            reversed_data.append([' '.join((word_idx[x] for x in indexed_line[1:len(indexed_line)-1]))])
        else:
            reversed_data.append(' '.join((word_idx[x] for x in indexed_line[1:len(indexed_line)-1])))
    return reversed_data

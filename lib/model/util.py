import codecs
import os

import numpy as np
from tqdm import tqdm


def embedding_matrix(model_path, vocab, embed_dim=300):
    embed_index = dict()
    file_sizes = {'wiki.en.vec': 2519428, 'wiki.de.vec': 2275261, 'wiki.fr.vec': 1152450}
    if model_path not in file_sizes:
        Exception("Unsupported embedding file")

    with codecs.open(model_path, encoding='utf-8') as embedding_file:
        for line in tqdm(embedding_file, total=file_sizes[os.path.basename(model_path)]):
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embed_index[word] = coefs

    embed_matrix = np.zeros((len(vocab), embed_dim))
    for word, i in vocab.items():
        embedding_vector = embed_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            embed_matrix[i] = embedding_vector
    return embed_matrix

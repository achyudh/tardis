import numpy as np
import codecs
from tqdm import tqdm


def embedding_matrix(model_path, vocab, embed_dim=300):
    embed_index = dict()
    with codecs.open(model_path, encoding='utf-8') as embedding_file:
        for line in tqdm(embedding_file):
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embed_index[word] = coefs

    word_index = dict([(word, id) for id, word in enumerate(vocab)])
    embed_matrix = np.zeros((len(word_index), embed_dim))
    for word, i in word_index.items():
        embedding_vector = embed_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            embed_matrix[i] = embedding_vector
    return embed_matrix

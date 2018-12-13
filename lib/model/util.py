import numpy as np
import gensim


def embedding_matrix(word_index, model_path='data/embeddings/word/googlenews_size300.bin', binary=True):
    if binary:
        size = int(model_path.split('.')[-2].split('/')[-1].split('_')[1][4:])
    else:
        size = int(model_path.split('/')[-1].split('_')[1][4:])
    w2v = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary)
    embedding_map = np.zeros((len(word_index) + 1, size))
    for word, i in word_index.items():
        if word in w2v:
            embedding_map[i] = w2v[word]
    return embedding_map

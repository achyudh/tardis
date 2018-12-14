import nltk


def build(lines, max_size=None):
    vocab = dict()
    vocab['UNK'] = float('inf')
    vocab['SOS'] = float('inf')
    vocab['EOS'] = float('inf')

    for source_line in lines:
        for word in nltk.word_tokenize(source_line):
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] +=1
    if max_size is None or len(vocab) < max_size:
        vocab = sorted(vocab.keys(), key=lambda x: vocab[x], reverse=True)
    else:
        vocab = sorted(vocab.keys(), key=lambda x: vocab[x], reverse=True)[:max_size]
    return {word: id for id, word in enumerate(vocab)}

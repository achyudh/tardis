import json

import sacrebleu
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from lib.data.util import reverse_indexing

# Global variables
DATASET = None
TARGET_VOCAB = None


def bleu_score(reference, candidate, log_outputs=True):
    # TODO: Find a workaround for Keras metric API limitation
    reference = [[x] for x in reference]
    # reference = reverse_indexing(reference, TARGET_VOCAB, ravel=True)
    candidate = reverse_indexing(candidate, TARGET_VOCAB)
    if log_outputs:
        with open('%s_output.json' % DATASET, 'w') as json_file:
            json.dump(list(zip(reference, candidate)), json_file)
    return corpus_bleu(reference, candidate, smoothing_function=SmoothingFunction().method4)


def multi_bleu_score(candidate, target_vocab):
    lang_pair = '-'.join(DATASET.split('_'))
    candidate = reverse_indexing(candidate, target_vocab)
    _, *refs = sacrebleu.download_test_set('wmt14', lang_pair)
    bleu = sacrebleu.corpus_bleu(candidate, refs)
    return bleu.score



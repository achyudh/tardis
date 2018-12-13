import sacrebleu
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from lib.data.util import reverse_indexing


def bleu_score(reference, candidate, target_vocab):
    # TODO: Find a workaround for Keras metric API limitation
    reference = reverse_indexing(reference, target_vocab)
    candidate = reverse_indexing(candidate, target_vocab)
    return corpus_bleu(reference, candidate, smoothing_function=SmoothingFunction().method4)


def multi_bleu_score(candidate, target_vocab, dataset):
    lang_pair = '-'.join(dataset.split('_'))
    candidate = reverse_indexing(candidate, target_vocab)
    _, *refs = sacrebleu.download_test_set('wmt14', lang_pair)
    bleu = sacrebleu.corpus_bleu(candidate, refs)
    return bleu.score



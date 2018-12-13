import os
from copy import deepcopy

from lib.data import fetch
from lib.model.util import embedding_matrix
from lib.model.args import get_args
from lib.model.seq2seq import Seq2Seq, TinySeq2Seq

if __name__ == '__main__':
    # Select GPU based on args
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.dataset == 'en_de':
        encoder_train_input, decoder_train_input, decoder_train_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path)
        encoder_test_input, decoder_test_input, decoder_test_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, source_vocab, target_vocab, splits='test')
        source_embedding_map = embedding_matrix('data/embeddings/wiki.en.vec', source_vocab)
        target_embedding_map = embedding_matrix('data/embeddings/wiki.de.vec', target_vocab)
    elif args.dataset == 'de_en':
        encoder_train_input, decoder_train_input, decoder_train_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, reverse=True)
        encoder_test_input, decoder_test_input, decoder_test_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, source_vocab, target_vocab, reverse=True, splits='test')
        source_embedding_map = embedding_matrix('data/embeddings/wiki.de.vec', source_vocab)
        target_embedding_map = embedding_matrix('data/embeddings/wiki.en.vec', target_vocab)
    else:
        raise Exception("Unsupported dataset")

    model = None
    model_config = deepcopy(args)
    source_vocab_size = len(source_vocab)
    target_vocab_size = len(target_vocab)
    model_config.source_vocab = source_vocab
    model_config.target_vocab = target_vocab
    model_config.source_vocab_size = source_vocab_size
    model_config.target_vocab_size = target_vocab_size
    model_config.source_embedding_map = source_embedding_map
    model_config.target_embedding_map = target_embedding_map

    if args.cpu:
        model = TinySeq2Seq(args)
    else:
        model = Seq2Seq(model_config)

    print(encoder_train_input.shape, decoder_train_input.shape, decoder_train_target.shape)
    model.train(encoder_train_input, decoder_train_input, decoder_train_target)
    model.evaluate(encoder_train_input, decoder_train_input, decoder_train_target)

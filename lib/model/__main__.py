import os
from copy import deepcopy

from lib.data import fetch
from lib.data.generator import WMTSequence
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
            fetch.en_de(args.dataset_path, source_vocab, target_vocab, one_hot=True, splits='test')
        source_embedding_map = embedding_matrix('data/embeddings/wiki.en.vec', source_vocab)
        target_embedding_map = embedding_matrix('data/embeddings/wiki.de.vec', target_vocab)
    elif args.dataset == 'de_en':
        encoder_train_input, decoder_train_input, decoder_train_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, reverse=True)
        encoder_test_input, decoder_test_input, decoder_test_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, source_vocab, target_vocab, one_hot=True, reverse=True, splits='test')
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

    training_generator = WMTSequence(encoder_train_input, decoder_train_input, decoder_train_target, model_config)
    # TODO: Create a separate validation split for the validation generator
    # validation_generator = WMTSequence(encoder_test_input, decoder_test_input, decoder_test_target, model_config)

    if args.cpu:
        model = TinySeq2Seq(args)
    else:
        model = Seq2Seq(model_config)

    # model.train(encoder_train_input, decoder_train_input, decoder_train_target)
    model.train_generator(generator=training_generator)
    model.evaluate(encoder_test_input, decoder_test_input, decoder_test_target)

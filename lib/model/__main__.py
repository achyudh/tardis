import os
from copy import deepcopy

from elephas.spark_model import SparkModel

from pyspark import SparkContext, SparkConf

from lib.data import fetch
from lib.data.generator import WMTSequence
from lib.model.util import embedding_matrix
from lib.model import metrics
from lib.model.args import get_args
from lib.model.seq2seq import Seq2Seq, TinySeq2Seq

root_dir = os.getcwd()

if __name__ == '__main__':
    # Select GPU based on args
    args = get_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.dataset == 'en_de':
        encoder_train_input, decoder_train_input, decoder_train_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path)
        encoder_dev_input, decoder_dev_input, decoder_dev_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, source_vocab, target_vocab, splits='dev')
        encoder_test_input, decoder_test_input, decoder_test_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, source_vocab, target_vocab, one_hot=True, splits='test')
        source_embedding_map = embedding_matrix(os.path.join(args.embedding_path, 'wiki.en.vec'), source_vocab)
        target_embedding_map = embedding_matrix(os.path.join(args.embedding_path, 'wiki.de.vec'), target_vocab)

    elif args.dataset == 'de_en':
        encoder_train_input, decoder_train_input, decoder_train_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, reverse=True)
        encoder_dev_input, decoder_dev_input, decoder_dev_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, source_vocab, target_vocab, reverse=True, splits='dev')
        encoder_test_input, decoder_test_input, decoder_test_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, source_vocab, target_vocab, one_hot=True, reverse=True, splits='test')
        source_embedding_map = embedding_matrix(os.path.join(args.embedding_path, 'wiki.de.vec'), source_vocab)
        target_embedding_map = embedding_matrix(os.path.join(args.embedding_path, 'wiki.en.vec'), target_vocab)

    elif args.dataset == 'en_vi':
        encoder_train_input, decoder_train_input, decoder_train_target, source_vocab, target_vocab = \
            fetch.en_vi(args.dataset_path)
        encoder_dev_input, decoder_dev_input, decoder_dev_target, source_vocab, target_vocab = \
            fetch.en_vi(args.dataset_path, source_vocab, target_vocab, splits='dev')
        encoder_test_input, decoder_test_input, decoder_test_target, source_vocab, target_vocab = \
            fetch.en_vi(args.dataset_path, source_vocab, target_vocab, one_hot=True, splits='test')
        source_embedding_map = embedding_matrix(os.path.join(args.embedding_path, 'wiki.en.vec'), source_vocab)
        target_embedding_map = embedding_matrix(os.path.join(args.embedding_path, 'wiki.vi.vec'), target_vocab)

    elif args.dataset == 'vi_en':
        encoder_train_input, decoder_train_input, decoder_train_target, source_vocab, target_vocab = \
            fetch.en_vi(args.dataset_path, reverse=True)
        encoder_dev_input, decoder_dev_input, decoder_dev_target, source_vocab, target_vocab = \
            fetch.en_vi(args.dataset_path, source_vocab, target_vocab, reverse=True, splits='dev')
        encoder_test_input, decoder_test_input, decoder_test_target, source_vocab, target_vocab = \
            fetch.en_vi(args.dataset_path, source_vocab, target_vocab, one_hot=True, reverse=True, splits='test')
        source_embedding_map = embedding_matrix(os.path.join(args.embedding_path, 'wiki.vi.vec'), source_vocab)
        target_embedding_map = embedding_matrix(os.path.join(args.embedding_path, 'wiki.en.vec'), target_vocab)

    else:
        raise Exception("Unsupported dataset")

    model = None
    metrics.DATASET = args.dataset
    metrics.TARGET_VOCAB = target_vocab

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
    validation_generator = WMTSequence(encoder_dev_input, decoder_dev_input, decoder_dev_target, model_config)

    if args.cpu:
        model = TinySeq2Seq(args)
    else:
        model = Seq2Seq(model_config)
        
    if args.ensemble:
        # TODO: increase number of workers and set master
        conf = SparkConf().setAppName('Tardis').set('spark.executor.instances', '1')
        sc = SparkContext(conf=conf).addFile(path=os.path.join(root_dir, 'dist', 'tardis-0.0.1-py3.6.egg'))
        model = SparkModel(model, frequency='epoch') # Distributed ensemble

    model.train_generator(training_generator, validation_generator)
    model.evaluate(encoder_test_input, decoder_test_input, decoder_test_target)

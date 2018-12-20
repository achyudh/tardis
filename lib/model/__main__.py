import os
from copy import deepcopy

import numpy as np
import tensorflow as tf
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from keras.backend.tensorflow_backend import set_session
from pyspark import SparkConf, SparkContext

from lib.data import fetch
from lib.data.generator import WMTSequence
from lib.model import metrics
from lib.model.args import get_args
from lib.model.seq2seq import Seq2Seq, EnsembleSeq2Seq
from lib.model.distributed.seq2seq import Seq2Seq as DistributedSeq2Seq
from lib.model.distributed.seq2seq import EnsembleSeq2Seq as DistributedEnsembleSeq2Seq
from lib.model.util import embedding_matrix, load_weights
from lib.model.distributed.util import EncoderSlice, DecoderSlice

if __name__ == '__main__':
    # Select GPU based on args
    args = get_args()
    root_dir = os.getcwd()

    # Set GPU usage
    if not args.cpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        set_session(sess)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    if args.dataset == 'en_de':
        encoder_train_input, decoder_train_input, decoder_train_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, dataset_size=args.dataset_size, source_vocab_size=args.source_vocab_size,
                        target_vocab_size=args.target_vocab_size)
        encoder_dev_input, decoder_dev_input, decoder_dev_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, source_vocab, target_vocab, splits='dev')
        encoder_test_input, decoder_test_input, decoder_test_target, raw_test_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, source_vocab, target_vocab, one_hot=True, splits='test')

        source_embedding_map = embedding_matrix(os.path.join(args.embedding_path, 'wiki.en.vec'), source_vocab)
        target_embedding_map = embedding_matrix(os.path.join(args.embedding_path, 'wiki.de.vec'), target_vocab)

    elif args.dataset == 'de_en':
        encoder_train_input, decoder_train_input, decoder_train_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, dataset_size=args.dataset_size, source_vocab_size=args.source_vocab_size,
                        target_vocab_size=args.target_vocab_size, reverse_lang=True)
        encoder_dev_input, decoder_dev_input, decoder_dev_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, source_vocab, target_vocab, reverse_lang=True, splits='dev')
        encoder_test_input, decoder_test_input, decoder_test_target, raw_test_target, source_vocab, target_vocab = \
            fetch.en_de(args.dataset_path, source_vocab, target_vocab, one_hot=True, reverse_lang=True, splits='test')

        source_embedding_map = embedding_matrix(os.path.join(args.embedding_path, 'wiki.de.vec'), source_vocab)
        target_embedding_map = embedding_matrix(os.path.join(args.embedding_path, 'wiki.en.vec'), target_vocab)

    elif args.dataset == 'en_vi':
        encoder_train_input, decoder_train_input, decoder_train_target, source_vocab, target_vocab = \
            fetch.en_vi(args.dataset_path, dataset_size=args.dataset_size, source_vocab_size=args.source_vocab_size,
                        target_vocab_size=args.target_vocab_size)
        encoder_dev_input, decoder_dev_input, decoder_dev_target, source_vocab, target_vocab = \
            fetch.en_vi(args.dataset_path, source_vocab, target_vocab, splits='dev')
        encoder_test_input, decoder_test_input, decoder_test_target, raw_test_target, source_vocab, target_vocab = \
            fetch.en_vi(args.dataset_path, source_vocab, target_vocab, one_hot=True, splits='test')

        source_embedding_map = embedding_matrix(os.path.join(args.embedding_path, 'wiki.en.vec'), source_vocab)
        target_embedding_map = embedding_matrix(os.path.join(args.embedding_path, 'wiki.vi.vec'), target_vocab)

    elif args.dataset == 'vi_en':
        encoder_train_input, decoder_train_input, decoder_train_target, source_vocab, target_vocab = \
            fetch.en_vi(args.dataset_path, dataset_size=args.dataset_size, source_vocab_size=args.source_vocab_size,
                        target_vocab_size=args.target_vocab_size, reverse=True)
        encoder_dev_input, decoder_dev_input, decoder_dev_target, source_vocab, target_vocab = \
            fetch.en_vi(args.dataset_path, source_vocab, target_vocab, reverse=True, splits='dev')
        encoder_test_input, decoder_test_input, decoder_test_target, raw_test_target, source_vocab, target_vocab = \
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
    if ',' in args.devices:
        model_config.devices = args.devices.split(',')
    else:
        model_config.devices = (args.devices, args.devices)
    model_config.source_vocab = source_vocab
    model_config.target_vocab = target_vocab
    model_config.source_vocab_size = source_vocab_size
    model_config.target_vocab_size = target_vocab_size
    model_config.source_embedding_map = source_embedding_map
    model_config.target_embedding_map = target_embedding_map

    if args.distributed:
        if args.single_threaded_worker:
            conf = SparkConf().setAppName('tardis').setMaster('local')
        else:
            conf = SparkConf().setAppName('tardis').setMaster('local[*]')
        sc = SparkContext.getOrCreate(conf=conf)

        generator_config = deepcopy(args)
        generator_config.batch_size = 1024
        generator_config.target_vocab = target_vocab
        model_config.input_split_index = encoder_train_input.shape[1]
        training_generator = WMTSequence(encoder_train_input, decoder_train_input, decoder_train_target, model_config)

        for raw_train_input, decoder_train_target in training_generator:
            encoder_train_input, decoder_train_input = raw_train_input
            train_input = np.hstack((encoder_train_input, decoder_train_input))
            train_rdd = to_simple_rdd(sc, train_input, decoder_train_target)

            if args.ensemble:
                model = DistributedEnsembleSeq2Seq(model_config)
            else:
                model = DistributedSeq2Seq(model_config)

            spark_model = SparkModel(model.model,
                                     frequency='epoch',
                                     mode='synchronous',
                                     batch_size=args.batch_size,
                                     custom_objects={'EncoderSlice': EncoderSlice, 'DecoderSlice': DecoderSlice})

            spark_model.fit(train_rdd,
                            batch_size=model_config.batch_size,
                            epochs=model_config.epochs,
                            validation_split=0.0,
                            verbose=1)

        model.evaluate(encoder_test_input, raw_test_target)

    else:
        training_generator = WMTSequence(encoder_train_input, decoder_train_input, decoder_train_target, model_config)
        validation_generator = WMTSequence(encoder_dev_input, decoder_dev_input, decoder_dev_target, model_config)

        if args.ensemble:
            model = EnsembleSeq2Seq(model_config)
        else:
            model = Seq2Seq(model_config)

        if args.load_checkpoint:
            load_weights(model.model, args.checkpoint_path)
        model.train_generator(training_generator, validation_generator)
        model.evaluate(encoder_test_input, raw_test_target)

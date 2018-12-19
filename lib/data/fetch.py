from lib.data import vocab
from lib.data.util import *


def en_de(path, source_vocab=None, target_vocab=None, reverse_lang=False, replace_unk=True, one_hot=False,
          dataset_size=None, source_vocab_size=None, target_vocab_size=None, splits='train'):

    if reverse_lang:
        source_lang, target_lang = 'de', 'en'
    else:
        source_lang, target_lang = 'en', 'de'

    if splits.lower() == 'train':
        source_data_path = os.path.join(path, 'en_de', 'train.%s' % source_lang)
        target_data_path = os.path.join(path, 'en_de', 'train.%s' % target_lang)
    elif splits.lower() == 'dev':
        source_data_path = os.path.join(path, 'en_de', 'test12.%s' % source_lang)
        target_data_path = os.path.join(path, 'en_de', 'test12.%s' % target_lang)
    elif splits.lower() == 'test':
        source_data_path = os.path.join(path, 'en_de', 'test15.%s' % source_lang)
        target_data_path = os.path.join(path, 'en_de', 'test15.%s' % target_lang)
    else:
        raise Exception("Unsupported dataset splits")

    source_data, target_data = load_dataset(source_data_path, target_data_path, dataset_size)

    if splits.lower() == 'test':
        raw_target_data = [x[4:-4] for x in target_data]

    if source_vocab is None:
        # Create source vocabulary
        source_vocab = vocab.build(source_data, max_size=source_vocab_size)
        print("Source vocabulary size:", len(source_vocab))

    if target_vocab is None:
        # Create target vocabulary
        target_vocab = vocab.build(target_data, max_size=target_vocab_size)
        print("Target vocabulary size:", len(target_vocab))

    if replace_unk:
        source_data = [replace_unknown(x, source_vocab) for x in source_data]
        target_data = [replace_unknown(x, target_vocab) for x in target_data]

    print("Source", splits, "split size:", len(source_data))
    print("Target", splits, "split size:", len(target_data))
    print("Converting words to indices for", splits, "split...")
    encoder_input_data, decoder_input_data, decoder_target_data = build_indices(source_data, target_data,
                                                                                source_vocab, target_vocab, one_hot)

    if splits.lower() == 'test':
        return encoder_input_data, decoder_input_data, decoder_target_data, raw_target_data, source_vocab, target_vocab
    else:
        return encoder_input_data, decoder_input_data, decoder_target_data, source_vocab, target_vocab


def en_vi(path, source_vocab=None, target_vocab=None, reverse=False, replace_unk=True, one_hot=False,
          dataset_size=None, source_vocab_size=None, target_vocab_size=None, splits='train'):

    if reverse:
        source_lang, target_lang = 'vi', 'en'
    else:
        source_lang, target_lang = 'en', 'vi'

    if splits.lower() == 'train':
        source_data_path = os.path.join(path, 'en_vi', 'train.%s' % source_lang)
        target_data_path = os.path.join(path, 'en_vi', 'train.%s' % target_lang)
    elif splits.lower() == 'dev':
        source_data_path = os.path.join(path, 'en_vi', 'test12.%s' % source_lang)
        target_data_path = os.path.join(path, 'en_vi', 'test12.%s' % target_lang)
    elif splits.lower() == 'test':
        source_data_path = os.path.join(path, 'en_vi', 'test13.%s' % source_lang)
        target_data_path = os.path.join(path, 'en_vi', 'test13.%s' % target_lang)
    else:
        raise Exception("Unsupported dataset splits")

    source_data, target_data = load_dataset(source_data_path, target_data_path, dataset_size)

    if splits.lower() == 'test':
        raw_target_data = [x[4:-4] for x in target_data]

    if source_vocab is None:
        # Create source vocabulary
        source_vocab = vocab.build(source_data, max_size=source_vocab_size)
        print("Source vocabulary size:", len(source_vocab))

    if target_vocab is None:
        # Create target vocabulary
        target_vocab = vocab.build(target_data, max_size=target_vocab_size)
        print("Target vocabulary size:", len(target_vocab))

    if replace_unk:
        source_data = [replace_unknown(x, source_vocab) for x in source_data]
        target_data = [replace_unknown(x, target_vocab) for x in target_data]

    print("Source", splits, "split size:", len(source_data))
    print("Target", splits, "split size:", len(target_data))
    print("Converting words to indices for", splits, "split...")
    encoder_input_data, decoder_input_data, decoder_target_data = build_indices(source_data, target_data, source_vocab,
                                                                                target_vocab, one_hot)


    if splits.lower() == 'test':
        return encoder_input_data, decoder_input_data, decoder_target_data, raw_target_data, source_vocab, target_vocab
    else:
        return encoder_input_data, decoder_input_data, decoder_target_data, source_vocab, target_vocab

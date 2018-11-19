from lib.model.args import get_args
from lib.model.seq2seq import Seq2Seq, TinySeq2Seq

from lib.data.data_utils import load_data, preprocess

if __name__ == '__main__':
    args = get_args()

    dataset = args.dataset # e.g: en_fr
    dataset_path = args.dataset_path

    [source_lang, target_lang] = dataset.split('_')
    encoder_input_data, decoder_input_data, decoder_target_data, source_vocab_size, target_vocab_size = load_data(dataset_path, source_lang, target_lang)

    model = None
    if args.cpu:
        model = TinySeq2Seq(args)
    else:
        model = Seq2Seq(encoder_input_data, decoder_input_data, decoder_target_data, source_vocab_size, target_vocab_size, args)

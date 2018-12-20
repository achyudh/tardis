import os

from argparse import ArgumentParser


def get_args():
    root_dir = os.getcwd()
    parser = ArgumentParser(description="Seq2Seq models for Neural Machine Translation (NMT)")
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--devices', type=str, default='0,1')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--single-threaded-worker', action='store_true')
    parser.add_argument('--epochs', type=int, default=7)
    parser.add_argument('--num-models', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=1000)
    parser.add_argument('--recurrent-unit', type=str, default='lstm')
    parser.add_argument('--num-encoder-layers', type=int, default=2)
    parser.add_argument('--num-decoder-layers', type=int, default=2)
    parser.add_argument('--dataset-size', type=int, default=0)
    parser.add_argument('--source-vocab-size', type=int, default=10000)
    parser.add_argument('--target-vocab-size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--decay', type=float, default=0.0)
    parser.add_argument('--beam-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--dataset', type=str, default='en_vi', choices=['en_de', 'de_en', 'en_vi', 'vi_en'])
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--load-checkpoint', action='store_true')
    parser.add_argument('--checkpoint-path', type=str, default=os.path.join(root_dir, 'data', 'checkpoints', 'en_vi',
                                                                            'lstm_el2_dl2_ds0_sv10000_tv10000_ep01.hdf5'))
    parser.add_argument('--embedding-dim', type=int, default=300)
    parser.add_argument('--embedding-path', help='embedding file path', default=os.path.join(root_dir, 'data', 'embeddings'))
    parser.add_argument('--dataset-path', help='dataset directory', default=os.path.join(root_dir, 'data', 'datasets'))
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--num-workers', type=int, default=1)

    args = parser.parse_args()
    return args

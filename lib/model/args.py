import os

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Seq2Seq models for Neural Machine Translation (NMT)")
    parser.add_argument('--cpu', type=bool, default='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--dataset', type=str, default='en_de', choices=['en_de', 'en_fr'])
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--save_path', type=str, default='data/checkpoints')
    parser.add_argument('--embedding_dim', type=int, default=300)
    parser.add_argument('--embedding_mode', type=str, default='static', choices=['rand', 'static', 'non-static'])
    parser.add_argument('--embedding_path', help='embedding file path', default=os.path.join(os.pardir, 'data', 'embeddings', 'googlenews_300.bin'))
    parser.add_argument('--dataset_path', help='dataset directory', default=os.path.join(os.pardir, 'data', 'datasets'))
    parser.add_argument('--word_vectors_file', help='word vectors filename', default='GoogleNews-vectors-negative300.txt')
    parser.add_argument('--weight_decay', type=float, default=0)

    args = parser.parse_args()
    return args

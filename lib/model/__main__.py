from lib.model.args import get_args
from lib.model.seq2seq import Seq2Seq, TinySeq2Seq

if __name__ == '__main__':
    args = get_args()
    if args.cpu:
        TinySeq2Seq(args)
    else:
        Seq2Seq(args)

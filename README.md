# Tardis
> Ensemble Seq2Seq neural machine translation model running on PySpark using Elephas

An ensemble of the neural machine translation model from from Sequence to Sequence Learning with Neural Networks by Sutskever et al. [1] trained over PySpark using Elephas. We assess the effectiveness of our model on the EN-FR and EN-DE datasets from WMT-14.

## Prerequisites
* Keras >= 2.2.4
* Elephas >= 0.4
* Pandas >= 0.23.4

## Getting started
* Download the en_de dataset under `data/datasets/en_de`:
  * Download [train.en](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en) and [train.de](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de)
  * Download [newstest2012.en](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.en), [newstest2012.de](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.de), [newstest2015.en](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.en) and [newstest2015.de](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.de)
* Repeat the same process for the en_vi dataset under `data/datasets/en_vi`

* Download the FastText WikiText embeddings for [English](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec), [German](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec) and [Vietnamese](https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.vi.vec)

* To run the single node Seq2Seq model on a GPU, issue the following command from the project root directory:
  - `python -m lib.model --gpu <gpu_no> --dataset <lang_pair> --batch-size <batch_size>`
* To run the single node TinySeq2Seq model on a CPU, issue the following command from the project root directory:
  - `python -m lib.model --cpu [--ensemble] --dataset <lang_pair> --batch-size <batch_size>`
* To run the TinySeq2Seq ensemble on multiple nodes:
  * Generate the egg file by running - must run after every change in the code:
  `python setup.py bdist_egg`
  * Issue the following command from the project root directory: (WIP)
  - `spark-submit --driver-memory 1G -m lib/model/__main__.py --cpu [--ensemble] --dataset <lang_pair> --batch-size <batch_size> --recurrent-unit gru`

 Note: Beam search is used by default during testing. Add the flag `--beam-size 0` to use greedy search.

## References

[1] Sutskever, I., Vinyals, O. and Le, Q.V., 2014. Sequence to sequence learning with neural networks. In Advances in neural information processing systems (pp. 3104-3112).
[2] Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. 2015. Effective Approaches to Attention-based Neural Machine Translation. In Empirical Methods in Natural Language Processing (EMNLP).

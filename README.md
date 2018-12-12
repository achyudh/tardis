# Tardis
> Ensemble Seq2Seq neural machine translation model running on PySpark using Elephas

An ensemble of the neural machine translation model from from Sequence to Sequence Learning with Neural Networks by Sutskever et al. [1] trained over PySpark using Elephas. We assess the effectiveness of our model on the EN-FR and EN-DE datasets from WMT-14.

## Prerequisites
* Keras >= 2.2.4
* Elephas >= 0.4
* Pandas >= 0.23.4

## Getting started
* Download the en_de dataset from https://nlp.stanford.edu/projects/nmt/ and move it to `data/datasets/en_de`. Similarly, the en_fr dataset from http://statmt.org/wmt14/translation-task.html should be placed in `data/datasets/en_fr.
* To run the single node Seq2Seq model on a GPU, issue the following command from the project root directory: 
  - `python -m lib.model --gpu <gpu_no> --dataset <lang_pair> --batch-size <batch_size>`
* To run the single node TinySeq2Seq model on a CPU, issue the following command from the project root directory: 
  - `python -m lib.model --cpu --dataset <lang_pair> --batch-size <batch_size>`

## References

[1] Sutskever, I., Vinyals, O. and Le, Q.V., 2014. Sequence to sequence learning with neural networks. In Advances in neural information processing systems (pp. 3104-3112).
[2] Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. 2015. Effective Approaches to Attention-based Neural Machine Translation. In Empirical Methods in Natural Language Processing (EMNLP).

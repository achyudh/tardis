# Tardis
### Seq2Seq Neural Machine Translation over PySpark

An ensemble of the neural machine translation model from from Sequence to Sequence Learning with Neural Networks by Sutskever et al. [1] trained over PySpark using Elephas. We assess the effectiveness of our model on the EN-FR and EN-DE datasets from WMT-14.

## References

[1] Sutskever, I., Vinyals, O. and Le, Q.V., 2014. Sequence to sequence learning with neural networks. In Advances in neural information processing systems (pp. 3104-3112).
[2] Minh-Thang Luong, Hieu Pham, and Christopher D. Manning. 2015. Effective Approaches to Attention-based Neural Machine Translation. In Empirical Methods in Natural Language Processing (EMNLP).

## How to Run

Get toy EN-FR dataset: http://www.manythings.org/anki/fra-eng.zip
Put under /data

Run

```
python main.py --gpu <gpu_no> --dataset <lang_pair> --dataset_path <data_path>
```

<img src="espresso/espresso_logo.png" align="right" style="padding-left: 20px" height="160px" />

# Espresso

Espresso is an open-source, modular, extensible end-to-end neural automatic speech recognition (ASR) toolkit based on the deep learning library [PyTorch](https://github.com/pytorch/pytorch) and the popular neural machine translation toolkit [`fairseq`](https://github.com/pytorch/fairseq). Espresso supports distributed training across GPUs and computing nodes, and features various decoding approaches commonly employed in ASR, including look-ahead word-based language model fusion, for which a fast, parallelized decoder is implemented. 

We provide state-of-the-art training recipes for the following speech datasets:
 - [WSJ](https://github.com/freewym/espresso/tree/master/examples/asr_wsj)
 - [LibriSpeech](https://github.com/freewym/espresso/tree/master/examples/asr_librispeech)
 - [Switchboard](https://github.com/freewym/espresso/tree/master/examples/asr_swbd)

### What's New:

- September 2019: We are in an effort of isolating Espresso from fairseq, resulting in a standalone package that can be directly `pip install`ed.

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.2.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library with the `--cuda_ext` and `--deprecated_fused_adam` options

Currently Espresso only support installing from source.

To install Espresso from source and develop locally:
```bash
git clone https://github.com/freewym/espresso
cd espresso
pip install --editable .
pip install kaldi_io
pip install sentencepiece
cd espresso/tools; make KALDI=<path/to/a/compiled/kaldi/directory>
```
add your Python path to `PATH` variable in `examples/asr_<dataset>/path.sh`, the current default is `~/anaconda3/bin`.

kaldi\_io is required for reading kaldi scp files. sentencepiece is required for subword pieces training/encoding.
Kaldi is required for data preparation, feature extraction and scoring for some datasets (e.g., Switchboard).

# License
Espresso is MIT-licensed.

# Citation

Please cite Espresso as:

```bibtex
@inproceedings{wang2019espresso,
  title = {Espresso: A Fast End-to-end Neural Speech Recognition Toolkit},
  author = {Yiming Wang and Tongfei Chen and Hainan Xu 
            and Shuoyang Ding and Hang Lv and Yiwen Shao 
            and Nanyun Peng and Lei Xie and Shinji Watanabe 
            and Sanjeev Khudanpur},
  booktitle = {2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  year = {2019},
}
```

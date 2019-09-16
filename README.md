<img src="espresso/espresso_logo.png" align="right" style="padding-left: 20px" height="160px" />

# Espresso

Espresso is an open-source, modular, extensible end-to-end neural automatic speech recognition (ASR) toolkit based on the deep learning library [PyTorch](https://github.com/pytorch/pytorch) and the popular neural machine translation toolkit [`fairseq`](https://github.com/pytorch/fairseq). Espresso supports distributed training across GPUs and computing nodes, and features various decoding approaches commonly employed in ASR, including look-ahead word-based language model fusion, for which a fast, parallelized decoder is implemented. 

We provide state-of-the-art training recipes for the following speech datasets:
 * [WSJ](https://github.com/freewym/espresso/tree/master/examples/asr_wsj)
 * [LibriSpeech](https://github.com/freewym/espresso/tree/master/examples/asr_librispeech)
 * [Switchboard](https://github.com/freewym/espresso/tree/master/examples/asr_swbd)

### What's New:

* June 2020: Transformer recipes released.
* April 2020: Both [E2E LF-MMI](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1423.pdf) (using [PyChain](https://github.com/YiwenShaoStephen/pychain)) and Cross-Entropy training for hybrid ASR are now supported. WSJ recipes are provided [here](https://github.com/freewym/espresso/tree/master/examples/asr_wsj/run_chain_e2e.sh) and [here](https://github.com/freewym/espresso/tree/master/examples/asr_wsj/run_xent.sh) as examples, respectively.
* March 2020: SpecAugment is supported and relevant recipes are released.
* September 2019: We are in an effort of isolating Espresso from fairseq, resulting in a standalone package that can be directly `pip install`ed.

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install Espresso** from source and develop locally:

``` bash
git clone https://github.com/freewym/espresso
cd espresso
pip install --editable .

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
pip install kaldi_io
pip install sentencepiece
cd espresso/tools; make KALDI=<path/to/a/compiled/kaldi/directory>
```

add your Python path to `PATH` variable in `examples/asr_<dataset>/path.sh`, the current default is `~/anaconda3/bin`.

kaldi\_io is required for reading kaldi scp files. sentencepiece is required for subword pieces training/encoding.
Kaldi is required for data preparation, feature extraction, scoring for some datasets (e.g., Switchboard), and decoding for all hybrid systems.
* If you want to use [PyChain](https://github.com/YiwenShaoStephen/pychain) for [LF-MMI](https://www.isca-speech.org/archive/Interspeech_2016/pdfs/0595.PDF) training, you also need to install PyChain (and OpenFst):

edit `PYTHON_DIR` variable in `espresso/tools/Makefile` (default: `~/anaconda3/bin`), and then
```bash
cd espresso/tools; make openfst pychain
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

# License

Espresso is MIT-licensed.

# Citation

Please cite Espresso as:

``` bibtex
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

#!/bin/bash
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e -o pipefail

stage=0
ngpus=1 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid

# model and data related
affix=
test_set="dev eval"
checkpoint=checkpoint_best.pt
wake_word0="HiXiaowen"
wake_word1="NihaoWenwen"


. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

dir=exp/tdnn_k2_${affix:+_$affix}

if [ ${stage} -le 0 ]; then
  echo "Stage 0: Data Preparation"
  mkdir -p data/log
  log_file=data/log/data_prep.log
  $train_cmd $log_file ./local/data_prep.py --data-dir data --seed 1 \
    --num-workers 16 --max-remaining-duration 0.3 --overlap-duration 0.3
fi

if [ ${stage} -le 1 ]; then
  echo "Stage 1: Graphs Generation"
  echo "Prepare the lexicon"
  mkdir -p data/lang
  cat > data/lang/lexiconp.txt <<EOF
HiXiaowen 1.0 hixiaowen
NihaoWenwen 1.0 nihaowenwen
FREETEXT 1.0 freetext
<sil> 1.0 SIL
EOF

  utils/lang/make_lexicon_fst.py --sil-prob 0.5 --sil-phone SIL \
    data/lang/lexiconp.txt > data/lang/L.fst.txt.sym

  echo "Prepare phones symbol table"
  cat > data/lang/phones.txt <<EOF
<eps> 0
SIL 1
hixiaowen 2
nihaowenwen 3
freetext 4
EOF

  echo "Prepare words symbol table"
  cat > data/lang/words.txt <<EOF
<eps> 0
<sil> 1
FREETEXT 2
HiXiaowen 3
NihaoWenwen 4
EOF

  utils/sym2int.pl -f 3 data/lang/phones.txt <data/lang/L.fst.txt.sym - | \
    utils/sym2int.pl -f 4 data/lang/words.txt - > data/lang/L.fst.txt

  echo "Prepare HMMs for phones"
  id_sil=`cat data/lang/phones.txt | grep "SIL" | awk '{print $2}'`
  id_freetext=`cat data/lang/phones.txt | grep "freetext" | awk '{print $2}'`
  id_word0=`cat data/lang/phones.txt | grep "hixiaowen" | awk '{print $2}'`
  id_word1=`cat data/lang/phones.txt | grep "nihaowenwen" | awk '{print $2}'`
  id_freetext=`cat data/lang/phones.txt | grep "freetext" | awk '{print $2}'`

   cat > data/lang/hmm_sil.txt <<EOF
0 0 0 $id_sil 0.5
0 0 1 0 0.5
0
EOF

  cat > data/lang/hmm_freetext.txt <<EOF
0 0 2 $id_freetext 0.5
0 1 3 0 0.5
1 1 4 0 0.5
1 2 5 0 0.5
2 2 6 0 0.5
2 3 7 0 0.5
3 3 8 0 0.5
3 0 9 0 0.5
0
EOF

  cat > data/lang/hmm_hixiaowen.txt <<EOF
0 0 10 $id_word0 0.5
0 1 11 0 0.5
1 1 12 0 0.5
1 2 13 0 0.5
2 2 14 0 0.5
2 3 15 0 0.5
3 3 16 0 0.5
3 0 17 0 0.5
0
EOF

  cat > data/lang/hmm_nihaowenwen.txt <<EOF
0 0 18 $id_word1 0.5
0 1 19 0 0.5
1 1 20 0 0.5
1 2 21 0 0.5
2 2 22 0 0.5
2 3 23 0 0.5
3 3 24 0 0.5
3 0 25 0 0.5
0
EOF

  echo "Prepare an unnormalized phone language model for the denominator graph"
  cat <<EOF > data/lang/phone_lm.txt
0 1 $id_sil
0 5 $id_sil
1 2 $id_word0
2 3 $id_sil
1 4 $id_word1
4 5 $id_sil
1 6 $id_freetext
6 7 $id_sil
3 2.3
5 2.3
7 0.0
EOF

  echo "Generate graphs for training"
  local/generate_graphs.py --hmm-paths data/lang/hmm_{sil,freetext,hixiaowen,nihaowenwen}.txt \
    --lexicon-fst-path data/lang/L.fst.txt --phone-lm-fsa-path data/lang/phone_lm.txt \
    --out-dir data
fi

[ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu -n $ngpus) || \
  echo "Unable to get $ngpus GPUs"
[ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
[ $(echo $free_gpu | sed 's/,/ /g' | awk '{print NF}') -ne "$ngpus" ] && \
  echo "number of GPU ids in --free-gpu=$free_gpu does not match --ngpus=$ngpus" && exit 1;

num_targets=26 # hard-coded for now. It's equal to the number of different labels in data/lang/hmm_*.txt

if [ ${stage} -le 2 ]; then
  echo "Stage 2: Model Training"
  opts=""
  valid_subset=dev
  mkdir -p $dir/log
  log_file=$dir/log/train.log
  [ -f $dir/checkpoint_last.pt ] && log_file="-a $log_file"
  update_freq=1
  CUDA_VISIBLE_DEVICES=$free_gpu speech_train.py data --task speech_recognition_hybrid --seed 1 \
    --log-interval $((1500/ngpus/update_freq)) --log-format simple \
    --num-workers 0 --data-buffer-size 0 --max-tokens 25600 --batch-size 128 --empty-cache-freq 50 \
    --valid-subset $valid_subset --batch-size-valid 128 --ddp-backend no_c10d --update-freq $update_freq \
    --distributed-world-size $ngpus --arch speech_tdnn_mobvoi \
    --max-epoch 15 --optimizer adam --lr 0.001 --weight-decay 0.0 \
    --lr-scheduler reduce_lr_on_plateau_v2 --lr-shrink 0.5 \
    --save-dir $dir --restore-file checkpoint_last.pt --save-interval-updates $((1500/ngpus/update_freq)) \
    --keep-interval-updates 5 --keep-last-epochs 5 --validate-interval 1 \
    --criterion k2_lattice_free_mmi --num-targets $num_targets --word-symbol-path data/lang/words.txt \
    --denominator-fst-path data/denominator.pt --HCL-fst-path data/HL.pt \
    --max-source-positions 9999 --max-target-positions 9999 $opts 2>&1 | tee $log_file
fi

if [ ${stage} -le 3 ]; then
  echo "Stage 3: Decoding"
fi

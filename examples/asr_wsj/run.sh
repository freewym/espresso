#!/bin/bash

# Copyright (c) 2019-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

set -e -o pipefail

stage=0
free_gpu=
data_dir=data-bin/wsj
exp_dir=exp/wsj/lstm
train_set=train_si284
valid_set=test_dev93
test_set=test_eval92
checkpoint=checkpoint_best.pt


if [ -f ./path.sh ]; then
  . ./path.sh
else
  . ./examples/asr_wsj/path.sh
fi
if [ -f ../../speech_tools/parse_options.sh ]; then
  . ../../speech_tools/parse_options.sh
else
  . ./speech_tools/parse_options.sh
fi

dict=$data_dir/lang/${train_set}_units.txt
nlsyms=$data_dir/lang/non_lang_syms.txt
train_text=$data_dir/$train_set/text
train_token_text=$data_dir/$train_set/token_text
valid_text=$data_dir/$valid_set/text
valid_token_text=$data_dir/$valid_set/token_text
test_text=$data_dir/$test_set/text
test_token_text=$data_dir/$test_set/token_text
if [ ${stage} -le 1 ]; then
  echo "Stage 1: Dictionary Preparation and Text Tokenization"
  mkdir -p $data_dir/lang
  
  echo "Making a non-linguistic symbol list..."
  cut -f 2- $train_text | tr " " "\n" | sort | uniq | grep "<" > $nlsyms
  cat $nlsyms

  echo "Making a dictionary and tokenizing text for training set..."
  python3 speech_tools/text2token.py --skip-ncols 1 --space "<space>" --non-lang-syms $nlsyms \
    $train_text > $train_token_text
  cut -f 2- -d" " $train_token_text | tr " " "\n" | grep -v -e '^\s*$' | sort | \
    uniq -c | awk '{print $2,$1}' > $dict
  wc -l $dict

  echo "Tokenizing text for validation and test set..."
  python3 speech_tools/text2token.py --skip-ncols 1 --space "<space>" --non-lang-syms $nlsyms \
    $valid_text > $valid_token_text
  python3 speech_tools/text2token.py --skip-ncols 1 --space "<space>" --non-lang-syms $nlsyms \
    $test_text > $test_token_text
fi

train_feat=$data_dir/dump/$train_set/deltafalse/feats.scp
valid_feat=$data_dir/dump/$valid_set/deltafalse/feats.scp
if [ ${stage} -le 2 ]; then
  echo "Stage 2: Model Training"
  mkdir -p $exp_dir/logs
  [ -z "$free_gpu" ] && free_gpu=$(free-gpu)
  CUDA_VISIBLE_DEVICES=$free_gpu python3 -u speech_train.py \
    --log-interval 1000 --log-format "simple" --seed 1 \
    --num-workers 0 --max-tokens 45000 --max-sentences 32 --max-sentences-valid 64 \
    --distributed-world-size 1 --distributed-rank 0 --distributed-port -1 \
    --max-epoch 20 --optimizer "adam" --lr 0.1 --weight-decay 0.0 \
    --lr-scheduler "reduce_lr_on_plateau" --lr-shrink 0.1 \
    --save-dir $exp_dir --save-interval-updates 100 --keep-interval-updates 10 \
    --keep-last-epochs 5 --validate-interval 100 \
    --arch "speech_conv_lstm_wsj" --criterion "cross_entropy_with_wer" \
    --train-feat-files $train_feat --train-text-files $train_token_text \
    --valid-feat-files $valid_feat --valid-text-files $valid_token_text \
    --dict $dict --max-source-positions 9999 --max-target-positions 999 2>&1 | tee $exp_dir/logs/train.log
fi
exit 0

test_feat=$data_dir/dump/$test_set/deltafalse/feats.scp
if [ ${stage} -le 2 ]; then
  echo "Stage 3: Decoding"
  [ -z "$free_gpu" ] && free_gpu=$(free-gpu)
  CUDA_VISIBLE_DEVICES=$free_gpu python3 -u speech_recognition.py \
    --max-tokens 45000 --max-sentences 32 --num-shards 1 --shard-id 0 \
    --test-feat-files $test_feat --test-text-files $test_token_text \
    --dict $dict --max-source-positions 9999 --max-target-positions 999 \
    --path $exp_dir/$checkpoint --beam 10 --max-len-a 0.5 --max-len-b 0 \
    --lenpen 1.0 --print-alignment 2>&1 | tee $exp_dir/logs/decode.log
fi

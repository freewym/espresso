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
validate_on_train=false


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

valid_subset=valid
if $validate_on_train; then
  valid_subset="$valid_subset train"
fi

dict=$data_dir/lang/${train_set}_units.txt
nlsyms=$data_dir/lang/non_lang_syms.txt
train_text=$data_dir/$train_set/text
if [ ${stage} -le 1 ]; then
  echo "Stage 1: Dictionary Preparation and Text Tokenization"
  mkdir -p $data_dir/lang
  
  echo "Making a non-linguistic symbol list..."
  cut -f 2- $train_text | tr " " "\n" | sort | uniq | grep "<" > $nlsyms
  cat $nlsyms

  echo "Making a dictionary and tokenizing text for train/valid/test set..."
  for dataset in $train_set $valid_set $test_set; do
    text=$data_dir/$dataset/text
    token_text=$data_dir/$dataset/token_text
    python3 speech_tools/text2token.py --skip-ncols 1 --space "<space>" \
      --non-lang-syms $nlsyms $text > $token_text
    if [ "$dataset" == "$train_set" ]; then
      cut -f 2- -d" " $token_text | tr " " "\n" | grep -v -e '^\s*$' | sort | \
        uniq -c | awk '{print $2,$1}' > $dict
      wc -l $dict
    fi
  done
fi

train_feat=$data_dir/dump/$train_set/deltafalse/feats.scp
train_token_text=$data_dir/$train_set/token_text
valid_feat=$data_dir/dump/$valid_set/deltafalse/feats.scp
valid_token_text=$data_dir/$valid_set/token_text
if [ ${stage} -le 2 ]; then
  echo "Stage 2: Model Training"
  mkdir -p $exp_dir/logs
  log_file=$exp_dir/logs/train.log
  [ -f $exp_dir/checkpoint_last.pt ] && log_file="-a $log_file"
  opts=""
  [ -f examples/asr_wsj/wer_output_filter ] && \
    opts="$opts --wer-output-filter examples/asr_wsj/wer_output_filter"
  [ -z "$free_gpu" ] && free_gpu=$(free-gpu)
  CUDA_VISIBLE_DEVICES=$free_gpu python3 -u speech_train.py --seed 1 \
    --log-interval 500 --log-format "simple" --print-training-sample-interval 500 \
    --num-workers 0 --max-tokens 24000 --max-sentences 32 \
    --valid-subset $valid_subset --max-sentences-valid 64 \
    --distributed-world-size 1 --distributed-rank 0 --distributed-port -1 \
    --max-epoch 20 --optimizer "adam" --lr 0.001 --weight-decay 0.0 \
    --lr-scheduler "reduce_lr_on_plateau" --lr-shrink 0.1 --min-lr "1e-15" \
    --save-dir $exp_dir --save-interval-updates 200 --keep-interval-updates 10 \
    --keep-last-epochs 5 --validate-interval 1 \
    --arch "speech_conv_lstm_wsj" --criterion "cross_entropy_with_wer" \
    --train-feat-files $train_feat --train-text-files $train_token_text \
    --valid-feat-files $valid_feat --valid-text-files $valid_token_text \
    --dict $dict --non-lang-syms $nlsyms \
    --max-source-positions 9999 --max-target-positions 999 $opts 2>&1 | tee $log_file
exit 0
fi

if [ ${stage} -le 3 ]; then
  echo "Stage 3: Decoding"
  opts=""
  [ -f examples/asr_wsj/wer_output_filter ] && \
    opts="$opts --wer-output-filter examples/asr_wsj/wer_output_filter"
  [ -z "$free_gpu" ] && free_gpu=$(free-gpu)
  for dataset in $valid_set $test_set; do
    feat=$data_dir/dump/$dataset/deltafalse/feats.scp
    text=$data_dir/$dataset/token_text
    CUDA_VISIBLE_DEVICES=$free_gpu python3 -u speech_recognition.py \
      --max-tokens 45000 --max-sentences 32 --num-shards 1 --shard-id 0 \
      --test-feat-files $feat --test-text-files $text \
      --dict $dict --non-lang-syms $nlsyms \
      --max-source-positions 9999 --max-target-positions 999 \
      --path $exp_dir/$checkpoint --beam 10 --max-len-a 0.5 --max-len-b 0 \
      --lenpen 1.0 --output-dir $exp_dir/decode_$dataset --print-alignment $opts \
      2>&1 | tee $exp_dir/logs/decode_$dataset.log
  done
fi

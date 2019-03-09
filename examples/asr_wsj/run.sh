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

# e2e model related
affix=
train_set=train_si284
valid_set=test_dev93
test_set=test_eval92
checkpoint=checkpoint_best.pt
validate_on_train=false
disable_cudnn=false

# LM related
lm_affix=
lm_checkpoint=checkpoint_best.pt
lm_shallow_fusion=false

# data related
dumpdir=data/dump   # directory to dump full features
wsj0=
wsj1=
if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  wsj0=/export/corpora5/LDC/LDC93S6B
  wsj1=/export/corpora5/LDC/LDC94S13B
fi
# feature configuration
do_delta=false


. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

lmdir=exp/lm_lstm${lm_affix:+_${lm_affix}}
dir=exp/lstm${affix:+_$affix}

if [ ${stage} -le 0 ]; then
  ### Task dependent. You have to make data the following preparation part by yourself.
  ### But you can utilize Kaldi recipes in most cases
  echo "Stage 0: Data Preparation"
  local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
  local/wsj_format_data.sh
fi

train_feat_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${train_feat_dir}
valid_feat_dir=${dumpdir}/${valid_set}/delta${do_delta}; mkdir -p ${valid_feat_dir}
test_feat_dir=${dumpdir}/${test_set}/delta${do_delta}; mkdir -p ${test_feat_dir}
if [ ${stage} -le 1 ]; then
  ### Task dependent. You have to design training and dev sets by yourself.
  ### But you can utilize Kaldi recipes in most cases
  echo "Stage 1: Feature Generation"
  fbankdir=fbank
  # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
  for x in $train_set $valid_set $test_set; do
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
      data/${x} exp/make_fbank/${x} ${fbankdir}
  done

  # compute global CMVN
  compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

  # dump features for training
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d ${train_feat_dir}/storage ]; then
    utils/create_split_dir.pl \
      /export/b{10,11,12,13}/${USER}/fairseq-data/egs/asr_wsj/dump/${train_set}/delta${do_delta}/storage \
      ${train_feat_dir}/storage
  fi
  dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
    data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${train_feat_dir}
  dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
    data/${valid_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/valid ${valid_feat_dir}
  dump.sh --cmd "$train_cmd" --nj 4 --do_delta $do_delta \
    data/${test_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/test ${test_feat_dir}
fi

dict=data/lang/${train_set}_units.txt
nlsyms=data/lang/non_lang_syms.txt
lmdatadir=data/lm_text
if [ ${stage} -le 2 ]; then
  echo "Stage 2: Dictionary Preparation and Text Tokenization"
  mkdir -p data/lang

  echo "$0: making a non-linguistic symbol list..."
  train_text=data/$train_set/text
  cut -f 2- $train_text | tr " " "\n" | sort | uniq | grep "<" > $nlsyms
  cat $nlsyms

  echo "$0: making a dictionary and tokenizing text for train/valid/test set..."
  for dataset in $train_set $valid_set $test_set; do
    text=data/$dataset/text
    token_text=data/$dataset/token_text
    text2token.py --skip-ncols 1 --space "<space>" --non-lang-syms $nlsyms $text > $token_text
    if [ "$dataset" == "$train_set" ]; then
      cut -f 2- -d" " $token_text | tr " " "\n" | grep -v -e '^\s*$' | sort | \
        uniq -c | awk '{print $2,$1}' > $dict
      wc -l $dict
    fi
  done

  echo "$0: preparing text for LM..."
  mkdir -p $lmdatadir
  for dataset in $train_set $valid_set $test_set; do
    token_text=data/$dataset/token_text
    cut -f 2- -d" " $token_text > $lmdatadir/$dataset.tokens
  done
  zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
    | grep -v "<" | tr "[:lower:]" "[:upper:]" \
    | text2token.py --space "<space>" > $lmdatadir/train_others.tokens
  cat $lmdatadir/$train_set.tokens $lmdatadir/train_others.tokens > $lmdatadir/train.tokens
fi

lmdict=$dict
if [ ${stage} -le 3 ]; then
  echo "Stage 3: Text Binarization for LM Training"
  lmdict=$dict
  mkdir -p $lmdatadir/logs
  ${decode_cmd} $lmdatadir/logs/preprocess.log \
    python3 ../../preprocess.py --task language_modeling_for_asr \
      --workers 30 --srcdict $lmdict --only-source \
      --trainpref $lmdatadir/train.tokens \
      --validpref $lmdatadir/$valid_set.tokens \
      --testpref $lmdatadir/$test_set.tokens \
      --destdir $lmdatadir
fi

if [ ${stage} -le 4 ]; then
  echo "Stage 4: LM Training"
  valid_subset=valid
  mkdir -p $lmdir/logs
  log_file=$lmdir/logs/train.log
  [ -f $lmdir/checkpoint_last.pt ] && log_file="-a $log_file"
  [ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu)
  [ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
  CUDA_VISIBLE_DEVICES=$free_gpu python3 ../../train.py $lmdatadir --seed 1 \
    --task language_modeling_for_asr --dict $lmdict \
    --log-interval 2000 --log-format simple \
    --num-workers 0 --max-tokens 25600 --max-sentences 128 \
    --valid-subset $valid_subset --max-sentences-valid 256 \
    --distributed-world-size 1 --distributed-rank 0 --distributed-port -1 \
    --max-epoch 25 --optimizer adam --lr 0.001 --weight-decay 5e-06 \
    --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
    --save-dir $lmdir --restore-file checkpoint_last.pt --save-interval-updates 2000 \
    --keep-interval-updates 5 --keep-last-epochs 5 --validate-interval 1 \
    --arch lstm_lm_wsj --dropout 0.1 --criterion cross_entropy \
    --sample-break-mode eos 2>&1 | tee $log_file
fi

if [ ${stage} -le 5 ]; then
  echo "Stage 5: LM Evaluation"
  for gen_subset in valid test; do
    log_file=$lmdir/logs/evaluation_$gen_subset.log
    [ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu)
    [ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
    CUDA_VISIBLE_DEVICES=$free_gpu python3 ../../eval_lm.py $lmdatadir \
      --task language_modeling_for_asr --dict $lmdict --gen-subset $gen_subset \
      --max-tokens 192000 --max-sentences 256 --sample-break-mode eos \
      --path $lmdir/$lm_checkpoint 2>&1 | tee $log_file
  done
fi

train_feat=$train_feat_dir/feats.scp
train_token_text=data/$train_set/token_text
valid_feat=$valid_feat_dir/feats.scp
valid_token_text=data/$valid_set/token_text
if [ ${stage} -le 6 ]; then
  echo "Stage 6: Model Training"
  valid_subset=valid
  if $validate_on_train; then
    valid_subset="$valid_subset,train"
  fi
  mkdir -p $dir/logs
  log_file=$dir/logs/train.log
  [ -f $dir/checkpoint_last.pt ] && log_file="-a $log_file"
  opts=""
  $disable_cudnn && opts="$opts --disable-cudnn"
  [ -f local/wer_output_filter ] && \
    opts="$opts --wer-output-filter local/wer_output_filter"
  [ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu)
  [ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
  CUDA_VISIBLE_DEVICES=$free_gpu speech_train.py --seed 1 \
    --log-interval 500 --log-format simple --print-training-sample-interval 1000 \
    --num-workers 0 --max-tokens 24000 --max-sentences 32 \
    --valid-subset $valid_subset --max-sentences-valid 64 \
    --distributed-world-size 1 --distributed-rank 0 --distributed-port -1 \
    --max-epoch 20 --optimizer adam --lr 0.001 --weight-decay 0.0 \
    --lr-scheduler reduce_lr_on_plateau_v2 --lr-shrink 0.5 --min-lr 1e-4 --start-reduce-lr-epoch 11 \
    --save-dir $dir --restore-file checkpoint_last.pt --save-interval-updates 200 \
    --keep-interval-updates 5 --keep-last-epochs 5 --validate-interval 1 \
    --arch speech_conv_lstm_wsj --criterion label_smoothed_cross_entropy_with_wer --label-smoothing 0.05 \
    --train-feat-files $train_feat --train-text-files $train_token_text \
    --valid-feat-files $valid_feat --valid-text-files $valid_token_text \
    --dict $dict --non-lang-syms $nlsyms --dropout 0.2 \
    --max-source-positions 9999 --max-target-positions 999 $opts 2>&1 | tee $log_file
fi

if [ ${stage} -le 7 ]; then
  echo "Stage 7: Decoding"
  opts=""
  path=$dir/$checkpoint
  decode_affix=
  if $lm_shallow_fusion; then
    path="$path:$lmdir/$lm_checkpoint"
    opts="$opts --lprob-weights 1,0.29"
    decode_affix=shallow_fusion
  fi
  [ -f local/wer_output_filter ] && \
    opts="$opts --wer-output-filter local/wer_output_filter"
  for dataset in $valid_set $test_set; do
    if [ "$dataset" == "$valid_set" ]; then
      feat=$valid_feat_dir/feats.scp
    elif [ "$dataset" == "$test_set" ]; then
      feat=$test_feat_dir/feats.scp
    fi
    text=data/$dataset/token_text
    [ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu)
    [ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
    CUDA_VISIBLE_DEVICES=$free_gpu speech_recognize.py \
      --max-tokens 45000 --max-sentences 1 --num-shards 1 --shard-id 0 \
      --test-feat-files $feat --test-text-files $text \
      --dict $dict --non-lang-syms $nlsyms \
      --max-source-positions 9999 --max-target-positions 999 \
      --path $path --beam 50 --max-len-a 0.5 --max-len-b 0 --lenpen 1.1 --no-early-stop \
      --results-path $dir/decode_$dataset${decode_affix:+_${decode_affix}} $opts \
      --print-alignment 2>&1 | tee $dir/logs/decode_$dataset${decode_affix:+_${decode_affix}}.log
  done
fi

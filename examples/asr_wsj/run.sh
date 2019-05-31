#!/bin/bash

# Copyright (c) 2019-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

set -e -o pipefail

stage=0
ngpus=1 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid

# E2E model related
affix=
train_set=train_si284
valid_set=test_dev93
test_set=test_eval92
checkpoint=checkpoint_best.pt
validate_on_train_subset=false # for monitoring E2E model training

# LM related
lm_affix=
lm_checkpoint=checkpoint_best.pt
lm_shallow_fusion=true # no LM fusion if false
use_wordlm=true # Only relevant when LM fusion is enabled. Use char LM if false
wordlm_affix=
wordlm_vocabsize=65000

# data related
dumpdir=data/dump   # directory to dump full features
wsj0=
wsj1=
if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  wsj0=/export/corpora5/LDC/LDC93S6B
  wsj1=/export/corpora5/LDC/LDC94S13B
fi
train_subset_size=500 # for validation if validate_on_train_subset is set to true
kaldi_scoring=true

# feature configuration
do_delta=false


. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

lmdir=exp/lm_lstm${lm_affix:+_${lm_affix}}
wordlmdir=exp/wordlm_lstm${wordlm_affix:+_${wordlm_affix}}
dir=exp/lstm${affix:+_$affix}

if [ ${stage} -le 0 ]; then
  echo "Stage 0: Data Preparation"
  local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
  echo "Preparing train and test data"
  srcdir=data/local/data
  for x in train_si284 test_eval92 test_eval93 test_dev93 test_eval92_5k test_eval93_5k test_dev93_5k dev_dt_05 dev_dt_20; do
    mkdir -p data/$x
    cp $srcdir/${x}_wav.scp data/$x/wav.scp || exit 1;
    cp $srcdir/$x.txt data/$x/text || exit 1;
    cp $srcdir/$x.spk2utt data/$x/spk2utt || exit 1;
    cp $srcdir/$x.utt2spk data/$x/utt2spk || exit 1;
    utils/filter_scp.pl data/$x/spk2utt $srcdir/spk2gender > data/$x/spk2gender || exit 1;
  done
  echo "Succeeded in formatting data."
fi

train_feat_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${train_feat_dir}
train_subset_feat_dir=${dumpdir}/${train_set}_${train_subset_size}/delta${do_delta}; mkdir -p ${train_subset_feat_dir}
valid_feat_dir=${dumpdir}/${valid_set}/delta${do_delta}; mkdir -p ${valid_feat_dir}
test_feat_dir=${dumpdir}/${test_set}/delta${do_delta}; mkdir -p ${test_feat_dir}
if [ ${stage} -le 1 ]; then
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

  # randomly select a subset of train set for optional diagnosis
  utils/subset_data_dir.sh data/${train_set} ${train_subset_size} data/${train_set}_${train_subset_size}
  utils/filter_scp.pl data/${train_set}_${train_subset_size}/utt2spk ${train_feat_dir}/feats.scp \
    > ${train_subset_feat_dir}/feats.scp
fi

dict=data/lang/${train_set}_units.txt
nlsyms=data/lang/non_lang_syms.txt
lmdatadir=data/lm_text
wordlmdict=data/lang/wordlist_$wordlm_vocabsize.txt
wordlmdatadir=data/wordlm_text
if [ ${stage} -le 2 ]; then
  echo "Stage 2: Dictionary Preparation and Text Tokenization"
  mkdir -p data/lang

  echo "$0: making a non-linguistic symbol list..."
  train_text=data/$train_set/text
  cut -f 2- $train_text | tr " " "\n" | sort | uniq | grep "<" > $nlsyms
  cat $nlsyms

  echo "$0: making a dictionary and tokenizing text for train/valid/test set..."
  for dataset in $train_set ${train_set}_${train_subset_size} $valid_set $test_set; do
    text=data/$dataset/text
    token_text=data/$dataset/token_text
    text2token.py --skip-ncols 1 --space "<space>" --non-lang-syms $nlsyms $text > $token_text
    if [ "$dataset" == "$train_set" ]; then
      cut -f 2- -d" " $token_text | tr " " "\n" | grep -v -e '^\s*$' | sort | \
        uniq -c | awk '{print $2,$1}' > $dict
      wc -l $dict
    fi
  done

  if ! $use_wordlm; then
    echo "$0: preparing text for char LM..."
    mkdir -p $lmdatadir
    for dataset in $train_set $valid_set $test_set; do
      token_text=data/$dataset/token_text
      cut -f 2- -d" " $token_text > $lmdatadir/$dataset.tokens
    done
    zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
      | grep -v "<" | tr "[:lower:]" "[:upper:]" \
      | text2token.py --space "<space>" > $lmdatadir/train_others.tokens
    cat $lmdatadir/$train_set.tokens $lmdatadir/train_others.tokens > $lmdatadir/train.tokens
  else
    echo "$0: preparing text and making word dictionary for word LM..."
    mkdir -p $wordlmdatadir
    for dataset in $train_set $valid_set $test_set; do
      text=data/$dataset/text
      cut -f 2- -d" " $text > $wordlmdatadir/$dataset
    done
    zcat ${wsj1}/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
      | grep -v "<" | tr "[:lower:]" "[:upper:]" > $wordlmdatadir/train_others
    cat $wordlmdatadir/$train_set $wordlmdatadir/train_others > $wordlmdatadir/train
    text2vocabulary.py --vocabsize $wordlm_vocabsize --exclude "<eos> <unk>" \
      --valid-text $wordlmdatadir/$valid_set --test-text $wordlmdatadir/$test_set \
      $wordlmdatadir/train > $wordlmdict \
      2> >(tee $(dirname $wordlmdict)/vocab${wordlm_vocabsize}_stats.log >&2)
  fi
fi

lmdict=$dict
if [ ${stage} -le 3 ]; then
  echo "Stage 3: Text Binarization for LM Training"
  if ! $use_wordlm; then
    echo "$0: binarizing char text..."
    mkdir -p $lmdatadir/logs
    ${decode_cmd} $lmdatadir/logs/preprocess.log \
      python3 ../../preprocess.py --task language_modeling_for_asr \
        --workers 30 --srcdict $lmdict --only-source \
        --trainpref $lmdatadir/train.tokens \
        --validpref $lmdatadir/$valid_set.tokens \
        --testpref $lmdatadir/$test_set.tokens \
        --destdir $lmdatadir
  else
    echo "$0: binarizing word text..."
    mkdir -p $wordlmdatadir/logs
    ${decode_cmd} $wordlmdatadir/logs/preprocess.log \
      python3 ../../preprocess.py --task language_modeling_for_asr \
        --workers 30 --srcdict $wordlmdict --only-source \
        --trainpref $wordlmdatadir/train \
        --validpref $wordlmdatadir/$valid_set \
        --testpref $wordlmdatadir/$test_set \
        --destdir $wordlmdatadir
  fi
fi

[ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu -n $ngpus) || \
  echo "Unable to get $ngpus GPUs"
[ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
[ $(echo $free_gpu | sed 's/,/ /g' | awk '{print NF}') -ne "$ngpus" ] && \
  echo "number of GPU ids in --free-gpu=$free_gpu does not match --ngpus=$ngpus" && exit 1;

if [ ${stage} -le 4 ] && ! $use_wordlm; then
  echo "Stage 4: char LM Training"
  valid_subset=valid
  mkdir -p $lmdir/logs
  log_file=$lmdir/logs/train.log
  [ -f $lmdir/checkpoint_last.pt ] && log_file="-a $log_file"
  CUDA_VISIBLE_DEVICES=$free_gpu python3 ../../train.py $lmdatadir --seed 1 \
    --task language_modeling_for_asr --dict $lmdict \
    --log-interval 2000 --log-format simple \
    --num-workers 0 --max-tokens 25600 --max-sentences 128 \
    --valid-subset $valid_subset --max-sentences-valid 256 \
    --distributed-world-size $ngpus --distributed-rank 0 --distributed-port 100 \
    --max-epoch 25 --optimizer adam --lr 0.001 --weight-decay 5e-06 \
    --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
    --save-dir $lmdir --restore-file checkpoint_last.pt --save-interval-updates 2000 \
    --keep-interval-updates 5 --keep-last-epochs 5 --validate-interval 1 \
    --arch lstm_lm_wsj --criterion cross_entropy --sample-break-mode eos 2>&1 | tee $log_file
fi

if [ ${stage} -le 5 ] && ! $use_wordlm; then
  echo "Stage 5: char LM Evaluation"
  for gen_subset in valid test; do
    log_file=$lmdir/logs/evaluation_$gen_subset.log
    python3 ../../eval_lm.py $lmdatadir --cpu \
      --task language_modeling_for_asr --dict $lmdict --gen-subset $gen_subset \
      --max-tokens 192000 --max-sentences 256 --sample-break-mode eos \
      --path $lmdir/$lm_checkpoint 2>&1 | tee $log_file
  done
fi

if [ ${stage} -le 6 ] && $use_wordlm; then
  echo "Stage 6: word LM Training"
  valid_subset=valid
  mkdir -p $wordlmdir/logs
  log_file=$wordlmdir/logs/train.log
  [ -f $wordlmdir/checkpoint_last.pt ] && log_file="-a $log_file"
  CUDA_VISIBLE_DEVICES=$free_gpu python3 ../../train.py $wordlmdatadir --seed 1 \
    --task language_modeling_for_asr --dict $wordlmdict \
    --log-interval 2000 --log-format simple \
    --num-workers 0 --max-tokens 6400 --max-sentences 256 \
    --valid-subset $valid_subset --max-sentences-valid 512 \
    --distributed-world-size $ngpus --distributed-rank 0 --distributed-port 100 \
    --max-epoch 20 --optimizer adam --lr 0.001 --weight-decay 1e-05 \
    --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
    --save-dir $wordlmdir --restore-file checkpoint_last.pt --save-interval-updates 2000 \
    --keep-interval-updates 5 --keep-last-epochs 5 --validate-interval 1 \
    --arch lstm_wordlm_wsj --criterion cross_entropy \
    --sample-break-mode eos 2>&1 | tee $log_file
fi

if [ ${stage} -le 7 ] && $use_wordlm; then
  echo "Stage 7: word LM Evaluation"
  for gen_subset in valid test; do
    log_file=$wordlmdir/logs/evaluation_$gen_subset.log
    python3 ../../eval_lm.py $wordlmdatadir --cpu \
      --task language_modeling_for_asr --dict $wordlmdict --gen-subset $gen_subset \
      --max-tokens 12800 --max-sentences 512 --sample-break-mode eos \
      --path $wordlmdir/$lm_checkpoint 2>&1 | tee $log_file
  done
fi

train_feat=$train_feat_dir/feats.scp
train_token_text=data/$train_set/token_text
valid_feat=$valid_feat_dir/feats.scp
valid_token_text=data/$valid_set/token_text
if [ ${stage} -le 8 ]; then
  echo "Stage 8: Model Training"
  opts=""
  valid_subset=valid
  if $validate_on_train_subset; then
    valid_subset="$valid_subset,train_subset"
    opts="$opts --train-subset-feat-files $train_subset_feat_dir/feats.scp"
    opts="$opts --train-subset-text-files data/${train_set}_${train_subset_size}/token_text"
  fi
  [ -f local/wer_output_filter ] && opts="$opts --wer-output-filter local/wer_output_filter"
  mkdir -p $dir/logs
  log_file=$dir/logs/train.log
  [ -f $dir/checkpoint_last.pt ] && log_file="-a $log_file"
  CUDA_VISIBLE_DEVICES=$free_gpu speech_train.py --seed 1 \
    --log-interval 400 --log-format simple --print-training-sample-interval 1000 \
    --num-workers 0 --max-tokens 24000 --max-sentences 32 \
    --valid-subset $valid_subset --max-sentences-valid 64 \
    --distributed-world-size $ngpus --distributed-rank 0 --distributed-port 100 \
    --max-epoch 30 --optimizer adam --lr 0.001 --weight-decay 0.0 \
    --lr-scheduler reduce_lr_on_plateau_v2 --lr-shrink 0.5 --min-lr 1e-5 --start-reduce-lr-epoch 11 \
    --save-dir $dir --restore-file checkpoint_last.pt --save-interval-updates 400 \
    --keep-interval-updates 5 --keep-last-epochs 5 --validate-interval 1 \
    --arch speech_conv_lstm_wsj --criterion label_smoothed_cross_entropy_with_wer \
    --label-smoothing 0.05 --smoothing-type temporal \
    --scheduled-sampling-probs 0.4 --start-scheduled-sampling-epoch 11 \
    --train-feat-files $train_feat --train-text-files $train_token_text \
    --valid-feat-files $valid_feat --valid-text-files $valid_token_text \
    --dict $dict --non-lang-syms $nlsyms \
    --max-source-positions 9999 --max-target-positions 999 $opts 2>&1 | tee $log_file
fi

if [ ${stage} -le 9 ]; then
  echo "Stage 9: Decoding"
  opts=""
  path=$dir/$checkpoint
  decode_affix=
  if $lm_shallow_fusion; then
    if ! $use_wordlm; then
      path="$path:$lmdir/$lm_checkpoint"
      opts="$opts --lm-weight 0.7 --coverage-weight 0.01"
      decode_affix=shallow_fusion
    else
      path="$path:$wordlmdir/$lm_checkpoint"
      opts="$opts --word-dict $wordlmdict --lm-weight 0.8 --oov-penalty 5e-7 --coverage-weight 0.01"
      decode_affix=shallow_fusion_wordlm
    fi
  fi
  [ -f local/wer_output_filter ] && opts="$opts --wer-output-filter local/wer_output_filter"
  for dataset in $valid_set $test_set; do
    if [ "$dataset" == "$valid_set" ]; then
      feat=$valid_feat_dir/feats.scp
    elif [ "$dataset" == "$test_set" ]; then
      feat=$test_feat_dir/feats.scp
    fi
    text=data/$dataset/token_text
    CUDA_VISIBLE_DEVICES=$(echo $free_gpu | sed 's/,/ /g' | awk '{print $1}') speech_recognize.py \
      --max-tokens 20000 --max-sentences 32 --num-shards 1 --shard-id 0 \
      --test-feat-files $feat --test-text-files $text \
      --dict $dict --non-lang-syms $nlsyms \
      --max-source-positions 9999 --max-target-positions 999 \
      --path $path --beam 50 --max-len-a 0.2 --max-len-b 0 --lenpen 1.0 \
      --results-path $dir/decode_$dataset${decode_affix:+_${decode_affix}} $opts \
      --print-alignment 2>&1 | tee $dir/logs/decode_$dataset${decode_affix:+_${decode_affix}}.log

    if $kaldi_scoring; then
      echo "verify WER by scoring with Kaldi..."
      local/score.sh data/$dataset $dir/decode_$dataset${decode_affix:+_${decode_affix}}
      cat $dir/decode_$dataset${decode_affix:+_${decode_affix}}/scoring_kaldi/wer
    fi
  done
fi

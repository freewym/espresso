#!/bin/bash

# Copyright (c) 2019-present, Hang Lyu
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

set -e -o pipefail

stage=0
ngpus=2 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid

# E2E model related
affix=
train_set=train_nodup
valid_set=train_dev
test_sets="train_dev eval2000 rt03"
checkpoint=checkpoint_best.pt

validate_on_train_subset=false # for monitoring E2E model training

# LM related
lm_affix=
lm_checkpoint=checkpoint_best.pt
lm_shallow_fusion=true # no LM fusion if false
sentencepiece_vocabsize=1000
sentencepiece_type=unigram

# data related
dumpdir=data/dump   # directory to dump full features
swbd1_dir=
eval2000_dir=
rt03_dir=
fisher_dirs=

if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  swbd1_dir=/export/corpora3/LDC/LDC97S62
  eval2000_dir="/export/corpora2/LDC/LDC2002S09/hub5e_00 /export/corpora2/LDC/LDC2002T43"
  rt03_dir=/export/corpora/LDC/LDC2007S10
  fisher_dirs="/export/corpora3/LDC/LDC2004T19/fe_03_p1_tran/ /export/corpora3/LDC/LDC2005T19/fe_03_p2_tran/"
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

if [ $stage -le 0 ]; then
  echo "Stage 0: Data Preparation"
  local/swbd1_data_download.sh ${swbd1_dir}
  local/swbd1_prepare_dict.sh
  local/swbd1_data_prep.sh ${swbd1_dir}
  local/eval2000_data_prep.sh ${eval2000_dir}
  local/rt03_data_prep.sh ${rt03_dir}
  # upsample audio from 8k to 16k to compare with espnet. (It may affact the
  # performance
  for x in train eval2000 rt03; do
	sed -i.bak -e "s/$/ sox -R -t wav - -t wav - rate 16000 dither | /" data/${x}/wav.scp
  done
  # normalize eval2000 ant rt03 texts by
  # 1) convert upper to lower
  # 2) remove tags (%AH) (%HESITATION) (%UH)
  # 3) remove <B_ASIDE> <E_ASIDE>
  # 4) remove "(" or ")"
  for x in eval2000 rt03; do
    cp data/${x}/text data/${x}/text.org
    paste -d "" \
      <(cut -f 1 -d" " data/${x}/text.org) \
      <(awk '{$1=""; print tolower($0)}' data/${x}/text.org | perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g") \
      | sed -e 's/\s\+/ /g' > data/${x}/text
    # rm data/${x}/text.org
  done
  echo "Succeeded in formatting data."
fi


train_feat_dir=$dumpdir/$train_set/delta${do_delta}; mkdir -p $train_feat_dir
valid_feat_dir=$dumpdir/$valid_set/delta${do_delta}; mkdir -p $valid_feat_dir
if [ $stage -le 1 ]; then
  echo "Stage 1: Feature Generation"
  fbankdir=fbank
  # Generate the fbank features; by default 80-dimensional fbanks with pitch on
  # each frame
  for x in train eval2000 rt03; do
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write-utt2num-frames true \
      data/$x exp/make_fbank/$x $fbankdir
    utils/fix_data_dir.sh data/$x
  done

  utils/subset_data_dir.sh --first data/train 4000 data/train_dev  # 5hr 6min
  n=$[`cat data/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last data/train $n data/train_nodev
  utils/data/remove_dup_utts.sh 300 data/train_nodev data/train_nodup  # 286hr

  # compute global CMVN
  compute-cmvn-stats scp:data/$train_set/feats.scp data/$train_set/cmvn.ark

  # dump features for training
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $train_feat_dir/storage ]; then
  utils/create_split_dir.pl \
    /export/b{14,15,16,17}/$USER/espnet-data/egs/swbd/asr1/dump/$train_set/delta${do_delta}/storage \
    $train_feat_dir/storage
  fi
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $valid_feat_dir/storage ]; then
  utils/create_split_dir.pl \
    /export/b{14,15,16,17}/$USER/espnet-data/egs/swbd/asr1/dump/$valid_set/delta${do_delta}/storage \
    $valid_feat_dir/storage
  fi
  dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
    data/$train_set/feats.scp data/$train_set/cmvn.ark exp/dump_feats/train $train_feat_dir
  dump.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta \
    data/$valid_set/feats.scp data/$train_set/cmvn.ark exp/dump_feats/dev $valid_feat_dir
  for rtask in $test_sets; do
    test_feat_dir=$dumpdir/$rtask/delta${do_delta}; mkdir -p $test_feat_dir
    dump.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta \
      data/$rtask/feats.scp data/$train_set/cmvn.ark exp/dump_feats/recog/$rtask \
      $test_feat_dir
  done
  echo "Succeeded in generating features for train_nodup, train_dev, eval2000 and rt03"
fi


dict=data/lang/${train_set}_${sentencepiece_type}${sentencepiece_vocabsize}_units.txt
sentencepiece_model=data/lang/${train_set}_${sentencepiece_type}${sentencepiece_vocabsize}
nlsyms=data/lang/non_lang_syms.txt
lmdatadir=data/lm_text
if [ $stage -le 2 ]; then
  echo "Stage 2: Dictionary Preparation and Text Tokenization"
  mkdir -p data/lang
  mkdir -p $lmdatadir

  echo "Making a non-linguistic symbol list..."
  train_text=data/$train_set/text
  cut -f 2- $train_text | tr " " "\n" | sort | uniq | grep "\[" > $nlsyms
  cat $nlsyms

  echo "Preparing extra corpus for subword LM training..."
  if [ -f $lmdatadir/fisher_text0 ]; then
    rm -rf $lmdatadir/fisher_text0
  fi
  for x in $fisher_dirs; do
    [ ! -d $x/data/trans ] \
      && "Cannot find transcripts in Fisher directory $x" && exit 1;
    cat $x/data/trans/*/*.txt | \
      grep -v ^# | grep -v ^$ | cut -d' ' -f4- >> $lmdatadir/fisher_text0
  done
  cat $lmdatadir/fisher_text0 | local/fisher_map_words.pl | \
    sed 's/^[ \t]*//'> $lmdatadir/fisher_text

  echo "Training sentencepiece model..."
  cut -f 2- -d" " data/$train_set/text | \
    cat - $lmdatadir/fisher_text > data/lang/input
  spm_train --bos_id=-1 --pad_id=0 --eos_id=1 --unk_id=2 --input=data/lang/input \
    --vocab_size=$((sentencepiece_vocabsize+3)) --character_coverage=1.0 \
    --model_type=$sentencepiece_type --model_prefix=$sentencepiece_model \
    --input_sentence_size=10000000 \
    --user_defined_symbols=$(cut -f 2- $train_text | tr " " "\n" | sort | uniq | grep "\[" | tr "\n" "," | sed 's/,$//')

  echo "Making a dictionary and tokenizing text for train/valid/test sets..."
  for dataset in $train_set $test_sets; do  # validation is included in tests
    text=data/$dataset/text
    token_text=data/$dataset/token_text
    spm_encode --model=${sentencepiece_model}.model --output_format=piece \
      <(cut -f 2- -d' ' $text) | paste -d" " <(cut -f 1 -d' ' $text) - > $token_text
    # prepare dict with train_set
    if [ "$dataset" == "$train_set" ]; then
      cut -f 2- -d" " $token_text | tr " " "\n" | grep -v -e '^\s*$' | sort | \
        uniq -c | awk '{print $2,$1}' > $dict
      wc -l $dict
    fi
  done

  echo "Preparing text for subword LM..."
  mkdir -p $lmdatadir
  for dataset in $train_set $test_sets; do
    token_text=data/$dataset/token_text
    cut -f 2- -d" " $token_text > $lmdatadir/$dataset.tokens
  done

  echo "Preparing extra corpus for subword LM training..." 
  cat $lmdatadir/fisher_text |\
    spm_encode --model=${sentencepiece_model}.model --output_format=piece |\
    cat $lmdatadir/$train_set.tokens - > $lmdatadir/train.tokens
fi


lmdict=$dict
if [ $stage -le 3 ]; then
  echo "Stage 3: Text Binarization for subword LM Training"
  mkdir -p $lmdatadir/logs
  for dataset in $test_sets; do test_paths="$test_paths $lmdatadir/$dataset.tokens"; done
  test_paths=$(echo $test_paths | awk '{$1=$1;print}' | tr ' ' ',')
  ${decode_cmd} $lmdatadir/logs/preprocess.log \
    python3 ../../preprocess.py --task language_modeling_for_asr \
      --workers 50 --srcdict $lmdict --only-source \
      --trainpref $lmdatadir/train.tokens \
      --validpref $lmdatadir/$valid_set.tokens \
      --testpref $test_paths \
      --destdir $lmdatadir   
fi


[ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu -n $ngpus) || \
  echo "Unable to get $ngpus GPUs"
[ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
[ $(echo $free_gpu | sed 's/,/ /g' | awk '{print NF}') -ne "$ngpus" ] && \
  echo "number of GPU ids in --free-gpu=$free_gpu does not match --ngpus=$ngpus" && exit 1;


if [ $stage -le 4 ]; then
  echo "Stage 4: subword LM Training"
  valid_subset=valid
  mkdir -p $lmdir/logs
  log_file=$lmdir/logs/train.log
  [ -f $lmdir/checkpoint_last.pt ] && log_file="-a $log_file"
  CUDA_VISIBLE_DEVICES=$free_gpu python3 ../../train.py $lmdatadir --seed 1 \
    --task language_modeling_for_asr --dict $lmdict \
    --log-interval 500 --log-format simple \
    --num-workers 0 --max-tokens 30720 --max-sentences 1024 \
    --valid-subset $valid_subset --max-sentences-valid 1536 \
    --distributed-world-size $ngpus --distributed-rank 0 --distributed-port 100 \
    --max-epoch 25 --optimizer adam --lr 0.001 --clip-norm 1.0 \
    --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
    --save-dir $lmdir --restore-file checkpoint_last.pt --save-interval-updates 500 \
    --keep-interval-updates 3 --keep-last-epochs 5 --validate-interval 1 \
    --arch lstm_lm_librispeech --criterion cross_entropy --sample-break-mode eos 2>&1 | tee $log_file
fi


if [ $stage -le 5 ]; then
  echo "Stage 5: subword LM Evaluation"
  gen_set_array=(test)
  num=$(echo $test_sets | awk '{print NF-1}')
  for i in $(seq $num); do gen_set_array[$i]="test$i"; done  #gen_set_array=(test test1 test2)
  test_set_array=($test_sets)
  for i in $(seq 0 $num); do
    log_file=$lmdir/logs/evaluation_${test_set_array[$i]}.log
    python3 ../../eval_lm.py $lmdatadir --cpu \
      --task language_modeling_for_asr --dict $lmdict --gen-subset ${gen_set_array[$i]} \
      --max-tokens 40960 --max-sentences 1536 --sample-break-mode eos \
      --path $lmdir/$lm_checkpoint 2>&1 | tee $log_file
  done
fi


train_feat=$train_feat_dir/feats.scp
train_token_text=data/$train_set/token_text
valid_feat=$valid_feat_dir/feats.scp
valid_token_text=data/$valid_set/token_text
if [ $stage -le 6 ]; then
  echo "Stage 6: Model Training"
  valid_subset=valid
  opts=""
  [ -f local/wer_output_filter ] && opts="$opts --wer-output-filter local/wer_output_filter"
  mkdir -p $dir/logs
  log_file=$dir/logs/train.log
  [ -f $dir/checkpoint_last.pt ] && log_file="-a $log_file"
  CUDA_VISIBLE_DEVICES=$free_gpu speech_train.py --seed 1 \
    --log-interval 1500 --log-format simple --print-training-sample-interval 2000 --ddp-backend "no_c10d" \
    --num-workers 0 --max-tokens 26000 --max-sentences 24 \
    --valid-subset $valid_subset --max-sentences-valid 48 \
    --distributed-world-size $ngpus --distributed-rank 0 --distributed-port 100 \
    --max-epoch 25 --optimizer adam --lr 0.001 --weight-decay 0.0 --clip-norm 2.0 \
    --lr-scheduler reduce_lr_on_plateau_v2 --lr-shrink 0.5 --min-lr 1e-5 --start-reduce-lr-epoch 10 \
    --save-dir $dir --restore-file checkpoint_last.pt --save-interval-updates 1500 \
    --keep-interval-updates 3 --keep-last-epochs 5 --validate-interval 1 --best-checkpoint-metric wer \
    --arch speech_conv_lstm_librispeech --criterion label_smoothed_cross_entropy_with_wer \
    --label-smoothing 0.1 --smoothing-type uniform \
    --scheduled-sampling-probs 1.0 --start-scheduled-sampling-epoch 1 \
    --train-feat-files $train_feat --train-text-files $train_token_text \
    --valid-feat-files $valid_feat --valid-text-files $valid_token_text \
    --dict $dict --remove-bpe sentencepiece --non-lang-syms $nlsyms \
    --max-source-positions 9999 --max-target-positions 999 $opts 2>&1 | tee $log_file
fi


if [ $stage -le 7 ]; then
  echo "Stage 7: Decoding"
  opts=""
  path=$dir/$checkpoint
  decode_affix=
  if $lm_shallow_fusion; then
    path="$path:$lmdir/$lm_checkpoint"
    opts="$opts --lm-weight 0.3 --coverage-weight 0.0"
    decode_affix=shallow_fusion
  fi
  [ -f local/wer_output_filter ] && opts="$opts --wer-output-filter local/wer_output_filter"
  for dataset in $test_sets; do
    feat=${dumpdir}/$dataset/delta${do_delta}/feats.scp
    text=data/$dataset/token_text
    CUDA_VISIBLE_DEVICES=$(echo $free_gpu | sed 's/,/ /g' | awk '{print $1}') speech_recognize.py \
      --max-tokens 16000 --max-sentences 24 --num-shards 1 --shard-id 0 \
      --test-feat-files $feat --test-text-files $text \
      --dict $dict --remove-bpe sentencepiece --non-lang-syms $nlsyms \
      --max-source-positions 9999 --max-target-positions 999 \
      --path $path --beam 30 --max-len-a 0.08 --max-len-b 0 --lenpen 1.0 \
      --results-path $dir/decode_$dataset${decode_affix:+_${decode_affix}} $opts \
      2>&1 | tee $dir/logs/decode_$dataset${decode_affix:+_${decode_affix}}.log
  done
  if $kaldi_scoring; then
    echo "verify WER by scoring with kaldi..."
    # The word level results are stored in decode_dir/decoded_results.txt
    local/score_sclite.sh data/eval2000 $dir/decode_eval2000${decode_affix:+_${decode_affix}}
    local/score_sclite.sh data/rt03 $dir/decode_rt03${decode_affix:+_${decode_affix}}
  fi
fi

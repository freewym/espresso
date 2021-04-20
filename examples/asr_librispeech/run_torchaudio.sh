#!/bin/bash
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e -o pipefail

stage=0
ngpus=1 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; will be automatically assigned if not specified

# E2E model related
affix=
train_set=train_960
valid_set=dev
test_set="test_clean test_other dev_clean dev_other"
checkpoint=checkpoint_best.pt
use_transformer=false

# LM related
lm_affix=
lm_checkpoint=checkpoint_best.pt
lm_shallow_fusion=true # no LM fusion if false
sentencepiece_vocabsize=5000
sentencepiece_type=unigram

# data related
corpus_root= # path to where you want to put the downloaded data; need to be specified if not on CLSP grid
if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  corpus_root=/export/corpora5
fi
folder_in_archive="LibriSpeech"
download=false # whether to download the corpus
data_dir=data
exp_dir=exp
tensorboard_logdir=

# feature configuration
apply_global_cmvn=true
overwrite_global_cmvn=false # overwrite the CMVN stats file even it already exists
apply_specaug=false


. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

lmdir=$exp_dir/lm_lstm${lm_affix:+_${lm_affix}}
if $use_transformer; then
  dir=$exp_dir/transformer${affix:+_$affix}
else
  dir=$exp_dir/lstm${affix:+_$affix}
fi

if [ ${stage} -le 0 ]; then
  echo "Stage 0: Data Downloading and Preparation"
  opts="--folder-in-archive $folder_in_archive"
  if $download; then
    opts="$opts --download"
  fi
  python3 ./local/prepare_librispeech.py $opts $corpus_root $data_dir

  mkdir -p $data_dir/$train_set
  cat $data_dir/{train_clean_100,train_clean_360,train_other_500}/wav.txt | sort -k1 > $data_dir/$train_set/wav.txt || exit 1;
  cat $data_dir/{train_clean_100,train_clean_360,train_other_500}/text.txt | sort -k1 > $data_dir/$train_set/text.txt || exit 1;

  mkdir -p $data_dir/$valid_set
  cat $data_dir/{dev_clean,dev_other}/wav.txt | sort -k1 > $data_dir/$valid_set/wav.txt || exit 1;
  cat $data_dir/{dev_clean,dev_other}/text.txt | sort -k1 > $data_dir/$valid_set/text.txt || exit 1;
fi

if [ ${stage} -le 1 ]; then
  echo "Stage 1: Data Preprocessing"
  for dataset in $train_set $valid_set $test_set; do
    wav2num_frames_file=$data_dir/$dataset/wav2num_frames
    if [ -f "$wav2num_frames_file" ]; then
      echo "$wav2num_frames_file exists, not overwriting it; continuing"
    else
      python3 ../../espresso/tools/wav2num_frames.py --num-workers 20 $data_dir/$dataset/wav.txt > $wav2num_frames_file
    fi
  done

  if $apply_global_cmvn; then
    gcmvn_file=$data_dir/$train_set/gcmvn.npz
    if [ -f "$gcmvn_file" ] && ! $overwrite_global_cmvn; then
      echo "$gcmvn_file exists, not overwriting it; continuing"
    else
      python3 ../../espresso/tools/compute_global_cmvn_stats.py --feature-type fbank --feat-dim 80 --num-workers 20 \
        $data_dir/$train_set/wav.txt $data_dir/$train_set
    fi
  fi
fi

if [ ${stage} -le 2 ]; then
  echo "Stage 2: Dump Json Files"
  for dataset in $train_set $valid_set $test_set; do
    wave=$data_dir/$dataset/wav.txt
    text=$data_dir/$dataset/text.txt
    utt2num_frames=$data_dir/$dataset/wav2num_frames
    if [[ $dataset == "$train_set" ]]; then
      name=train
    elif [[ $dataset == "$valid_set" ]]; then
      name=valid
    else
      name=$dataset
    fi
    python3 ../../espresso/tools/asr_prep_json.py --wave-files $wave --text-files $text --utt2num-frames-files $utt2num_frames \
      --output $data_dir/$name.json
  done
fi

dict=$data_dir/lang/${train_set}_${sentencepiece_type}${sentencepiece_vocabsize}_units.txt
sentencepiece_model=$data_dir/lang/${train_set}_${sentencepiece_type}${sentencepiece_vocabsize}
lmdatadir=$data_dir/lm_text
if [ ${stage} -le 3 ]; then
  echo "Stage 3: Dictionary Preparation and Text Tokenization"
  mkdir -p $data_dir/lang
  cut -f 2- -d" " $data_dir/${train_set}/text > $data_dir/lang/input
  echo "$0: training sentencepiece model..."
  python3 ../../scripts/spm_train.py --bos_id=-1 --pad_id=0 --eos_id=1 --unk_id=2 --input=$data_dir/lang/input \
    --vocab_size=$((sentencepiece_vocabsize+3)) --character_coverage=1.0 \
    --model_type=$sentencepiece_type --model_prefix=$sentencepiece_model \
    --input_sentence_size=10000000
  echo "$0: making a dictionary and tokenizing text for train/valid/test set..."
  for dataset in $train_set $valid_set $test_set; do
    text=$data_dir/$dataset/text.txt
    token_text=$data_dir/$dataset/token_text
    cut -f 2- -d" " $text | \
      python3 ../../scripts/spm_encode.py --model=${sentencepiece_model}.model --output_format=piece | \
      paste -d" " <(cut -f 1 -d" " $text) - > $token_text
    if [ "$dataset" == "$train_set" ]; then
      cut -f 2- -d" " $token_text | tr ' ' '\n' | sort | uniq -c | \
        awk '{print $2,$1}' | sort > $dict
      wc -l $dict
    fi
  done

  echo "$0: preparing text for subword LM..."
  mkdir -p $lmdatadir
  for dataset in $train_set $valid_set $test_set; do
    token_text=$data_dir/$dataset/token_text
    cut -f 2- -d" " $token_text > $lmdatadir/$dataset.tokens
  done
  if [ ! -e $lmdatadir/librispeech-lm-norm.txt.gz ]; then
    wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz -P $lmdatadir
  fi
  echo "$0: preparing extra corpus for subword LM training..."
  zcat $lmdatadir/librispeech-lm-norm.txt.gz | \
    python3 ../../scripts/spm_encode.py --model=${sentencepiece_model}.model --output_format=piece | \
    cat $lmdatadir/$train_set.tokens - > $lmdatadir/train.tokens
fi

lmdict=$dict
if [ ${stage} -le 4 ]; then
  echo "Stage 4: Text Binarization for subword LM Training"
  mkdir -p $lmdatadir/log
  for dataset in $test_set; do test_paths="$test_paths $lmdatadir/$dataset.tokens"; done
  test_paths=$(echo $test_paths | awk '{$1=$1;print}' | tr ' ' ',')
  ${decode_cmd} $lmdatadir/log/preprocess.log \
    python3 ../../fairseq_cli/preprocess.py --task language_modeling_for_asr \
      --workers 50 --srcdict $lmdict --only-source \
      --trainpref $lmdatadir/train.tokens \
      --validpref $lmdatadir/$valid_set.tokens \
      --testpref $test_paths \
      --destdir $lmdatadir
fi

if  [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  [ -z "$free_gpu" ] && free_gpu=$(free-gpu -n $ngpus) || echo "Unable to get $ngpus GPUs"
else
  [ -z "$free_gpu" ] && free_gpu=$(echo $(seq 0 $(($ngpus-1))) | sed 's/ /,/g')
fi
[ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
[ $(echo $free_gpu | sed 's/,/ /g' | awk '{print NF}') -ne "$ngpus" ] && \
  echo "number of GPU ids in --free-gpu=$free_gpu does not match --ngpus=$ngpus" && exit 1;

if [ ${stage} -le 5 ]; then
  echo "Stage 5: subword LM Training"
  valid_subset=valid
  mkdir -p $lmdir/log
  log_file=$lmdir/log/train.log
  [ -f $lmdir/checkpoint_last.pt ] && log_file="-a $log_file"
  opts=""
  [ ! -z "$tensorboard_logdir" ] && opts="$opts --tensorboard-logdir $tensorboard_logdir"
  update_freq=$(((2+ngpus-1)/ngpus))
  CUDA_VISIBLE_DEVICES=$free_gpu python3 ../../fairseq_cli/train.py $lmdatadir --seed 1 \
    --task language_modeling_for_asr --dict $lmdict \
    --log-interval $((16000/ngpus/update_freq)) --log-format simple \
    --num-workers 4 --max-tokens 32000 --batch-size 1024 --curriculum 1 \
    --valid-subset $valid_subset --batch-size-valid 1536 --update-freq $update_freq \
    --distributed-world-size $ngpus \
    --max-epoch 30 --optimizer adam --lr 0.001 --clip-norm 1.0 \
    --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
    --save-dir $lmdir --restore-file checkpoint_last.pt --save-interval-updates $((16000/ngpus/update_freq)) \
    --keep-interval-updates 3 --keep-last-epochs 5 --validate-interval 1 \
    --arch lstm_lm_librispeech --criterion cross_entropy --sample-break-mode eos $opts 2>&1 | tee $log_file
fi

if [ ${stage} -le 6 ]; then
  echo "Stage 6: subword LM Evaluation"
  gen_set_array=(test)
  num=$(echo $test_set | awk '{print NF-1}')
  for i in $(seq $num); do gen_set_array[$i]="test$i"; done
  test_set_array=($test_set)
  for i in $(seq 0 $num); do
    log_file=$lmdir/log/evaluation_${test_set_array[$i]}.log
    python3 ../../fairseq_cli/eval_lm.py $lmdatadir --cpu \
      --task language_modeling_for_asr --dict $lmdict --gen-subset ${gen_set_array[$i]} \
      --max-tokens 40960 --batch-size 1536 --sample-break-mode eos \
      --path $lmdir/$lm_checkpoint 2>&1 | tee $log_file
  done
fi

if [ ${stage} -le 7 ]; then
  echo "Stage 7: Model Training"
  valid_subset=valid
  mkdir -p $dir/log
  log_file=$dir/log/train.log
  [ -f $dir/checkpoint_last.pt ] && log_file="-a $log_file"
  opts=""
  [ ! -z "$tensorboard_logdir" ] && opts="$opts --tensorboard-logdir $tensorboard_logdir"
  if $apply_global_cmvn; then
    gcmvn_file=$data_dir/$train_set/gcmvn.npz
    [ ! -f "$gcmvn_file" ] && echo "$gcmvn_file not found. Please generate it first" && exit 1;
    opts="$opts --global-cmvn-stats-path $gcmvn_file"
  fi
  if $use_transformer; then
    update_freq=$(((8+ngpus-1)/ngpus))
    opts="$opts --arch speech_transformer_librispeech --max-tokens 22000 --max-epoch 100 --lr-scheduler tri_stage"
    opts="$opts --warmup-steps $((25000/ngpus/update_freq)) --hold-steps $((900000/ngpus/update_freq)) --decay-steps $((1550000/ngpus/update_freq))"
    if $apply_specaug; then
      specaug_config="{'W': 80, 'F': 27, 'T': 100, 'num_freq_masks': 2, 'num_time_masks': 2, 'p': 1.0}"
    fi
  else
    update_freq=$(((2+ngpus-1)/ngpus))
    opts="$opts --arch speech_conv_lstm_librispeech"
    if $apply_specaug; then
      opts="$opts --max-epoch 95 --lr-scheduler tri_stage"
      opts="$opts --warmup-steps $((2000/ngpus/update_freq)) --hold-steps $((600000/ngpus/update_freq)) --decay-steps $((1040000/ngpus/update_freq))"
      opts="$opts --encoder-rnn-layers 5"
      specaug_config="{'W': 80, 'F': 27, 'T': 100, 'num_freq_masks': 2, 'num_time_masks': 2, 'p': 1.0}"
    else
      opts="$opts --max-epoch 30 --lr-scheduler reduce_lr_on_plateau_v2 --lr-shrink 0.5 --start-reduce-lr-epoch 10"
    fi
  fi
  CUDA_VISIBLE_DEVICES=$free_gpu speech_train.py $data_dir --task speech_recognition_espresso --seed 1 \
    --log-interval $((8000/ngpus/update_freq)) --log-format simple --print-training-sample-interval $((4000/ngpus/update_freq)) \
    --num-workers 4 --data-buffer-size 0 --max-tokens 26000 --batch-size 24 --curriculum 1 --empty-cache-freq 50 \
    --valid-subset $valid_subset --batch-size-valid 48 --ddp-backend legacy_ddp --update-freq $update_freq \
    --distributed-world-size $ngpus \
    --optimizer adam --lr 0.001 --weight-decay 0.0 --clip-norm 2.0 \
    --save-dir $dir --restore-file checkpoint_last.pt --save-interval-updates $((6000/ngpus/update_freq)) \
    --keep-interval-updates 3 --keep-last-epochs 5 --validate-interval 1 --best-checkpoint-metric wer \
    --criterion label_smoothed_cross_entropy_v2 --label-smoothing 0.1 --smoothing-type uniform \
    --dict $dict --bpe sentencepiece --sentencepiece-model ${sentencepiece_model}.model \
    --max-source-positions 9999 --max-target-positions 999 \
    $opts --specaugment-config "$specaug_config" 2>&1 | tee $log_file
fi

if [ ${stage} -le 8 ]; then
  echo "Stage 8: Decoding"
  opts=""
  path=$dir/$checkpoint
  decode_affix=
  if $lm_shallow_fusion; then
    opts="$opts --lm-path $lmdir/$lm_checkpoint"
    opts="$opts --lm-weight 0.47 --eos-factor 1.5"
    if $apply_specaug; then
      # overwrite the existing opts
      opts="$opts --lm-weight 0.4"
    fi
    decode_affix=shallow_fusion
  fi
  for dataset in $test_set; do
    decode_dir=$dir/decode_$dataset${decode_affix:+_${decode_affix}}
    CUDA_VISIBLE_DEVICES=$(echo $free_gpu | sed 's/,/ /g' | awk '{print $1}') speech_recognize.py $data_dir \
      --task speech_recognition_espresso --max-tokens 15000 --batch-size 24 \
      --num-shards 1 --shard-id 0 --dict $dict --bpe sentencepiece --sentencepiece-model ${sentencepiece_model}.model \
      --gen-subset $dataset --max-source-positions 9999 --max-target-positions 999 \
      --path $path --beam 60 --max-len-a 0.08 --max-len-b 0 --lenpen 1.0 \
      --results-path $decode_dir $opts

    echo "log saved in ${decode_dir}/decode.log"
  done
fi

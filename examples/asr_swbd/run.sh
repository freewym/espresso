#!/bin/bash
# Copyright (c) Hang Lyu, Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e -o pipefail

stage=0
ngpus=1 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid

# E2E model related
affix=
train_set=train_nodup
valid_set=train_dev
test_set="train_dev eval2000 rt03"
checkpoint=checkpoint_best.pt
use_transformer=false

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

# feature configuration
do_delta=false
apply_specaug=false


. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

lmdir=exp/lm_lstm${lm_affix:+_${lm_affix}}
if $use_transformer; then
  dir=exp/transformer${affix:+_$affix}
else
  dir=exp/lstm${affix:+_$affix}
fi

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
      /export/b1{4,5,6,7}/$USER/fairseq-data/egs/asr_swbd/dump/$train_set/delta${do_delta}/storage \
      $train_feat_dir/storage
  fi
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $valid_feat_dir/storage ]; then
    utils/create_split_dir.pl \
      /export/b1{4,5,6,7}/$USER/fairseq-data/egs/asr_swbd/dump/$valid_set/delta${do_delta}/storage \
      $valid_feat_dir/storage
  fi
  dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
    data/$train_set/feats.scp data/$train_set/cmvn.ark exp/dump_feats/train $train_feat_dir
  dump.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta \
    data/$valid_set/feats.scp data/$train_set/cmvn.ark exp/dump_feats/dev $valid_feat_dir
  for rtask in $test_set; do
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

  echo "$0: making a non-linguistic symbol list..."
  train_text=data/$train_set/text
  cut -f 2- $train_text | tr " " "\n" | sort | uniq | grep "\[" > $nlsyms
  cat $nlsyms

  echo "$0: preparing extra corpus for subword LM training..."
  if [ -f $lmdatadir/fisher_text0 ]; then
    rm -rf $lmdatadir/fisher_text0
  fi
  for x in $fisher_dirs; do
    [ ! -d $x/data/trans ] \
      && "Cannot find transcripts in Fisher directory $x" && exit 1;
    cat $x/data/trans/*/*.txt | \
      grep -v "^#" | grep -v "^$" | cut -d" " -f4- >> $lmdatadir/fisher_text0
  done
  cat $lmdatadir/fisher_text0 | local/fisher_map_words.pl | \
    sed 's/^[ \t]*//'> $lmdatadir/fisher_text

  echo "$0: training sentencepiece model..."
  cut -f 2- -d" " data/$train_set/text | cat - $lmdatadir/fisher_text > data/lang/input
  python3 ../../scripts/spm_train.py --bos_id=-1 --pad_id=0 --eos_id=1 --unk_id=2 --input=data/lang/input \
    --vocab_size=$((sentencepiece_vocabsize+3)) --character_coverage=1.0 \
    --model_type=$sentencepiece_type --model_prefix=$sentencepiece_model \
    --input_sentence_size=10000000 \
    --user_defined_symbols=$(cat $nlsyms | tr "\n" "," | sed 's/,$//')

  echo "$0: tokenizing text for train/valid/test sets..."
  for dataset in $train_set $test_set; do  # validation is included in tests
    text=data/$dataset/text
    token_text=data/$dataset/token_text
    cut -f 2- -d" " $text | \
      python3 ../../scripts/spm_encode.py --model=${sentencepiece_model}.model --output_format=piece | \
      paste -d" " <(cut -f 1 -d" " $text) - > $token_text
    cut -f 2- -d" " $token_text > $lmdatadir/$dataset.tokens
  done

  echo "$0: tokenizing extra corpus for subword LM training..."
  cat $lmdatadir/fisher_text | \
    python3 ../../scripts/spm_encode.py --model=${sentencepiece_model}.model --output_format=piece | \
    cat $lmdatadir/$train_set.tokens - > $lmdatadir/train.tokens

  echo "$0: making a subword dictionary with swbd+fisher text"
  cat $lmdatadir/train.tokens | tr " " "\n" | grep -v -e "^\s*$" | sort | \
    uniq -c | awk '{print $2,$1}' > $dict
  wc -l $dict
fi

lmdict=$dict
if [ $stage -le 3 ]; then
  echo "Stage 3: Text Binarization for subword LM Training"
  mkdir -p $lmdatadir/log
  test_paths= && for dataset in $test_set; do test_paths="$test_paths $lmdatadir/$dataset.tokens"; done
  test_paths=$(echo $test_paths | awk '{$1=$1;print}' | tr ' ' ',')
  ${decode_cmd} $lmdatadir/log/preprocess.log \
    python3 ../../fairseq_cli/preprocess.py --user-dir espresso --task language_modeling_for_asr \
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
  mkdir -p $lmdir/log
  log_file=$lmdir/log/train.log
  [ -f $lmdir/checkpoint_last.pt ] && log_file="-a $log_file"
  CUDA_VISIBLE_DEVICES=$free_gpu python3 ../../fairseq_cli/train.py $lmdatadir --seed 1 --user-dir espresso \
    --task language_modeling_for_asr --dict $lmdict \
    --log-interval $((1000/ngpus)) --log-format simple \
    --num-workers 0 --max-tokens 25600 --max-sentences 1024 \
    --valid-subset $valid_subset --max-sentences-valid 1536 \
    --distributed-world-size $ngpus --distributed-port $(if [ $ngpus -gt 1 ]; then echo 100; else echo -1; fi) \
    --max-epoch 25 --optimizer adam --lr 0.001 --clip-norm 1.0 \
    --lr-scheduler reduce_lr_on_plateau --lr-shrink 0.5 \
    --save-dir $lmdir --restore-file checkpoint_last.pt --save-interval-updates $((1000/ngpus)) \
    --keep-interval-updates 3 --keep-last-epochs 5 --validate-interval 1 \
    --arch lstm_lm_swbd --criterion cross_entropy --sample-break-mode eos 2>&1 | tee $log_file
fi

if [ $stage -le 5 ]; then
  echo "Stage 5: subword LM Evaluation"
  gen_set_array=(test)
  num=$(echo $test_set | awk '{print NF-1}')
  for i in $(seq $num); do gen_set_array[$i]="test$i"; done  #gen_set_array=(test test1 test2)
  test_set_array=($test_set)
  for i in $(seq 0 $num); do
    log_file=$lmdir/log/evaluation_${test_set_array[$i]}.log
    python3 ../../fairseq_cli/eval_lm.py $lmdatadir --user-dir espresso --cpu \
      --task language_modeling_for_asr --dict $lmdict --gen-subset ${gen_set_array[$i]} \
      --max-tokens 40960 --max-sentences 1536 --sample-break-mode eos \
      --path $lmdir/$lm_checkpoint 2>&1 | tee $log_file
  done
fi

if [ $stage -le 6 ]; then
  echo "Stage 6: Dump Json Files"
  train_feat=$train_feat_dir/feats.scp
  train_token_text=data/$train_set/token_text
  train_utt2num_frames=data/$train_set/utt2num_frames
  valid_feat=$valid_feat_dir/feats.scp
  valid_token_text=data/$valid_set/token_text
  valid_utt2num_frames=data/$valid_set/utt2num_frames
  asr_prep_json.py --feat-files $train_feat --token-text-files $train_token_text --utt2num-frames-files $train_utt2num_frames --output data/train.json
  asr_prep_json.py --feat-files $valid_feat --token-text-files $valid_token_text --utt2num-frames-files $valid_utt2num_frames --output data/valid.json
  for dataset in $test_set; do
    feat=${dumpdir}/$dataset/delta${do_delta}/feats.scp
    utt2num_frames=data/$dataset/utt2num_frames
    # only score train_dev with built-in scorer
    text_opt= && [ "$dataset" == "train_dev" ] && text_opt="--token-text-files data/$dataset/token_text"
    asr_prep_json.py --feat-files $feat $text_opt --utt2num-frames-files $utt2num_frames --output data/$dataset.json
  done
fi

if [ $stage -le 7 ]; then
  echo "Stage 7: Model Training"
  valid_subset=valid
  opts=""
  [ -f local/wer_output_filter ] && opts="$opts --wer-output-filter local/wer_output_filter"
  mkdir -p $dir/log
  log_file=$dir/log/train.log
  [ -f $dir/checkpoint_last.pt ] && log_file="-a $log_file"
  update_freq=$(((2+ngpus-1)/ngpus))
  if $use_transformer; then
    opts="$opts --arch speech_transformer_swbd --max-epoch 100 --lr-scheduler tri_stage"
    opts="$opts --warmup-steps $((25000/ngpus/update_freq)) --hold-steps $((180000/ngpus/update_freq)) --decay-steps $((320000/ngpus/update_freq))"
    if $apply_specaug; then
      specaug_config="{'W': 40, 'F': 18, 'T': 70, 'num_freq_masks': 2, 'num_time_masks': 2, 'p': 0.2}"
    fi
  else
    opts="$opts --arch speech_conv_lstm_swbd --scheduled-sampling-probs 0.9,0.8,0.7,0.6 --start-scheduled-sampling-epoch 6"
    if $apply_specaug; then
      opts="$opts --max-epoch 100 --lr-scheduler tri_stage --warmup-steps $((1000/ngpus/update_freq)) --hold-steps $((180000/ngpus/update_freq)) --decay-steps $((360000/ngpus/update_freq))"
      opts="$opts --encoder-rnn-hidden-size 1024 --encoder-rnn-layers 5 --decoder-embed-dim 512 --decoder-hidden-size 1024"
      opts="$opts --decoder-out-embed-dim 3072 --attention-dim 512 --dropout 0.4"
      specaug_config="{'W': 40, 'F': 18, 'T': 70, 'num_freq_masks': 2, 'num_time_masks': 2, 'p': 0.2}"
    else
      opts="$opts --max-epoch 35 --lr-scheduler reduce_lr_on_plateau_v2 --lr-shrink 0.5 --start-reduce-lr-epoch 14"
    fi
  fi
  CUDA_VISIBLE_DEVICES=$free_gpu speech_train.py data --task speech_recognition_espresso --seed 1 --user-dir espresso \
    --log-interval $((3000/ngpus/update_freq)) --log-format simple --print-training-sample-interval $((4000/ngpus/update_freq)) \
    --num-workers 0 --data-buffer-size 0 --max-tokens 26000 --max-sentences 48 --curriculum 2 --empty-cache-freq 50 \
    --valid-subset $valid_subset --max-sentences-valid 64 --ddp-backend no_c10d --update-freq $update_freq \
    --distributed-world-size $ngpus --distributed-port $(if [ $ngpus -gt 1 ]; then echo 100; else echo -1; fi) \
    --optimizer adam --lr 0.001 --weight-decay 0.0 --clip-norm 2.0 \
    --save-dir $dir --restore-file checkpoint_last.pt --save-interval-updates $((3000/ngpus/update_freq)) \
    --keep-interval-updates 3 --keep-last-epochs 5 --validate-interval 1 --best-checkpoint-metric wer \
    --criterion label_smoothed_cross_entropy_v2 --label-smoothing 0.1 --smoothing-type uniform \
    --dict $dict --bpe sentencepiece --sentencepiece-vocab ${sentencepiece_model}.model --non-lang-syms $nlsyms \
    --max-source-positions 9999 --max-target-positions 999 \
    $opts --specaugment-config "$specaug_config" 2>&1 | tee $log_file
fi

if [ $stage -le 8 ]; then
  echo "Stage 8: Decoding"
  [ ! -d $KALDI_ROOT ] && echo "Expected $KALDI_ROOT to exist" && exit 1;
  opts=""
  path=$dir/$checkpoint
  decode_affix=
  if $lm_shallow_fusion; then
    path="$path:$lmdir/$lm_checkpoint"
    opts="$opts --lm-weight 0.25"
    decode_affix=shallow_fusion
  fi
  [ -f local/wer_output_filter ] && opts="$opts --wer-output-filter local/wer_output_filter"
  for dataset in $test_set; do
    decode_dir=$dir/decode_${dataset}${decode_affix:+_${decode_affix}}
    CUDA_VISIBLE_DEVICES=$(echo $free_gpu | sed 's/,/ /g' | awk '{print $1}') speech_recognize.py data \
      --task speech_recognition_espresso --user-dir espresso --max-tokens 24000 --max-sentences 48 \
      --num-shards 1 --shard-id 0 --dict $dict --bpe sentencepiece --sentencepiece-vocab ${sentencepiece_model}.model \
      --non-lang-syms $nlsyms --gen-subset $dataset --max-source-positions 9999 --max-target-positions 999 \
      --path $path --beam 35 --max-len-a 0.1 --max-len-b 0 --lenpen 1.0 \
      --results-path $decode_dir $opts

    echo "log saved in ${decode_dir}/decode.log"
    echo "Scoring with kaldi..."
    local/score_e2e.sh data/$dataset $decode_dir
    if [ "$dataset" == "train_dev" ]; then
      echo -n "tran_dev: " && cat $decode_dir/scoring/wer | grep WER
    elif [ "$dataset" == "eval2000" ] || [ "$dataset" == "rt03" ]; then
      echo -n "$dataset: " && grep Sum $decode_dir/scoring/$dataset.ctm.filt.sys | \
        awk '{print "WER="$11"%, Sub="$8"%, Ins="$10"%, Del="$9"%"}' | tee $decode_dir/wer
      echo -n "swbd subset: " && grep Sum $decode_dir/scoring/$dataset.ctm.swbd.filt.sys | \
        awk '{print "WER="$11"%, Sub="$8"%, Ins="$10"%, Del="$9"%"}' | tee $decode_dir/wer_swbd
      subset=callhm && [ "$dataset" == "rt03" ] && subset=fsh
      echo -n "$subset subset: " && grep Sum $decode_dir/scoring/$dataset.ctm.$subset.filt.sys | \
        awk '{print "WER="$11"%, Sub="$8"%, Ins="$10"%, Del="$9"%"}' | tee $decode_dir/wer_$subset
      echo "WERs saved in $decode_dir/wer*"
    fi
  done
fi

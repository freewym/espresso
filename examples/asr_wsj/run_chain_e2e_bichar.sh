#!/bin/bash
# Copyright (c) Yiming Wang, Yiwen Shao
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e -o pipefail

stage=-10
ngpus=1 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid

# model and data related
affix=
lang=data/lang_chain_e2e_char
tree_dir=exp/chain/e2e_bichar_tree  # it's actually just a trivial tree (no tree building)
whole_train_set=train_si284_sp  # will be split into train_set and valid_set
train_set=train_si284_novalid_spe2e
valid_set=train_si284_valid_spe2e
test_set="test_dev93 test_eval92"
dumpdir=data/dump   # directory to dump full features
checkpoint=checkpoint_best.pt

wsj0=
wsj1=
if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  wsj0=/export/corpora5/LDC/LDC93S6B
  wsj1=/export/corpora5/LDC/LDC94S13B
fi

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

dir=exp/tdnn_chain_e2e_bichar${affix:+_$affix}

local/data_prep_char.sh --stage $stage --wsj0 $wsj0 --wsj1 $wsj1 || exit 1;

if [ $stage -le 0 ]; then
  echo "Stage 0: Create the $lang Directory that Has a Specific HMM Topolopy"
  rm -rf $lang
  cp -r data/lang_char $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 1 ]; then
  echo "Stage 1: Generate Denominator Graph and Numerator Fsts"
  echo "$0: Estimating a phone language model for the denominator graph..."
  mkdir -p $tree_dir/log
  $train_cmd $tree_dir/log/make_phone_lm.log \
    cat data/${whole_train_set}_hires/text \| \
    steps/nnet3/chain/e2e/text_to_phones.py --between-silprob 0.1 \
    data/lang_char \| \
    utils/sym2int.pl -f 2- data/lang_char/phones.txt \| \
    chain-est-phone-lm --num-extra-lm-states=2000 \
    ark:- $tree_dir/phone_lm.fst
  nj=32
  steps/nnet3/chain/e2e/prepare_e2e.sh --nj $nj --cmd "$train_cmd" \
    --type biphone --shared-phones true --tie true data/${whole_train_set}_hires $lang $tree_dir
  echo "$0: Making denominator fst..."
  $decode_cmd $tree_dir/log/make_den_fst.log \
    chain-make-den-fst $tree_dir/tree $tree_dir/0.trans_mdl $tree_dir/phone_lm.fst \
      $tree_dir/den.fst $tree_dir/normalization.fst || exit 1
  echo "$0: Making numerator fsts..."
  abs_treedir=`utils/make_absolute.sh $tree_dir`
  $decode_cmd JOB=1:$nj $tree_dir/log/make_num_fst_e2e.JOB.log \
    chain-make-num-fst-e2e $tree_dir/0.trans_mdl $tree_dir/normalization.fst \
      scp:$tree_dir/fst.JOB.scp ark,scp:$abs_treedir/fst_nor.JOB.ark,$abs_treedir/fst_nor.JOB.scp || exit 1
  for n in $(seq $nj); do
    cat $tree_dir/fst_nor.$n.scp || exit 1
  done > $tree_dir/fst_nor.scp || exit 1
fi

if [ ${stage} -le 2 ]; then
  echo "Stage 2: Split the Whole Train Set into Train/Valid Set"
  # Get list of validation utterances.
  data=data/${whole_train_set}_hires
  set +e
  awk '{print $1}' $data/utt2spk | utils/shuffle_list.pl 2>/dev/null | head -300 > valid_uttlist
  set -e
  if [ -f $data/utt2uniq ]; then  # this matters if you use data augmentation.
    echo "File $data/utt2uniq exists, so augmenting valid_uttlist to"
    echo "include all perturbed versions of the same 'real' utterances."
    mv valid_uttlist valid_uttlist.tmp
    utils/utt2spk_to_spk2utt.pl $data/utt2uniq > uniq2utt
    cat valid_uttlist.tmp | utils/apply_map.pl $data/utt2uniq | \
      sort | uniq | utils/apply_map.pl uniq2utt | \
      awk '{for(n=1;n<=NF;n++) print $n;}' | sort  > valid_uttlist
    rm uniq2utt valid_uttlist.tmp 2>/dev/null
  fi
  # generate train/valid data dir
  utils/filter_scp.pl --exclude valid_uttlist $data/utt2spk | cut -d" " -f1 > novalid_uttlist || exit 1
  utils/subset_data_dir.sh --utt-list novalid_uttlist $data data/${train_set}_hires || exit 1
  utils/subset_data_dir.sh --utt-list valid_uttlist $data data/${valid_set}_hires || exit 1

  # generate train/valid numerator fst file
  utils/filter_scp.pl novalid_uttlist $tree_dir/fst_nor.scp > $tree_dir/fst_novalid_nor.scp || exit 1
  utils/filter_scp.pl valid_uttlist $tree_dir/fst_nor.scp > $tree_dir/fst_valid_nor.scp || exit 1
  rm valid_uttlist novalid_uttlist 2>/dev/null

  # not all fsts can be generated successfully, just filter out those not having the fst
  for dataset in $train_set $valid_set; do
    tag=novalid && [[ "$dataset" == "$valid_set" ]] && tag=valid
    cp data/${dataset}_hires/feats.scp data/${dataset}_hires/feats.scp.tmp
    utils/filter_scp.pl $tree_dir/fst_${tag}_nor.scp data/${dataset}_hires/feats.scp.tmp \
      > data/${dataset}_hires/feats.scp || exit 1
    rm data/${dataset}_hires/feats.scp.tmp 2>/dev/null
    utils/fix_data_dir.sh data/${dataset}_hires || exit 1
  done
fi

if [ ${stage} -le 3 ]; then
  echo "Stage 3: Dump Feature"
  for dataset in $train_set $valid_set $test_set; do
    nj=8
    utils/split_data.sh data/${dataset}_hires $nj
    sdata=data/${dataset}_hires/split$nj
    mkdir -p $dumpdir/${dataset}_hires; abs_featdir=`utils/make_absolute.sh $dumpdir/${dataset}_hires`
    $train_cmd JOB=1:$nj $abs_featdir/log/dump_feature.JOB.log \
      apply-cmvn --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp \
        scp:$sdata/JOB/feats.scp ark:- \| \
      copy-feats --compress=true --compression-method=2 ark:- \
        ark,scp:$abs_featdir/feats.JOB.ark,$abs_featdir/feats.JOB.scp || exit 1
    for n in $(seq $nj); do
      cat $abs_featdir/feats.$n.scp || exit 1
    done > $abs_featdir/feats.scp || exit 1
    rm $abs_featdir/feats.*.scp 2>/dev/null
    cat data/${dataset}_hires/utt2num_frames > $abs_featdir/utt2num_frames || exit 1
    cat data/${dataset}_hires/utt2spk > $abs_featdir/utt2spk || exit 1
  done
fi

if [ ${stage} -le 4 ]; then
  echo "Stage 4: Make Graphs"
  for lmtype in tgpr bd_tgpr; do
    utils/lang/check_phones_compatible.sh \
      data/lang_char_test_$lmtype/phones.txt $lang/phones.txt
    utils/mkgraph.sh --self-loop-scale 1.0 data/lang_char_test_$lmtype $tree_dir $tree_dir/graph_$lmtype || exit 1
  done
fi

if [ ${stage} -le 5 ]; then
  echo "Stage 5: Dump Json Files"
  train_feat=$dumpdir/${train_set}_hires/feats.scp
  train_fst=${tree_dir}/fst_novalid_nor.scp
  train_text=data/${train_set}_hires/text
  train_utt2num_frames=data/${train_set}_hires/utt2num_frames
  valid_feat=$dumpdir/${valid_set}_hires/feats.scp
  valid_fst=${tree_dir}/fst_valid_nor.scp
  valid_text=data/${valid_set}_hires/text
  valid_utt2num_frames=data/${valid_set}_hires/utt2num_frames
  mkdir -p data/chain_e2e_bichar
  asr_prep_json.py --feat-files $train_feat --numerator-fst-files $train_fst --text-files $train_text \
    --utt2num-frames-files $train_utt2num_frames --output data/chain_e2e_bichar/train.json
  asr_prep_json.py --feat-files $valid_feat --numerator-fst-files $valid_fst --text-files $valid_text \
    --utt2num-frames-files $valid_utt2num_frames --output data/chain_e2e_bichar/valid.json
  for dataset in $test_set; do
    nj=$(wc -l <data/${dataset}_hires/spk2utt)
    utils/split_data.sh data/${dataset}_hires $nj
    utils/split_data.sh $dumpdir/${dataset}_hires $nj
    for n in $(seq $nj); do
      feat=$dumpdir/${dataset}_hires/split$nj/$n/feats.scp
      text=data/${dataset}_hires/split$nj/$n/text
      utt2num_frames=data/${dataset}_hires/split$nj/$n/utt2num_frames
      asr_prep_json.py --feat-files $feat --text-files $text --utt2num-frames-files $utt2num_frames \
        --output data/chain_e2e_bichar/$dataset.$n.json
    done
  done
fi

[ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu -n $ngpus) || \
  echo "Unable to get $ngpus GPUs"
[ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
[ $(echo $free_gpu | sed 's/,/ /g' | awk '{print NF}') -ne "$ngpus" ] && \
  echo "number of GPU ids in --free-gpu=$free_gpu does not match --ngpus=$ngpus" && exit 1;

num_targets=$(tree-info ${tree_dir}/tree | grep num-pdfs | awk '{print $2}')

if [ ${stage} -le 6 ]; then
  echo "Stage 6: Model Training"
  valid_subset=valid
  mkdir -p $dir/log
  log_file=$dir/log/train.log
  [ -f $dir/checkpoint_last.pt ] && log_file="-a $log_file"
  update_freq=1
  CUDA_VISIBLE_DEVICES=$free_gpu speech_train.py data/chain_e2e_bichar --task speech_recognition_hybrid --seed 1 --user-dir espresso \
    --log-interval $((200/ngpus/update_freq)) --log-format simple \
    --num-workers 0 --data-buffer-size 0 --max-tokens 120000 --batch-size 128 --curriculum 1 --empty-cache-freq 50 \
    --valid-subset $valid_subset --batch-size-valid 128 --ddp-backend no_c10d --update-freq $update_freq \
    --distributed-world-size $ngpus \
    --max-epoch 30 --optimizer adam --lr 0.001 --weight-decay 0.0 --start-reduce-lr-epoch 11 \
    --lr-scheduler reduce_lr_on_plateau_v2 --lr-shrink 0.5 \
    --save-dir $dir --restore-file checkpoint_last.pt --save-interval-updates $((400/ngpus/update_freq)) \
    --keep-interval-updates 5 --keep-last-epochs 5 --validate-interval 1 --best-checkpoint-metric nll_loss \
    --arch speech_tdnn_wsj --criterion lattice_free_mmi --num-targets $num_targets \
    --dropout 0.2 --kernel-sizes "[3]*6" --strides "[1]*5+[3]" --dilations "[1,1,1,3,3,3]" --num-layers 6 \
    --denominator-fst-path $tree_dir/den.fst --leaky-hmm-coefficient 1e-03 --output-l2-regularization-coefficient 5e-05 \
    --max-source-positions 9999 --max-target-positions 9999 2>&1 | tee $log_file
fi

if [ ${stage} -le 7 ]; then
  echo "Stage 7: Decoding"
  rm $dir/.error 2>/dev/null || true
  queue_opt="--num-threads 4"
  path=$dir/$checkpoint
  for dataset in $test_set; do
    (
      data_affix=$(echo $dataset | sed s/test_//)
      nj=$(wc -l <data/${dataset}_hires/spk2utt)
      for lmtype in tgpr bd_tgpr; do
        graph_dir=$tree_dir/graph_${lmtype}
        $decode_cmd $queue_opt JOB=1:$nj $dir/decode_${lmtype}_${data_affix}/log/decode.JOB.log \
          dump_posteriors.py data/chain_e2e_bichar --cpu --task speech_recognition_hybrid --user-dir espresso \
            --max-tokens 120000 --batch-size 128 --num-shards 1 --shard-id 0 --num-targets $num_targets \
            --gen-subset $dataset.JOB \
            --max-source-positions 9999 --path $path \| \
          latgen-faster-mapped --max-active=7000 --min-active=20 --beam=15 --lattice-beam=8 --acoustic-scale=1.0 \
            --allow-partial=true --word-symbol-table="$graph_dir/words.txt" \
            $tree_dir/0.trans_mdl $graph_dir/HCLG.fst ark:- \
            "ark:| lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >$dir/decode_${lmtype}_${data_affix}/lat.JOB.gz" || exit 1
        local/score.sh --cmd "$decode_cmd" data/${dataset}_hires $graph_dir $dir/decode_${lmtype}_${data_affix} || exit 1
        echo $nj > $dir/decode_${lmtype}_${data_affix}/num_jobs
      done
      steps/lmrescore.sh --cmd "$decode_cmd" --self-loop-scale 1.0 --mode 3 data/lang_char_test_{tgpr,tg} \
        data/${dataset}_hires $dir/decode_{tgpr,tg}_${data_affix} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang_char_test_bd_{tgpr,fgconst} \
        data/${dataset}_hires $dir/decode_bd_tgpr_${data_affix}{,_fg} || exit 1
  ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
  for dataset in $test_set; do
    data_affix=$(echo $dataset | sed s/test_//)
    for x in $dir/decode_{tgpr_${data_affix},tg_${data_affix},bd_tgpr_${data_affix},bd_tgpr_${data_affix}_fg}; do
      grep WER $x/wer_* | utils/best_wer.sh
    done
  done
fi

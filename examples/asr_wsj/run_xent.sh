#!/bin/bash
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# need to make the following soft links to corresponding dirs in Kaldi:
# ln -s <kaldi-dir>/egs/wsj/s5/exp/tri4b exp/tri4b
# ln -s <kaldi-dir>/egs/wsj/s5/exp/tri4b_ali_train_si284_sp data/tri4b_ali_train_si284_sp
# ln -s <kaldi-dir>/egs/wsj/s5/data/lang_test_tgpr data/lang_test_tgpr
# ln -s <kaldi-dir>/egs/wsj/s5/data/lang_test_tg data/lang_test_tg
# ln -s <kaldi-dir>/egs/wsj/s5/data/lang_test_bd_tgpr data/lang_test_bd_tgpr
# ln -s <kaldi-dir>/egs/wsj/s5/data/lang_test_bd_fgconst data/lang_test_bd_fgconst
# ln -s <kaldi-dir>/egs/wsj/s5/data/train_si284_sp_hires data/train_si284_sp_hires
# ln -s <kaldi-dir>/egs/wsj/s5/data/test_dev93_hires data/test_dev93_hires
# ln -s <kaldi-dir>/egs/wsj/s5/data/test_eval92_hires data/test_eval92_hires

set -e -o pipefail

stage=0
ngpus=1 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid

# model and data related
affix=
gmm=tri4b
lang=data/lang_test
whole_train_set=train_si284_sp  # will be split into train_set and valid_set
train_set=train_si284_novalid_sp
valid_set=train_si284_valid_sp
test_set="test_dev93 test_eval92"
dumpdir=data/dump   # directory to dump full features
checkpoint=checkpoint_best.pt


. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

dir=exp/tdnn_xent${affix:+_$affix}

if [ ${stage} -le 0 ]; then
  echo "Stage 0: Split the Whole Train Set into Train/Valid Data and Ali Dirs"
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
  rm valid_uttlist novalid_uttlist 2>/dev/null

  # generate train/valid ali dir
  steps/subset_ali_dir.sh $data data/${train_set}_hires data/${gmm}_ali_${whole_train_set} \
    data/${gmm}_ali_${train_set} || exit 1
  steps/subset_ali_dir.sh $data data/${valid_set}_hires data/${gmm}_ali_${whole_train_set} \
    data/${gmm}_ali_${valid_set} || exit 1
fi

if [ ${stage} -le 1 ]; then
  echo "Stage 1: Convert Alignments from transition-ids to pdf-ids"
  for dataset in $train_set $valid_set; do
    abs_alidir=`utils/make_absolute.sh data/${gmm}_ali_$dataset`
    nj=$(cat ${abs_alidir}/num_jobs)
    $decode_cmd JOB=1:$nj ${abs_alidir}/log/ali_to_pdf.JOB.log \
      ali-to-pdf ${abs_alidir}/final.mdl \
        "ark,s,cs:gunzip -c ${abs_alidir}/ali.JOB.gz |" \
        ark,scp:${abs_alidir}/ali_pdf.JOB.ark,${abs_alidir}/ali_pdf.JOB.scp || exit 1
    for n in $(seq $nj); do
      cat ${abs_alidir}/ali_pdf.$n.scp || exit 1
    done > ${abs_alidir}/ali_pdf.scp || exit 1
    rm ${abs_alidir}/ali_pdf.*.scp 2>/dev/null

    # not all alignments can be generated successfully, just filter out those not having the alignment
    cp data/${dataset}_hires/feats.scp data/${dataset}_hires/feats.scp.tmp
    utils/filter_scp.pl ${abs_alidir}/ali_pdf.scp data/${dataset}_hires/feats.scp.tmp \
      > data/${dataset}_hires/feats.scp || exit 1
    rm data/${dataset}_hires/feats.scp.tmp 2>/dev/null
    utils/fix_data_dir.sh data/${dataset}_hires || exit 1
  done
fi

if [ ${stage} -le 2 ]; then
  echo "Stage 2: Dump Feature"
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

if [ ${stage} -le 3 ]; then
  echo "Stage 3: Make Graphs"
  for lmtype in tgpr bd_tgpr; do
    utils/mkgraph.sh ${lang}_$lmtype exp/$gmm exp/$gmm/graph_$lmtype || exit 1
  done
fi

num_targets=$(tree-info data/${gmm}_ali_${train_set}/tree | grep num-pdfs | awk '{print $2}')
state_prior_file=data/xent/state_prior.vec

if [ ${stage} -le 4 ]; then
  echo "Stage 4: Dump Json Files and Estimate Initial State Prior from Alignments"
  train_feat=$dumpdir/${train_set}_hires/feats.scp
  train_ali=data/${gmm}_ali_${train_set}/ali_pdf.scp
  train_text=data/${train_set}_hires/text
  train_utt2num_frames=data/${train_set}_hires/utt2num_frames
  valid_feat=$dumpdir/${valid_set}_hires/feats.scp
  valid_ali=data/${gmm}_ali_${valid_set}/ali_pdf.scp
  valid_text=data/${valid_set}_hires/text
  valid_utt2num_frames=data/${valid_set}_hires/utt2num_frames
  mkdir -p data/xent
  asr_prep_json.py --feat-files $train_feat --alignment-file $train_ali --text-files $train_text --utt2num-frames-files $train_utt2num_frames --output data/xent/train.json
  asr_prep_json.py --feat-files $valid_feat --alignment-file $valid_ali --text-files $valid_text --utt2num-frames-files $valid_utt2num_frames --output data/xent/valid.json
  for dataset in $test_set; do
    nj=$(wc -l <data/${dataset}_hires/spk2utt)
    utils/split_data.sh data/${dataset}_hires $nj
    utils/split_data.sh $dumpdir/${dataset}_hires $nj
    for n in $(seq $nj); do
      feat=$dumpdir/${dataset}_hires/split$nj/$n/feats.scp
      text=data/${dataset}_hires/split$nj/$n/text
      utt2num_frames=data/${dataset}_hires/split$nj/$n/utt2num_frames
      asr_prep_json.py --feat-files $feat --text-files $text --utt2num-frames-files $utt2num_frames --output data/xent/$dataset.$n.json
    done
  done
  
  estimate_initial_state_prior_from_alignments.py --alignment-files $train_ali --prior-dim $num_targets \
    --prior-floor 5e-6 --output $state_prior_file
fi

[ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu -n $ngpus) || \
  echo "Unable to get $ngpus GPUs"
[ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
[ $(echo $free_gpu | sed 's/,/ /g' | awk '{print NF}') -ne "$ngpus" ] && \
  echo "number of GPU ids in --free-gpu=$free_gpu does not match --ngpus=$ngpus" && exit 1;

if [ ${stage} -le 5 ]; then
  echo "Stage 5: Model Training"
  valid_subset=valid
  mkdir -p $dir/log
  log_file=$dir/log/train.log
  [ -f $dir/checkpoint_last.pt ] && log_file="-a $log_file"
  CUDA_VISIBLE_DEVICES=$free_gpu speech_train.py data/xent --task speech_recognition_hybrid --seed 1 --user-dir espresso \
    --log-interval $((100/ngpus)) --log-format simple --num-workers 0 --max-tokens 160000 --max-sentences 256 \
    --valid-subset $valid_subset --max-sentences-valid 256 --ddp-backend no_c10d \
    --distributed-world-size $ngpus --distributed-port $(if [ $ngpus -gt 1 ]; then echo 100; else echo -1; fi) \
    --max-epoch 40 --optimizer adam --lr 0.001 --weight-decay 0.0 \
    --lr-scheduler reduce_lr_on_plateau_v2 --lr-shrink 0.5 \
    --save-dir $dir --restore-file checkpoint_last.pt --save-interval-updates $((200/ngpus)) \
    --keep-interval-updates 5 --keep-last-epochs 5 --validate-interval 1 \
    --arch speech_tdnn_wsj --criterion subsampled_cross_entropy_with_accuracy --num-targets $num_targets \
    --initial-state-prior-file $state_prior_file --state-prior-update-interval 10 --state-prior-update-smoothing 0.01 \
    --chunk-width 150 --chunk-left-context 10 --chunk-right-context 10 --label-delay -3 \
    --max-source-positions 9999 --max-target-positions 9999 2>&1 | tee $log_file
fi

if [ ${stage} -le 6 ]; then
  echo "Stage 6: Decoding"
  rm $dir/.error 2>/dev/null || true
  queue_opt="--num-threads 4"
  path=$dir/$checkpoint
  for dataset in $test_set; do
    (
      data_affix=$(echo $dataset | sed s/test_//)
      nj=$(wc -l <data/${dataset}_hires/spk2utt)
      for lmtype in tgpr bd_tgpr; do
        graph_dir=exp/$gmm/graph_${lmtype}
        $decode_cmd $queue_opt JOB=1:$nj $dir/decode_${lmtype}_${data_affix}/log/decode.JOB.log \
          dump_posteriors.py data/xent --cpu --task speech_recognition_hybrid --user-dir espresso \
            --max-tokens 256000 --max-sentences 256 --num-shards 1 --shard-id 0 --num-targets $num_targets \
            --gen-subset $dataset.JOB --chunk-width 150 --chunk-left-context 10 --chunk-right-context 10 --label-delay -3 \
            --max-source-positions 9999 --path $path --apply-log-softmax \| \
          latgen-faster-mapped --max-active=7000 --min-active=20 --beam=15 --lattice-beam=8 --acoustic-scale=0.1 \
            --allow-partial=true --word-symbol-table="$graph_dir/words.txt" \
            exp/$gmm/final.mdl $graph_dir/HCLG.fst ark:- "ark:|gzip -c >$dir/decode_${lmtype}_${data_affix}/lat.JOB.gz" || exit 1
        local/score.sh --cmd "$decode_cmd" data/${dataset}_hires $graph_dir $dir/decode_${lmtype}_${data_affix} || exit 1
        echo $nj > $dir/decode_${lmtype}_${data_affix}/num_jobs
      done
      steps/lmrescore.sh --cmd "$decode_cmd" --mode 3 ${lang}_{tgpr,tg} \
        data/${dataset}_hires $dir/decode_{tgpr,tg}_${data_affix} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" ${lang}_bd_{tgpr,fgconst} \
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

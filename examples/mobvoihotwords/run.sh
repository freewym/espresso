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

dir=exp/tdnn_k2${affix:+_$affix}

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
FREETEXT 1.0 freetext
HiXiaowen 1.0 hixiaowen
NihaoWenwen 1.0 nihaowenwen
<sil> 1.0 SIL
EOF

  utils/lang/make_lexicon_fst.py --sil-prob 0.5 --sil-phone SIL --sil-disambig '#1' \
    data/lang/lexiconp.txt > data/lang/L_disambig.fst.sym
  cat <(head -n -1 data/lang/L_disambig.fst.sym) <(echo -e "1\t1\t#0\t#0") <(tail -n 1 data/lang/L_disambig.fst.sym) \
    > data/lang/L_disambig.fst.sym.temp
  cat data/lang/L_disambig.fst.sym.temp > data/lang/L_disambig.fst.sym
  rm -f data/lang/L_disambig.fst.sym.temp

  echo "Prepare phones symbol table"
  cat > data/lang/phones.txt <<EOF
<eps> 0
SIL 1
freetext 2
hixiaowen 3
nihaowenwen 4
#0 5
#1 6
EOF

  echo "Prepare words symbol table"
  cat > data/lang/words.txt <<EOF
<eps> 0
<sil> 1
FREETEXT 2
HiXiaowen 3
NihaoWenwen 4
#0 5
EOF

  utils/sym2int.pl -f 3 data/lang/phones.txt <data/lang/L_disambig.fst.sym - | \
    utils/sym2int.pl -f 4 data/lang/words.txt - > data/lang/L_disambig.fst.txt

  echo "Prepare HMMs for phones"
  id_sil=`cat data/lang/phones.txt | grep "SIL" | awk '{print $2}'`
  id_freetext=`cat data/lang/phones.txt | grep "freetext" | awk '{print $2}'`
  id_word0=`cat data/lang/phones.txt | grep "hixiaowen" | awk '{print $2}'`
  id_word1=`cat data/lang/phones.txt | grep "nihaowenwen" | awk '{print $2}'`

   cat > data/lang/hmm_sil.fst.txt <<EOF
0 0 0 0 0.693147181
0 1 1 $id_sil 0.693147181
1
EOF

  cat > data/lang/hmm_freetext.fst.txt <<EOF
0 0 2 0 0.693147181
0 1 3 $id_freetext 0.693147181
1 1 4 0 0.693147181
1 2 5 0 0.693147181
2 2 6 0 0.693147181
2 3 7 0 0.693147181
3 3 8 0 0.693147181
3 4 9 0 0.693147181
4
EOF

  cat > data/lang/hmm_hixiaowen.fst.txt <<EOF
0 0 10 0 0.693147181
0 1 11 $id_word0 0.693147181
1 1 12 0 0.693147181
1 2 13 0 0.693147181
2 2 14 0 0.693147181
2 3 15 0 0.693147181
3 3 16 0 0.693147181
3 4 17 0 0.693147181
4
EOF

  cat > data/lang/hmm_nihaowenwen.fst.txt <<EOF
0 0 18 0 0.693147181
0 1 19 $id_word1 0.693147181
1 1 20 0 0.693147181
1 2 21 0 0.693147181
2 2 22 0 0.693147181
2 3 23 0 0.693147181
3 3 24 0 0.693147181
3 4 25 0 0.693147181
4
EOF

  echo "Prepare an unnormalized phone language model for the denominator graph"
  cat > data/lang/phone_lm.fsa.txt <<EOF
0 1 $id_sil
0 7 $id_sil
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
  log_file=data/log/generate_graphs.log
  $train_cmd $log_file local/create_H_and_denominator.py --hmm-paths data/lang/hmm_{sil,freetext,hixiaowen,nihaowenwen}.fst.txt \
    --phone-lm-fsa-path data/lang/phone_lm.fsa.txt --out-dir data
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
  valid_subset=valid
  mkdir -p $dir/log
  log_file=$dir/log/train.log
  [ -f $dir/checkpoint_last.pt ] && log_file="-a $log_file"
  update_freq=1
  $cuda_cmd $log_file speech_train.py data --task speech_recognition_hybrid --seed 1 \
    --log-interval $((1500/ngpus/update_freq)) --log-format simple --use-k2-dataset \
    --num-workers 0 --data-buffer-size 0 --max-tokens 25600 --batch-size 128 --empty-cache-freq 50 \
    --valid-subset $valid_subset --batch-size-valid 128 --ddp-backend no_c10d --update-freq $update_freq \
    --distributed-world-size $ngpus --arch speech_tdnn_mobvoi \
    --max-epoch 15 --optimizer adam --lr 0.001 --weight-decay 0.0 \
    --lr-scheduler reduce_lr_on_plateau_v2 --lr-shrink 0.5 \
    --save-dir $dir --restore-file checkpoint_last.pt --save-interval-updates $((1500/ngpus/update_freq)) \
    --keep-interval-updates 5 --keep-last-epochs 5 --validate-interval 1 \
    --criterion k2_lattice_free_mmi --num-targets $num_targets --word-symbol-table-path data/lang/words.txt \
    --phone-symbol-table-path data/lang/phones.txt --denominator-graph-path data/denominator.pt \
    --H-path data/H.pt --L-path data/lang/L_disambig.fst.txt \
    --max-source-positions 9999 --max-target-positions 9999 $opts || exit 1;
fi

if [ ${stage} -le 3 ]; then
  echo "Stage 3: Dump Posteriors for Evaluation"
  path=$dir/$checkpoint
  for dataset in $test_set; do
    mkdir -p $dir/decode_$dataset/log
    log_file=$dir/decode_$dataset/log/dump_posteriors.log
    $cuda_cmd $log_file dump_posteriors.py data --use-k2-dataset \
      --task speech_recognition_hybrid --max-tokens 25600 --max-sentences 128 \
      --num-shards 1 --shard-id 0 --num-targets $num_targets --gen-subset $dataset \
      --max-source-positions 9999 --max-target-positions 9999 --path $path \
    \| copy-matrix ark:- ark,scp:$dir/decode_$dataset/posteriors.ark,$dir/decode_$dataset/posteriors.scp || exit 1;
    echo "log saved in $log_file"
  done
fi

if [ ${stage} -le 4 ]; then
  echo "Stage 4: Decoding"
  lang_test=data/lang_test
  rm -rf $lang_test
  cp -r data/lang $lang_test
  utils/lang/make_lexicon_fst.py --sil-prob 0.0 --sil-phone SIL --sil-disambig '#1' \
    $lang_test/lexiconp.txt > $lang_test/L_disambig.fst.sym
  cat <(head -n -1 $lang_test/L_disambig.fst.sym) <(echo -e "0\t0\t#0\t#0") <(tail -n 1 $lang_test/L_disambig.fst.sym) \
    > $lang_test/L_disambig.fst.sym.temp
  cat $lang_test/L_disambig.fst.sym.temp > $lang_test/L_disambig.fst.sym
  rm -f $lang_test/L_disambig.fst.sym.temp

  utils/sym2int.pl -f 3 $lang_test/phones.txt <$lang_test/L_disambig.fst.sym - | \
    utils/sym2int.pl -f 4 $lang_test/words.txt - > $lang_test/L_disambig.fst.txt

  for wake_word in $wake_word0 $wake_word1; do
    if [[ "$wake_word" == "$wake_word0" ]]; then
      wake_word0_cost_range="-1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0"
      wake_word1_cost_range="0.0"
    else
      wake_word0_cost_range="0.0"
      wake_word1_cost_range="-1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0"
    fi
    for wake_word0_cost in $wake_word0_cost_range; do
      for wake_word1_cost in $wake_word1_cost_range; do
        sil_id=`cat $lang_test/words.txt | grep "<sil>" | awk '{print $2}'`
        freetext_id=`cat $lang_test/words.txt | grep "FREETEXT" | awk '{print $2}'`
        id0=`cat $lang_test/words.txt | grep $wake_word0 | awk '{print $2}'`
        id1=`cat $lang_test/words.txt | grep $wake_word1 | awk '{print $2}'`
        mkdir -p $lang_test/lm
        cat <<EOF > $lang_test/lm/fsa.txt
0 1 $sil_id
0 4 $sil_id 7.0
1 4 $freetext_id 0.0
4 0 $sil_id
1 2 $id0 $wake_word0_cost
1 3 $id1 $wake_word1_cost
2 0 $sil_id
3 0 $sil_id
0
EOF
        local/create_decoding_graph.py --H-path data/H.pt --L-path $lang_test/L_disambig.fst.txt --G-path $lang_test/lm/fsa.txt \
          --first-phone-disambig-id 5 --first-word-disambig-id 5 $lang_test/graph || exit 1;

        rm $dir/.error 2>/dev/null || true
        for dataset in $test_set; do
          (
            score_dir=$dir/decode_$dataset/score_${wake_word}_${wake_word0_cost}_${wake_word1_cost}
            mkdir -p $score_dir
            $decode_cmd $dir/decode_$dataset/log/decode_${wake_word}.log \
              local/decode_best_path.py --beam=10 --word-symbol-table $lang_test/words.txt \
                $lang_test/graph/HCLG.pt $dir/decode_$dataset/posteriors.scp $score_dir/hyp.txt || exit 1;
            local/evaluate.py --wake-word $wake_word \
              data/supervisions_${dataset}.json $score_dir/hyp.txt $score_dir/metrics || exit 1;
          ) || touch $dir/.error &
        done
        wait
        [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
      done
    done
  done
  for dataset in $test_set; do
    for wake_word in $wake_word0 $wake_word1; do
      echo "Results on $dataset set with wake word ${wake_word}:"
      cat $dir/decode_$dataset/score_${wake_word}_*/metrics
    done
  done
fi

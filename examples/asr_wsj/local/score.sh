#!/bin/bash

# Copyright (c) 2019-present, Yiming Wang
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


# begin configuration section.
cmd=run.pl
#end configuration section.

echo "$0 $@"  # Print the command line for logging
[ -f ./path.sh ] && . ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 [--cmd (run.pl|queue.pl...)] <data-dir> decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  exit 1;
fi

data=$1
dir=$2


ref_filtering_cmd="cat"
[ -x local/wer_output_filter ] && ref_filtering_cmd="local/wer_output_filter"
[ -x local/wer_ref_filter ] && ref_filtering_cmd="local/wer_ref_filter"
hyp_filtering_cmd="cat"
[ -x local/wer_output_filter ] && hyp_filtering_cmd="local/wer_output_filter"
[ -x local/wer_hyp_filter ] && hyp_filtering_cmd="local/wer_hyp_filter"

mkdir -p $dir/scoring_kaldi/log
$ref_filtering_cmd $data/text > $dir/scoring_kaldi/test_filt.txt || exit 1;
$hyp_filtering_cmd $dir/decoded_results.txt > $dir/scoring_kaldi/hyp_filt.txt || exit 1;

$cmd $dir/scoring_kaldi/log/score.log \
  cat $dir/scoring_kaldi/hyp_filt.txt \| \
  compute-wer --text --mode=present \
  ark:$dir/scoring_kaldi/test_filt.txt ark,p:- ">&" $dir/scoring_kaldi/wer || exit 1;


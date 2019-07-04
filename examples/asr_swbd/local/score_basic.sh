#!/bin/bash
# Copyright (c) 2012,         Johns Hopkins University (Author: Daniel Povey)
#               2019-present, Yiming Wang
# Apache 2.0

# begin configuration section.
cmd=run.pl
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: local/score_basic.sh [--cmd (run.pl|queue.pl...)] <data-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  exit 1;
fi

data=$1
dir=$2

for f in $data/text $dir/decoded_results.txt; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done


function filter_text {
  perl -e 'foreach $w (@ARGV) { $bad{$w} = 1; }
   while(<STDIN>) { @A  = split(" ", $_); $id = shift @A; print "$id ";
     foreach $a (@A) { if (!defined $bad{$a}) { print "$a "; }} print "\n"; }' \
   '[noise]' '[laughter]' '[vocalized-noise]' '<unk>' '%hesitation'
}

mkdir -p $dir/scoring/log
filter_text <$data/text >$dir/scoring/test_filt.txt || exit 1;
filter_text <$dir/decoded_results.txt >$dir/scoring/hyp_filt.txt || exit 1;

$cmd $dir/scoring/log/score.log \
  compute-wer --text --mode=present \
  ark:$dir/scoring/test_filt.txt ark:$dir/scoring/hyp_filt.txt ">&" \
  $dir/scoring/wer || exit 1;

exit 0

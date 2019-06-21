#!/bin/bash
# Copyright 2019-present Hang Lyu

# begin configuration section.
cmd=run.pl
stage=0
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: local/score_sclite.sh [--cmd (run.pl|queue.pl...)] <data-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2|3)               # start scoring script from part-way through."
  exit 1;
fi

data=$1
dir=$2

hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

for f in $data/stm $data/glm $dir/decoded_results.txt; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

name=`basename $data`; # e.g. eval2000

mkdir -p $dir/scoring/log

if [ $stage -le 0 ]; then
  # prepare the $name.ctm files for test set
  python3 local/prepare_ctm.py $dir/decoded_results.txt $dir/scoring/$name.ctm
fi

if [ $stage -le 1 ]; then
  # Remove some stuff we don't want to score, from the ctm.
  # the big expression in parentheses contains all the things that get mapped
  # by the glm file, into hesitations.
  # The -$ expression removes partial words.
  # the aim here is to remove all the things that appear in the reference as optionally
  # deletable (inside parentheses), as if we delete these there is no loss, while
  # if we get them correct there is no gain.
  for x in $dir/scoring/$name.ctm; do
    cp $x $dir/scoring/tmpf;
    cat $dir/scoring/tmpf | grep -i -v -E '\[NOISE|LAUGHTER|VOCALIZED-NOISE\]' | \
    grep -i -v -E '<UNK>' | \
    grep -i -v -E ' (UH|UM|EH|MM|HM|AH|HUH|HA|ER|OOF|HEE|ACH|EEE|EW)$' | \
    grep -v -- '-$' > $x;
    python local/map_acronyms_ctm.py -i $x -o $x.mapped -M data/local/dict_nosp/acronyms.map
    cp $x $x.bk
    mv $x.mapped $x
  done
fi

# Score the set...
if [ $stage -le 2 ]; then
  $cmd $dir/scoring/log/score.log \
    cp $data/stm $dir/scoring/ '&&' \
    $hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/scoring/stm $dir/scoring/${name}.ctm || exit 1;
fi

# For eval2000 score the subsets
case "$name" in
  eval2000*)
    # Score only the, swbd part...
    if [ $stage -le 3 ]; then
      $cmd $dir/scoring/log/score.swbd.log \
        grep -v '^en_' $data/stm '>' $dir/scoring/stm.swbd '&&' \
        grep -v '^en_' $dir/scoring/${name}.ctm '>' $dir/scoring/${name}.ctm.swbd '&&' \
        $hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/scoring/stm.swbd $dir/scoring/${name}.ctm.swbd || exit 1;
    fi
    # Score only the, callhome part...
    if [ $stage -le 3 ]; then
      $cmd $dir/scoring/log/score.callhm.log \
        grep -v '^sw_' $data/stm '>' $dir/scoring/stm.callhm '&&' \
        grep -v '^sw_' $dir/scoring/${name}.ctm '>' $dir/scoring/${name}.ctm.callhm '&&' \
        $hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/scoring/stm.callhm $dir/scoring/${name}.ctm.callhm || exit 1;
    fi
    ;;
rt03* )
  # Score only the swbd part...
  if [ $stage -le 3 ]; then
    $cmd $dir/scoring/log/score.swbd.log \
      grep -v '^fsh_' $data/stm '>' $dir/scoring/stm.swbd '&&' \
      grep -v '^fsh_' $dir/scoring/${name}.ctm '>' $dir/scoring/${name}.ctm.swbd '&&' \
      $hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/scoring/stm.swbd $dir/scoring/${name}.ctm.swbd || exit 1;
  fi
  # Score only the fisher part...
  if [ $stage -le 3 ]; then
    $cmd $dir/scoring/log/score.fsh.log \
      grep -v '^sw_' $data/stm '>' $dir/scoring/stm.fsh '&&' \
      grep -v '^sw_' $dir/scoring/${name}.ctm '>' $dir/scoring/${name}.ctm.fsh '&&' \
      $hubscr -p $hubdir -V -l english -h hub5 -g $data/glm -r $dir/scoring/stm.fsh $dir/scoring/${name}.ctm.fsh || exit 1;
  fi
 ;;
esac

exit 0

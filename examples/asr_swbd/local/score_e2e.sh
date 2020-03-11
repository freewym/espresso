#!/bin/bash
# Copyright (c) 2012,         Johns Hopkins University (Author: Daniel Povey)
#               2019-present, Yiming Wang
# All rights reserved.


orig_args=
for x in "$@"; do orig_args="$orig_args '$x'"; done

# begin configuration section.  we include all the options that score_sclite.sh or
# score_basic.sh might need, or parse_options.sh will die.
# CAUTION: these default values do not have any effect because of the
# way pass things through to the scripts that this script calls.
cmd=run.pl
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: local/score.sh [options] <data-dir> <decode-dir>" && exit;
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  exit 1;
fi

data=$1

if [ -f $data/stm ]; then # use sclite scoring.
  echo "$data/stm exists: using local/score_sclite.sh"
  eval local/score_sclite_e2e.sh $orig_args
else
  echo "$data/stm does not exist: using local/score_basic.sh"
  eval local/score_basic_e2e.sh $orig_args
fi

#!/bin/bash
# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e -o pipefail

stage=0

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

if [ ${stage} -le 0 ]; then
  echo "Stage 0: Data Preparation"
  mkdir -p data/log
  ${train_cmd} data/log/data_prep.log \
    python3 local/data_prep.py --data-dir data --seed 1 --num-workers 16 \
      --max-remaining-duration 0.3 --overlap-duration 0.3
fi

if [ ${stage} -le 1 ]; then
  echo "Stage 1: Graph Generation"
fi

if [ ${stage} -le 2 ]; then
  echo "Stage 2: Model Training"
fi

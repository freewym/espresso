MAIN_ROOT=$PWD/../..
KALDI_ROOT=$MAIN_ROOT/speech_tools/kaldi

# BEGIN from kaldi path.sh
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
# END

export PATH=~/anaconda3/bin:$PATH
export PATH=$MAIN_ROOT:$MAIN_ROOT/speech_tools:$PATH
export PATH=$MAIN_ROOT/tools/sentencepiece/build/src:$PATH
export PYTHONPATH=$MAIN_ROOT:$MAIN_ROOT/speech_tools:$PYTHONPATH
export PYTHONUNBUFFERED=1


KALDI =

.PHONY: all clean

all: kaldi

ifneq ($(strip $(KALDI)),)
kaldi:
		ln -s $(KALDI) kaldi
else
kaldi:
		test -d kaldi || git clone https://github.com/kaldi-asr/kaldi.git
		cd kaldi/tools; $(MAKE) all
		cd kaldi/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
endif

clean:
		rm -rf kaldi

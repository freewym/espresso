KALDI =

.PHONY: all clean

all: kaldi

ifneq ($(strip $(KALDI)),)
kaldi:
		ln -s $(KALDI) kaldi
else
kaldi:
		# git clone https://github.com/kaldi-asr/kaldi.git kaldi_github; cd kaldi_github/tools; $(MAKE) all
		# cd kaldi_github/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
		# ln -nfs kaldi_github kaldi
endif

clean:
		rm -fr kaldi

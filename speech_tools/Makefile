KALDI =

.PHONY: all clean

all: kaldi kaldi-io-for-python

kaldi-io-for-python:
	git clone https://github.com/vesis84/kaldi-io-for-python.git
	ln -sf kaldi-io-for-python/kaldi_io/kaldi_io.py kaldi_io.py

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
		rm -fr kaldi kaldi-io-for-python kaldi_io.py

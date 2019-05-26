KALDI =

.PHONY: all clean

all: kaldi kaldi-io-for-python sentencepiece

kaldi-io-for-python:
	rm -rf kaldi-io-for-python
	git clone https://github.com/vesis84/kaldi-io-for-python.git
	ln -sf kaldi-io-for-python/kaldi_io/kaldi_io.py kaldi_io.py

sentencepiece:
	rm -rf sentencepiece
	git clone https://github.com/google/sentencepiece.git
	cd sentencepiece && mkdir build && cd build && (cmake3 .. || cmake ..) && $(MAKE)

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
		rm -rf kaldi kaldi-io-for-python kaldi_io.py sentencepiece

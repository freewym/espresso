KALDI =
PYTHON_DIR = ~/anaconda3/bin

CXX ?= g++

WGET ?= wget

# Note: OpenFst requires a relatively recent C++ compiler with C++11 support,
# e.g. g++ >= 4.7, Apple clang >= 5.0 or LLVM clang >= 3.3.
OPENFST_VERSION ?= 1.7.5

# Default features configured for OpenFST; can be overridden in the make command line.
OPENFST_COMFIGURE ?= --enable-static --enable-shared --enable-ngram-fsts

CPPFLAGS ?= -D_GLIBCXX_USE_CXX11_ABI=0
CXXFLAGS ?= -D_GLIBCXX_USE_CXX11_ABI=0

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

clean: openfst_cleaned
	rm -rf pychain
	rm -rf kaldi

openfst_cleaned:
	$(MAKE) -C openfst-$(OPENFST_VERSION) clean

.PHONY: openfst # so target will be made even though "openfst" exists.
openfst: openfst_compiled openfst-$(OPENFST_VERSION)/lib
	-rm -f openfst
	-ln -s openfst-$(OPENFST_VERSION) openfst

.PHONY: openfst_compiled
openfst_compiled: openfst-$(OPENFST_VERSION)/Makefile
	$(MAKE) -C openfst-$(OPENFST_VERSION) install MAKEOVERRIDES=

openfst-$(OPENFST_VERSION)/lib: | openfst-$(OPENFST_VERSION)/Makefile
	-cd openfst-$(OPENFST_VERSION) && [ -d lib64 ] && [ ! -d lib ] && ln -s lib64 lib

# Add the -O flag to CXXFLAGS on cygwin as it can fix the compilation error
# "file too big".
ifeq ($(OSTYPE),cygwin)
  # Note: OSTYPE path is probably dead for latest cygwin64 (installed on 2016/11/11).
  openfst_add_CXXFLAGS = -O -Wa,-mbig-obj
else ifeq ($(OS),Windows_NT)
  # This new OS path is confirmed working on Windows 10 / Cygwin64.
  openfst_add_CXXFLAGS = -O -Wa,-mbig-obj
else
  openfst_add_CXXFLAGS =
endif

openfst-$(OPENFST_VERSION)/Makefile: openfst-$(OPENFST_VERSION)
	cd openfst-$(OPENFST_VERSION)/ && \
	 ./configure --prefix=`pwd` $(OPENFST_CONFIGURE) CXX="$(CXX)" CPPFLAGS="$(CPPFLAGS)" CXXFLAGS="$(CXXFLAGS) $(openfst_add_CXXFLAGS)" LDFLAGS="$(LDFLAGS)" LIBS="-ldl"

openfst-$(OPENFST_VERSION): openfst-$(OPENFST_VERSION).tar.gz
	tar xozf openfst-$(OPENFST_VERSION).tar.gz

openfst-$(OPENFST_VERSION).tar.gz:
	$(WGET) -T 10 -t 1 http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-$(OPENFST_VERSION).tar.gz || \
	$(WGET) -T 10 -t 3 https://www.openslr.org/resources/2/openfst-$(OPENFST_VERSION).tar.gz;

.PHONY: pychain
pychain:
	test -d pychain || git clone https://github.com/YiwenShaoStephen/pychain.git
	export OPENFST_PATH=`pwd`/openfst && \
	export LD_LIBRARY_PATH=`pwd`/openfst/lib:$$LD_LIBRARY_PATH && \
	export PATH=$(PYTHON_DIR):$$PATH && \
	cd pychain/openfst_binding && python3 setup.py install && \
	cd ../pytorch_binding && python3 setup.py install

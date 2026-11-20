CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra

BINS := bin/gainmerge bin/make_test bin/check_master

all: $(BINS)

bin/%: src/%.cpp src/wav.h src/dsp.h | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin:
	mkdir -p bin

clean:
	rm -rf bin

test: all
	@mkdir -p out
	bin/make_test out/test_stereo.wav out/truth.wav
	bin/gainmerge out/test_stereo.wav out/master.wav
	bin/check_master out/master.wav out/truth.wav out/test_stereo.wav \
		out/master.wav.report.txt

.PHONY: all clean test

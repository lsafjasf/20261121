JROOT ?= $(CURDIR)/.jack/root
JLIB := $(JROOT)/usr/lib/x86_64-linux-gnu

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=gnu11
JACK_CFLAGS := -I$(JROOT)/usr/include
JACK_LIBS := -L$(JLIB) -Wl,-rpath,$(JLIB) -ljack -lpthread -lm

BINS := build/fade_router build/virtual_device build/recorder

all: $(BINS)

build/%: src/%.c | build
	$(CC) $(CFLAGS) $(JACK_CFLAGS) -o $@ $< $(JACK_LIBS)

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean

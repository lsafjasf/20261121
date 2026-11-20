CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Iinclude -Isrc
BUILD_DIR := build

.PHONY: all test artifacts clean

all: $(BUILD_DIR)/peak_limiter_offline $(BUILD_DIR)/limiter_tests

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/peak_limiter_offline: src/offline_main.cpp src/TestSupport.hpp src/Signals.hpp include/TruePeakLimiter.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) src/offline_main.cpp -o $@

$(BUILD_DIR)/limiter_tests: tests/core_tests.cpp src/TestSupport.hpp src/Signals.hpp include/TruePeakLimiter.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/core_tests.cpp -o $@

test: $(BUILD_DIR)/limiter_tests
	$(BUILD_DIR)/limiter_tests

artifacts: $(BUILD_DIR)/peak_limiter_offline
	$(BUILD_DIR)/peak_limiter_offline --output-dir artifacts

clean:
	rm -rf $(BUILD_DIR)

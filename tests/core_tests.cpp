#include "Signals.hpp"
#include "TestSupport.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using tpltest::LimitedResult;
using tpltest::StereoBuffer;

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void checkClose(double actual, double expected, double tolerance,
                const std::string& message) {
    check(std::abs(actual - expected) <= tolerance,
          message + " (actual=" + tpltest::fmtFixed(actual, 8) +
              ", expected=" + tpltest::fmtFixed(expected, 8) + ")");
}

void testConfigurationAndLatency() {
    tpl::PeakLimiterConfig config;
    config.lookaheadMs = 3.0;
    config.detectorHalfWidth = 10;
    for (double sampleRate : {44100.0, 48000.0, 96000.0}) {
        config.sampleRate = sampleRate;
        tpl::TruePeakLimiter limiter(config);
        const int expectedLookahead =
            std::max(1, static_cast<int>(std::lround(sampleRate * 0.003)));
        check(limiter.lookaheadSamples() == expectedLookahead,
              "lookahead samples must follow sample rate");
        check(limiter.latencySamples() == expectedLookahead + 10,
              "latency must include detector group delay");
        checkClose(limiter.latencyMs(),
                   1000.0 * limiter.latencySamples() / sampleRate, 1.0e-9,
                   "latency milliseconds");
    }

    config.sampleRate = 48000.0;
    tpl::TruePeakLimiter limiter(config);
    config.oversampleFactor = 3;
    check(!limiter.configure(config), "unsupported oversampling must be rejected");
}

void testBypassIsExactlyDelayed() {
    constexpr double sampleRate = 48000.0;
    const StereoBuffer input = tpltest::makeAlignmentProbe(sampleRate, 0.3);
    tpl::PeakLimiterConfig config;
    config.sampleRate = sampleRate;
    config.bypass = true;

    const LimitedResult bypassed = tpltest::runLimiter(input, config);
    tpl::TruePeakLimiter reference(config);
    const int latency = reference.latencySamples();
    check(latency == std::max(1, static_cast<int>(std::lround(48000.0 * 0.003))) +
                         reference.detectorDelaySamples(),
          "48 kHz latency must be lookahead plus detector delay");
    const StereoBuffer expected = tpltest::delayedCopy(input, latency);
    const double difference = tpltest::maxAbsDifference(bypassed.audio, expected);
    checkClose(difference, 0.0, 1.0e-7, "bypass must equal input delayed by latency");
    for (int i = 0; i < latency; ++i) {
        check(bypassed.audio.left[static_cast<std::size_t>(i)] == 0.0f &&
                  bypassed.audio.right[static_cast<std::size_t>(i)] == 0.0f,
              "bypass must emit silence during latency");
    }
}

void testChunkingIsConsistent() {
    constexpr double sampleRate = 48000.0;
    const StereoBuffer input = tpltest::makeNearNyquistProbe(sampleRate, 0.35);
    tpl::PeakLimiterConfig config;
    config.sampleRate = sampleRate;

    const LimitedResult oneShot = tpltest::runLimiter(input, config);

    tpl::TruePeakLimiter limiter(config);
    StereoBuffer chunked{input.left, input.right};
    std::vector<float> gain(input.frames(), 1.0f);
    const std::vector<std::size_t> chunks = {1, 7, 64, 13, 256, 5, 511};
    std::size_t position = 0;
    std::size_t chunkIndex = 0;
    while (position < chunked.frames()) {
        const std::size_t count = std::min(
            chunks[chunkIndex % chunks.size()], chunked.frames() - position);
        limiter.process(chunked.left.data() + position,
                        chunked.right.data() + position, count,
                        gain.data() + position);
        position += count;
        ++chunkIndex;
    }

    checkClose(tpltest::maxAbsDifference(chunked, oneShot.audio), 0.0, 1.0e-7,
               "block chunking must not change output");
    for (std::size_t i = 0; i < gain.size(); ++i) {
        checkClose(gain[i], oneShot.gain[i], 1.0e-7,
                   "block chunking must not change gain telemetry");
    }
}

void testStereoLinkage() {
    constexpr double sampleRate = 48000.0;
    const StereoBuffer input = tpltest::makeAlternatingHits(sampleRate, 1.5);
    tpl::PeakLimiterConfig config;
    config.sampleRate = sampleRate;
    config.ceilingDbTP = -1.0;

    const LimitedResult limited = tpltest::runLimiter(input, config);
    tpl::TruePeakLimiter reference(config);
    const int latency = reference.latencySamples();
    double minimumGain = 1.0;
    double maximumChannelGainDeltaDb = 0.0;
    for (std::size_t i = static_cast<std::size_t>(latency); i < input.frames(); ++i) {
        minimumGain = std::min(minimumGain, static_cast<double>(limited.gain[i]));
        const double inL = input.left[i - static_cast<std::size_t>(latency)];
        const double inR = input.right[i - static_cast<std::size_t>(latency)];
        const double outL = limited.audio.left[i];
        const double outR = limited.audio.right[i];
        if (std::abs(inL) > 2.0e-3 && std::abs(inR) > 2.0e-3) {
            const double gainL = outL / inL;
            const double gainR = outR / inR;
            maximumChannelGainDeltaDb =
                std::max(maximumChannelGainDeltaDb,
                         std::abs(20.0 * std::log10(std::max(gainL, 1.0e-9) /
                                                    std::max(gainR, 1.0e-9))));
        }
    }
    check(minimumGain < 0.55, "alternating hits must engage limiting");
    check(maximumChannelGainDeltaDb < 0.01,
          "both channels must receive the same linked gain");
}

void testTruePeakCeiling() {
    const tpltest::TruePeakAnalyzer analyzer(16, 64);
    for (int oversampling : {4, 8}) {
        for (double sampleRate : {44100.0, 48000.0, 96000.0}) {
            const StereoBuffer input = tpltest::makeNearNyquistProbe(sampleRate, 0.7);
            if (oversampling == 8) {
                check(analyzer.measureDb(input) >
                          tpltest::linearToDb(tpltest::samplePeak(input)) + 1.5,
                      "near-Nyquist probe must expose inter-sample peak");
            }
            tpl::PeakLimiterConfig config;
            config.sampleRate = sampleRate;
            config.ceilingDbTP = -1.0;
            config.oversampleFactor = oversampling;
            const LimitedResult limited = tpltest::runLimiter(input, config);
            const double outputPeakDb = analyzer.measureDb(limited.audio);
            check(outputPeakDb <= -1.0 + 0.02,
                  "near-Nyquist output must stay below ceiling at " +
                      tpltest::fmtFixed(sampleRate, 0) + " Hz, " +
                      std::to_string(oversampling) + "x (measured " +
                      tpltest::fmtFixed(outputPeakDb, 4) + " dBTP)");
        }
    }
}

void testHighFrequencyPhaseSweep() {
    constexpr double sampleRate = 48000.0;
    const tpltest::TruePeakAnalyzer analyzer(16, 64);
    for (double normalizedFrequency : {0.40, 0.46, 0.48, 0.49}) {
        for (int phaseIndex = 0; phaseIndex < 8; ++phaseIndex) {
            const double phase = 2.0 * tpltest::kPi * phaseIndex / 8.0;
            const std::size_t frames = tpltest::framesFor(sampleRate, 0.12);
            StereoBuffer input{std::vector<float>(frames),
                               std::vector<float>(frames)};
            const double fadeFrames = 0.01 * sampleRate;
            for (std::size_t i = 0; i < frames; ++i) {
                const double envelope = std::min(
                    1.0, std::min(static_cast<double>(i) / fadeFrames,
                                  static_cast<double>(frames - 1 - i) / fadeFrames));
                const double value = 1.6 * envelope * std::sin(
                    2.0 * tpltest::kPi * normalizedFrequency *
                        static_cast<double>(i) + phase);
                input.left[i] = static_cast<float>(value);
                input.right[i] = static_cast<float>(value);
            }
            tpl::PeakLimiterConfig config;
            config.sampleRate = sampleRate;
            config.ceilingDbTP = -1.0;
            const LimitedResult limited = tpltest::runLimiter(input, config);
            const double outputPeakDb = analyzer.measureDb(limited.audio);
            check(outputPeakDb <= -1.0 + 0.02,
                  "phase sweep output must stay below ceiling at f/fs=" +
                      tpltest::fmtFixed(normalizedFrequency, 3) +
                      ", phase=" + tpltest::fmtFixed(phase, 3) +
                      " (measured " + tpltest::fmtFixed(outputPeakDb, 4) +
                      " dBTP)");
        }
    }
}

void testSampleRateRebuild() {
    constexpr double firstRate = 44100.0;
    constexpr double secondRate = 48000.0;
    const StereoBuffer input = tpltest::makeNearNyquistProbe(secondRate, 0.4);
    tpl::PeakLimiterConfig config;
    config.sampleRate = firstRate;
    config.ceilingDbTP = -1.0;

    tpl::TruePeakLimiter switched(config);
    StereoBuffer warmup{input.left, input.right};
    std::vector<float> warmupGain(input.frames(), 1.0f);
    switched.process(warmup.left.data(), warmup.right.data(), warmup.frames(),
                     warmupGain.data());
    check(switched.currentGain() < 0.99, "warm-up rate must engage the limiter");

    check(switched.setSampleRate(secondRate), "sample-rate switch must be accepted");
    checkClose(switched.currentGain(), 1.0, 1.0e-9,
               "sample-rate switch must reset gain state");
    check(switched.latencySamples() ==
              std::max(1, static_cast<int>(std::lround(secondRate * 0.003))) +
                  switched.detectorDelaySamples(),
          "sample-rate switch must rebuild delay state");

    StereoBuffer switchedAudio{input.left, input.right};
    std::vector<float> switchedGain(input.frames(), 1.0f);
    switched.process(switchedAudio.left.data(), switchedAudio.right.data(),
                     switchedAudio.frames(), switchedGain.data());

    config.sampleRate = secondRate;
    const LimitedResult fresh = tpltest::runLimiter(input, config);
    checkClose(tpltest::maxAbsDifference(switchedAudio, fresh.audio), 0.0, 1.0e-7,
               "state after sample-rate switch must match a fresh instance");
    for (std::size_t i = 0; i < switchedGain.size(); ++i) {
        checkClose(switchedGain[i], fresh.gain[i], 1.0e-7,
                   "gain after sample-rate switch must match a fresh instance");
    }
}

}  // namespace

int main() {
    testConfigurationAndLatency();
    testBypassIsExactlyDelayed();
    testChunkingIsConsistent();
    testStereoLinkage();
    testTruePeakCeiling();
    testHighFrequencyPhaseSweep();
    testSampleRateRebuild();

    if (failures == 0) {
        std::cout << "All peak-limiter tests passed.\n";
        return EXIT_SUCCESS;
    }
    std::cerr << failures << " peak-limiter test(s) failed.\n";
    return EXIT_FAILURE;
}

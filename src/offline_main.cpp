#include "Signals.hpp"
#include "TestSupport.hpp"
#include "TruePeakLimiter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tpltest::LimitedResult;
using tpltest::StereoBuffer;
using tpltest::TruePeakAnalyzer;

constexpr double kCeilingDbTP = -1.0;
constexpr double kCeilingToleranceDb = 0.01;

struct SignalMetrics {
    double sampleRate = 0.0;
    std::string name;
    std::size_t frames = 0;
    int lookaheadSamples = 0;
    int detectorDelaySamples = 0;
    int latencySamples = 0;
    double inputSamplePeakDb = 0.0;
    double inputTruePeakDb = 0.0;
    double outputSamplePeakDb = 0.0;
    double outputTruePeakDb = 0.0;
    double outputMarginDb = 0.0;
    double maxGainReductionDb = 0.0;
    double maxStereoGainDeltaDb = 0.0;
    bool ceilingPass = false;
};

struct BypassMetrics {
    double sampleRate = 0.0;
    int reportedLatency = 0;
    int measuredLatency = 0;
    double maxAlignmentError = 0.0;
    bool exact = false;
};

struct RateSwitchMetrics {
    double fromRate = 0.0;
    double toRate = 0.0;
    int newLatency = 0;
    double maxOutputError = 0.0;
    double maxGainError = 0.0;
    bool pass = false;
};

struct AppOptions {
    fs::path outputDir = "artifacts";
    bool writeWav = true;
};

AppOptions parseArguments(int argc, char** argv) {
    AppOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--output-dir" && i + 1 < argc) {
            options.outputDir = argv[++i];
        } else if (argument == "--no-wav") {
            options.writeWav = false;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: peak_limiter_offline [--output-dir DIR] [--no-wav]\n"
                << "Writes limiter_report.json, limiter_metrics.csv and audition WAVs.\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    return options;
}

double maxChannelGainDeltaDb(const StereoBuffer& input,
                             const LimitedResult& result,
                             int latency) {
    double maximum = 0.0;
    for (std::size_t i = static_cast<std::size_t>(latency);
         i < input.frames(); ++i) {
        const double inL = input.left[i - static_cast<std::size_t>(latency)];
        const double inR = input.right[i - static_cast<std::size_t>(latency)];
        if (std::abs(inL) < 2.0e-3 || std::abs(inR) < 2.0e-3) {
            continue;
        }
        const double gainL = result.audio.left[i] / inL;
        const double gainR = result.audio.right[i] / inR;
        maximum = std::max(
            maximum,
            std::abs(20.0 * std::log10(std::max(gainL, 1.0e-12) /
                                       std::max(gainR, 1.0e-12))));
    }
    return maximum;
}

SignalMetrics evaluateSignal(const StereoBuffer& input, double sampleRate,
                             const std::string& name,
                             const TruePeakAnalyzer& analyzer) {
    tpl::PeakLimiterConfig config;
    config.sampleRate = sampleRate;
    config.ceilingDbTP = kCeilingDbTP;
    const LimitedResult limited = tpltest::runLimiter(input, config);
    tpl::TruePeakLimiter reference(config);

    SignalMetrics metrics;
    metrics.sampleRate = sampleRate;
    metrics.name = name;
    metrics.frames = input.frames();
    metrics.lookaheadSamples = reference.lookaheadSamples();
    metrics.detectorDelaySamples = reference.detectorDelaySamples();
    metrics.latencySamples = reference.latencySamples();
    metrics.inputSamplePeakDb = tpltest::linearToDb(tpltest::samplePeak(input));
    metrics.inputTruePeakDb = analyzer.measureDb(input);
    metrics.outputSamplePeakDb = tpltest::linearToDb(tpltest::samplePeak(limited.audio));
    metrics.outputTruePeakDb = analyzer.measureDb(limited.audio);
    metrics.outputMarginDb = kCeilingDbTP - metrics.outputTruePeakDb;
    metrics.ceilingPass = metrics.outputTruePeakDb <=
                          kCeilingDbTP + kCeilingToleranceDb;

    double minimumGain = 1.0;
    for (float gain : limited.gain) {
        minimumGain = std::min(minimumGain, static_cast<double>(gain));
    }
    metrics.maxGainReductionDb =
        -20.0 * std::log10(std::max(minimumGain, 1.0e-12));
    metrics.maxStereoGainDeltaDb =
        maxChannelGainDeltaDb(input, limited, metrics.latencySamples);
    return metrics;
}

BypassMetrics evaluateBypass(double sampleRate) {
    const StereoBuffer input = tpltest::makeAlignmentProbe(sampleRate, 0.35);
    tpl::PeakLimiterConfig config;
    config.sampleRate = sampleRate;
    config.bypass = true;
    tpl::TruePeakLimiter limiter(config);
    StereoBuffer processed{input.left, input.right};
    limiter.process(processed.left.data(), processed.right.data(), processed.frames());

    const int latency = limiter.latencySamples();
    const StereoBuffer expected = tpltest::delayedCopy(input, latency);
    BypassMetrics metrics;
    metrics.sampleRate = sampleRate;
    metrics.reportedLatency = latency;
    metrics.measuredLatency = -1;
    metrics.maxAlignmentError =
        tpltest::maxAbsDifference(processed, expected);
    metrics.exact = metrics.maxAlignmentError <= 1.0e-7;

    for (std::size_t i = 0; i < processed.frames(); ++i) {
        if (processed.left[i] != 0.0f || processed.right[i] != 0.0f) {
            metrics.measuredLatency = static_cast<int>(i);
            break;
        }
    }
    if (metrics.measuredLatency != latency) {
        metrics.exact = false;
    }
    return metrics;
}

RateSwitchMetrics evaluateRateSwitch(double fromRate, double toRate) {
    const StereoBuffer warmup = tpltest::makeNearNyquistProbe(fromRate, 0.30);
    const StereoBuffer freshInput = tpltest::makeNearNyquistProbe(toRate, 0.45);

    tpl::PeakLimiterConfig config;
    config.sampleRate = fromRate;
    config.ceilingDbTP = kCeilingDbTP;
    tpl::TruePeakLimiter switched(config);
    StereoBuffer warmupAudio{warmup.left, warmup.right};
    std::vector<float> warmupGain(warmup.frames(), 1.0f);
    switched.process(warmupAudio.left.data(), warmupAudio.right.data(),
                     warmupAudio.frames(), warmupGain.data());
    const bool switchedOk = switched.setSampleRate(toRate);

    StereoBuffer switchedAudio{freshInput.left, freshInput.right};
    std::vector<float> switchedGain(freshInput.frames(), 1.0f);
    if (switchedOk) {
        switched.process(switchedAudio.left.data(), switchedAudio.right.data(),
                         switchedAudio.frames(), switchedGain.data());
    }

    config.sampleRate = toRate;
    const LimitedResult fresh = tpltest::runLimiter(freshInput, config);
    tpl::TruePeakLimiter freshReference(config);
    RateSwitchMetrics metrics;
    metrics.fromRate = fromRate;
    metrics.toRate = toRate;
    metrics.newLatency = switched.latencySamples();
    metrics.maxOutputError =
        tpltest::maxAbsDifference(switchedAudio, fresh.audio);
    metrics.maxGainError = 0.0;
    for (std::size_t i = 0; i < switchedGain.size(); ++i) {
        metrics.maxGainError =
            std::max(metrics.maxGainError,
                     std::abs(static_cast<double>(switchedGain[i]) - fresh.gain[i]));
    }
    metrics.pass = switchedOk &&
                   metrics.newLatency == freshReference.latencySamples() &&
                   metrics.maxOutputError <= 1.0e-7 &&
                   metrics.maxGainError <= 1.0e-7;
    return metrics;
}

StereoBuffer concatenateAudition(double sampleRate) {
    const StereoBuffer near = tpltest::makeNearNyquistProbe(sampleRate, 1.05);
    const StereoBuffer hits = tpltest::makeAlternatingHits(sampleRate, 2.35);
    const std::size_t gap = tpltest::framesFor(sampleRate, 0.14);
    const std::size_t total = near.frames() + gap + hits.frames();
    StereoBuffer program{std::vector<float>(total), std::vector<float>(total)};

    std::copy(near.left.begin(), near.left.end(), program.left.begin());
    std::copy(near.right.begin(), near.right.end(), program.right.begin());
    const std::size_t hitOffset = near.frames() + gap;
    std::copy(hits.left.begin(), hits.left.end(),
              program.left.begin() + static_cast<std::ptrdiff_t>(hitOffset));
    std::copy(hits.right.begin(), hits.right.end(),
              program.right.begin() + static_cast<std::ptrdiff_t>(hitOffset));

    const std::size_t fade = std::min(
        tpltest::framesFor(sampleRate, 0.02), total);
    for (std::size_t i = 0; i < fade; ++i) {
        const std::size_t index = total - 1 - i;
        const float gain = static_cast<float>(i) / static_cast<float>(fade);
        program.left[index] *= gain;
        program.right[index] *= gain;
    }
    return program;
}

std::string timestampUtc() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t value = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &value);
#else
    gmtime_r(&value, &utc);
#endif
    std::ostringstream stream;
    stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

void writeCsv(const fs::path& path, const std::vector<SignalMetrics>& metrics) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write " + path.string());
    }
    output << "sample_rate_hz,signal,frames,lookahead_samples,"
              "detector_delay_samples,latency_samples,latency_ms,"
              "ceiling_dbTP,input_sample_peak_dBFS,input_true_peak_dbTP,"
              "output_sample_peak_dBFS,output_true_peak_dbTP,"
              "output_margin_db,max_gain_reduction_db,"
              "max_stereo_gain_delta_db,ceiling_pass\n";
    output << std::fixed << std::setprecision(6);
    for (const SignalMetrics& item : metrics) {
        const double latencyMs = 1000.0 * item.latencySamples / item.sampleRate;
        output << item.sampleRate << ',' << item.name << ',' << item.frames << ','
               << item.lookaheadSamples << ',' << item.detectorDelaySamples << ','
               << item.latencySamples << ',' << latencyMs << ','
               << kCeilingDbTP << ',' << item.inputSamplePeakDb << ','
               << item.inputTruePeakDb << ',' << item.outputSamplePeakDb << ','
               << item.outputTruePeakDb << ',' << item.outputMarginDb << ','
               << item.maxGainReductionDb << ',' << item.maxStereoGainDeltaDb << ','
               << (item.ceilingPass ? "true" : "false") << '\n';
    }
}

void writeJson(const fs::path& path, const std::vector<SignalMetrics>& signals,
               const std::vector<BypassMetrics>& bypasses,
               const std::vector<RateSwitchMetrics>& switches, bool overallPass,
               const fs::path& auditionDir) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write " + path.string());
    }
    output << std::fixed << std::setprecision(6);
    output << "{\n"
           << "  \"generated_utc\": \"" << timestampUtc() << "\",\n"
           << "  \"overall_pass\": " << (overallPass ? "true" : "false") << ",\n"
           << "  \"limiter\": {\n"
           << "    \"ceiling_dbTP\": " << kCeilingDbTP << ",\n"
           << "    \"lookahead_ms\": 3.0,\n"
           << "    \"release_ms\": 90.0,\n"
           << "    \"internal_safety_margin_db\": 0.50,\n"
           << "    \"oversample_factor\": 8,\n"
           << "    \"detector_half_width_samples\": 48,\n"
           << "    \"stereo_linked\": true,\n"
           << "    \"bypass_latency_matched\": true\n"
           << "  },\n"
           << "  \"measurement\": {\n"
           << "    \"true_peak_oversample_factor\": 16,\n"
           << "    \"true_peak_filter_half_width_samples\": 64,\n"
           << "    \"ceiling_tolerance_db\": " << kCeilingToleranceDb << "\n"
           << "  },\n"
           << "  \"probes\": {\n"
           << "    \"near_nyquist_phase\": {\n"
           << "      \"tones_normalized_to_fs\": [0.484375, 0.453125],\n"
           << "      \"relative_amplitudes\": [0.80, 0.52],\n"
           << "      \"phases_radians\": [3.4033920414, 3.9269908170],\n"
           << "      \"gain\": 1.25,\n"
           << "      \"edge_fade_ms\": 35.0,\n"
           << "      \"note\": \"digital sample peak is intentionally below true peak\"\n"
           << "    },\n"
           << "    \"alternating_lr_hits\": {\n"
           << "      \"hit_frequency_hz\": 73.0,\n"
           << "      \"hit_amplitude\": 2.05,\n"
           << "      \"hit_spacing_ms\": 290.0,\n"
           << "      \"hit_decay_ms\": 55.0,\n"
           << "      \"left_pilot_hz\": 997.0,\n"
           << "      \"right_pilot_hz\": 1493.0,\n"
           << "      \"note\": \"linked stereo gain is measured from both output channels\"\n"
           << "    }\n"
           << "  },\n"
           << "  \"signals\": [\n";

    for (std::size_t i = 0; i < signals.size(); ++i) {
        const SignalMetrics& item = signals[i];
        output << "    {\n"
               << "      \"sample_rate_hz\": " << item.sampleRate << ",\n"
               << "      \"name\": \"" << item.name << "\",\n"
               << "      \"frames\": " << item.frames << ",\n"
               << "      \"lookahead_samples\": " << item.lookaheadSamples << ",\n"
               << "      \"detector_delay_samples\": " << item.detectorDelaySamples << ",\n"
               << "      \"latency_samples\": " << item.latencySamples << ",\n"
               << "      \"input_sample_peak_dBFS\": " << item.inputSamplePeakDb << ",\n"
               << "      \"input_true_peak_dbTP\": " << item.inputTruePeakDb << ",\n"
               << "      \"output_sample_peak_dBFS\": " << item.outputSamplePeakDb << ",\n"
               << "      \"output_true_peak_dbTP\": " << item.outputTruePeakDb << ",\n"
               << "      \"output_margin_db\": " << item.outputMarginDb << ",\n"
               << "      \"max_gain_reduction_db\": " << item.maxGainReductionDb << ",\n"
               << "      \"max_stereo_gain_delta_db\": " << item.maxStereoGainDeltaDb << ",\n"
               << "      \"ceiling_pass\": " << (item.ceilingPass ? "true" : "false") << "\n"
               << "    }" << (i + 1 == signals.size() ? "\n" : ",\n");
    }
    output << "  ],\n"
           << "  \"bypass_alignment\": [\n";
    for (std::size_t i = 0; i < bypasses.size(); ++i) {
        const BypassMetrics& item = bypasses[i];
        output << "    {\"sample_rate_hz\": " << item.sampleRate
               << ", \"reported_latency_samples\": " << item.reportedLatency
               << ", \"measured_latency_samples\": " << item.measuredLatency
               << ", \"max_alignment_error\": " << item.maxAlignmentError
               << ", \"pass\": " << (item.exact ? "true" : "false") << "}"
               << (i + 1 == bypasses.size() ? "\n" : ",\n");
    }
    output << "  ],\n"
           << "  \"sample_rate_switch\": [\n";
    for (std::size_t i = 0; i < switches.size(); ++i) {
        const RateSwitchMetrics& item = switches[i];
        output << "    {\"from_hz\": " << item.fromRate
               << ", \"to_hz\": " << item.toRate
               << ", \"new_latency_samples\": " << item.newLatency
               << ", \"max_output_error\": " << item.maxOutputError
               << ", \"max_gain_error\": " << item.maxGainError
               << ", \"pass\": " << (item.pass ? "true" : "false") << "}"
               << (i + 1 == switches.size() ? "\n" : ",\n");
    }
    output << "  ],\n"
           << "  \"artifacts\": {\n"
           << "    \"metrics_csv\": \"limiter_metrics.csv\",\n"
           << "    \"audition_directory\": \"" << auditionDir.string() << "\",\n"
           << "    \"audition_files\": [\n"
           << "      \"audition/01_input_aligned_48k.wav\",\n"
           << "      \"audition/02_limited_48k.wav\",\n"
           << "      \"audition/03_bypass_48k.wav\",\n"
           << "      \"audition/04_limited_preview_48k_pcm24.wav\"\n"
           << "    ]\n"
           << "  }\n"
           << "}\n";
}

bool allSignalsPass(const std::vector<SignalMetrics>& signals) {
    for (const SignalMetrics& item : signals) {
        if (!item.ceilingPass || item.maxGainReductionDb < 0.5 ||
            item.maxStereoGainDeltaDb > 0.001) {
            return false;
        }
    }
    return true;
}

bool allBypassesPass(const std::vector<BypassMetrics>& bypasses) {
    for (const BypassMetrics& item : bypasses) {
        if (!item.exact) {
            return false;
        }
    }
    return true;
}

bool allSwitchesPass(const std::vector<RateSwitchMetrics>& switches) {
    for (const RateSwitchMetrics& item : switches) {
        if (!item.pass) {
            return false;
        }
    }
    return true;
}

void printResults(const std::vector<SignalMetrics>& signals,
                  const std::vector<BypassMetrics>& bypasses,
                  const std::vector<RateSwitchMetrics>& switches) {
    std::cout << std::left << std::setw(9) << "Rate" << std::setw(21) << "Signal"
              << std::right << std::setw(10) << "In TP" << std::setw(10)
              << "Out TP" << std::setw(9) << "Margin" << std::setw(9)
              << "GR" << std::setw(10) << "L/R dB" << std::setw(8) << "Status"
              << '\n';
    std::cout << std::string(96, '-') << '\n';
    for (const SignalMetrics& item : signals) {
        std::cout << std::left << std::setw(9) << tpltest::fmtFixed(item.sampleRate, 0)
                  << std::setw(21) << item.name << std::right << std::setw(10)
                  << tpltest::fmtFixed(item.inputTruePeakDb, 2) << std::setw(10)
                  << tpltest::fmtFixed(item.outputTruePeakDb, 2) << std::setw(9)
                  << tpltest::fmtFixed(item.outputMarginDb, 2) << std::setw(9)
                  << tpltest::fmtFixed(item.maxGainReductionDb, 2) << std::setw(10)
                  << tpltest::fmtFixed(item.maxStereoGainDeltaDb, 4) << std::setw(8)
                  << (item.ceilingPass ? "PASS" : "FAIL") << '\n';
    }
    std::cout << '\n';
    for (const BypassMetrics& item : bypasses) {
        std::cout << "Bypass " << tpltest::fmtFixed(item.sampleRate, 0)
                  << " Hz: reported/measured latency " << item.reportedLatency
                  << "/" << item.measuredLatency << " samples, max error "
                  << item.maxAlignmentError << " -> "
                  << (item.exact ? "PASS" : "FAIL") << '\n';
    }
    for (const RateSwitchMetrics& item : switches) {
        std::cout << "Rate switch " << tpltest::fmtFixed(item.fromRate, 0)
                  << " -> " << tpltest::fmtFixed(item.toRate, 0)
                  << " Hz: latency " << item.newLatency
                  << ", max output error " << item.maxOutputError
                  << " -> " << (item.pass ? "PASS" : "FAIL") << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const AppOptions options = parseArguments(argc, argv);
        fs::create_directories(options.outputDir);

        const TruePeakAnalyzer analyzer(16, 64);
        std::vector<SignalMetrics> signals;
        for (double sampleRate : {44100.0, 48000.0, 96000.0}) {
            signals.push_back(evaluateSignal(
                tpltest::makeNearNyquistProbe(sampleRate, 1.25), sampleRate,
                "near_nyquist_phase", analyzer));
            signals.push_back(evaluateSignal(
                tpltest::makeAlternatingHits(sampleRate, 2.40), sampleRate,
                "alternating_lr_hits", analyzer));
        }

        std::vector<BypassMetrics> bypasses;
        for (double sampleRate : {44100.0, 48000.0, 96000.0}) {
            bypasses.push_back(evaluateBypass(sampleRate));
        }

        std::vector<RateSwitchMetrics> switches;
        switches.push_back(evaluateRateSwitch(44100.0, 48000.0));
        switches.push_back(evaluateRateSwitch(48000.0, 96000.0));
        switches.push_back(evaluateRateSwitch(96000.0, 44100.0));

        const bool overallPass =
            allSignalsPass(signals) && allBypassesPass(bypasses) &&
            allSwitchesPass(switches);

        const fs::path csvPath = options.outputDir / "limiter_metrics.csv";
        const fs::path jsonPath = options.outputDir / "limiter_report.json";
        const fs::path auditionDir = options.outputDir / "audition";
        writeCsv(csvPath, signals);
        writeJson(jsonPath, signals, bypasses, switches, overallPass,
                  fs::path("audition"));

        if (options.writeWav) {
            fs::create_directories(auditionDir);
            constexpr double auditionRate = 48000.0;
            const StereoBuffer program = concatenateAudition(auditionRate);

            tpl::PeakLimiterConfig limitedConfig;
            limitedConfig.sampleRate = auditionRate;
            limitedConfig.ceilingDbTP = kCeilingDbTP;
            tpl::TruePeakLimiter limitedLimiter(limitedConfig);
            const int latency = limitedLimiter.latencySamples();
            const StereoBuffer aligned = tpltest::delayedCopy(program, latency);

            const LimitedResult limited = tpltest::runLimiter(program, limitedConfig);
            tpl::PeakLimiterConfig bypassConfig = limitedConfig;
            bypassConfig.bypass = true;
            const LimitedResult bypass = tpltest::runLimiter(program, bypassConfig);

            tpltest::WavWriter::writeFloat32(
                (auditionDir / "01_input_aligned_48k.wav").string(),
                aligned, static_cast<int>(auditionRate));
            tpltest::WavWriter::writeFloat32(
                (auditionDir / "02_limited_48k.wav").string(),
                limited.audio, static_cast<int>(auditionRate));
            tpltest::WavWriter::writeFloat32(
                (auditionDir / "03_bypass_48k.wav").string(),
                bypass.audio, static_cast<int>(auditionRate));
            tpltest::WavWriter::writePcm24(
                (auditionDir / "04_limited_preview_48k_pcm24.wav").string(),
                limited.audio, static_cast<int>(auditionRate));
        }

        printResults(signals, bypasses, switches);
        std::cout << "\nWrote " << csvPath << " and " << jsonPath;
        if (options.writeWav) {
            std::cout << " plus WAVs in " << auditionDir;
        }
        std::cout << "\nOverall: " << (overallPass ? "PASS" : "FAIL") << '\n';
        return overallPass ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}


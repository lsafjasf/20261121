#include "aec.hpp"
#include "demo.hpp"
#include "report.hpp"
#include "wav_io.hpp"

#include <charconv>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
    bool demo = true;
    std::filesystem::path far_path;
    std::filesystem::path mic_path;
    std::filesystem::path near_path;
    std::filesystem::path echo_path;
    std::filesystem::path out_dir = "artifacts/demo";
    int sample_rate = 16000;
    int filter_taps = 640;
    int delay_min_ms = 0;
    int delay_max_ms = 240;
    double step_size = 0.50;
    double duration = 26.0;
    double dt_start = 10.0;
    double dt_end = 14.0;
    bool dt_start_set = false;
    bool dt_end_set = false;
    std::vector<aec::DelayStep> delay_steps = {
        {0.0, 48}, {5.0, 112}, {16.0, 72}, {21.0, 144}};
};

void print_usage() {
    std::cout
        << "duplex_aec_debug - variable-delay duplex AEC visual debugger\n\n"
        << "Usage:\n"
        << "  duplex_aec_debug --demo [--out-dir DIR]\n"
        << "  duplex_aec_debug --far far.wav --mic mic.wav [options]\n\n"
        << "Options:\n"
        << "  --demo                    synthesize a reproducible demo (default)\n"
        << "  --far PATH                far-end reference WAV (16-bit PCM)\n"
        << "  --mic PATH                microphone WAV containing echo (+ near end)\n"
        << "  --near-clean PATH         optional near-end truth for metrics\n"
        << "  --true-echo PATH          optional echo truth for metrics\n"
        << "  --out-dir DIR             artifact directory (default artifacts/demo)\n"
        << "  --filter-taps N           NLMS taps after bulk alignment (default 640)\n"
        << "  --delay-min-ms N          minimum bulk delay (default 0)\n"
        << "  --delay-max-ms N          maximum bulk delay (default 240)\n"
        << "  --step-size X             NLMS step size (default 0.50)\n"
        << "  --dt-start S              nominal double-talk start for report\n"
        << "  --dt-end S                nominal double-talk end for report\n"
        << "  --duration S              demo duration (default 26)\n"
        << "  --delay-schedule LIST     demo schedule t:delay,t:delay in seconds/ms\n"
        << "  --help                    show this help\n";
}

bool parse_double(const std::string& text, double& value) {
    char* end = nullptr;
    value = std::strtod(text.c_str(), &end);
    return end != text.c_str() && *end == '\0';
}

bool parse_int(const std::string& text, int& value) {
    char* end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0') {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool parse_schedule(const std::string& text, std::vector<aec::DelayStep>& steps,
                    std::string& error) {
    std::vector<aec::DelayStep> parsed;
    std::stringstream items(text);
    std::string item;
    while (std::getline(items, item, ',')) {
        const std::size_t colon = item.find(':');
        if (colon == std::string::npos) {
            error = "bad delay schedule item: " + item;
            return false;
        }
        aec::DelayStep step;
        int delay = 0;
        if (!parse_double(item.substr(0, colon), step.time_seconds) ||
            !parse_int(item.substr(colon + 1), delay) || delay < 0) {
            error = "bad delay schedule item: " + item;
            return false;
        }
        step.delay_ms = delay;
        parsed.push_back(step);
    }
    if (parsed.empty()) {
        error = "delay schedule is empty";
        return false;
    }
    std::sort(parsed.begin(), parsed.end(),
              [](const auto& left, const auto& right) {
                  return left.time_seconds < right.time_seconds;
              });
    steps = std::move(parsed);
    return true;
}

std::optional<Options> parse_options(int argc, char** argv, std::string& error) {
    Options options;
    bool explicit_demo = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&](const char* name) -> std::optional<std::string> {
            if (i + 1 >= argc) {
                error = std::string("missing value after ") + name;
                return std::nullopt;
            }
            return std::string(argv[++i]);
        };
        if (arg == "--help" || arg == "-h") {
            return std::nullopt;
        } else if (arg == "--demo") {
            explicit_demo = true;
            options.demo = true;
        } else if (arg == "--far") {
            const auto text = value("--far");
            if (!text) return std::nullopt;
            options.far_path = *text;
            options.demo = false;
        } else if (arg == "--mic") {
            const auto text = value("--mic");
            if (!text) return std::nullopt;
            options.mic_path = *text;
            options.demo = false;
        } else if (arg == "--near-clean") {
            const auto text = value("--near-clean");
            if (!text) return std::nullopt;
            options.near_path = *text;
        } else if (arg == "--true-echo") {
            const auto text = value("--true-echo");
            if (!text) return std::nullopt;
            options.echo_path = *text;
        } else if (arg == "--out-dir") {
            const auto text = value("--out-dir");
            if (!text) return std::nullopt;
            options.out_dir = *text;
        } else if (arg == "--filter-taps") {
            const auto text = value("--filter-taps");
            if (!text || !parse_int(*text, options.filter_taps)) {
                error = "invalid --filter-taps";
                return std::nullopt;
            }
        } else if (arg == "--delay-min-ms") {
            const auto text = value("--delay-min-ms");
            if (!text || !parse_int(*text, options.delay_min_ms)) {
                error = "invalid --delay-min-ms";
                return std::nullopt;
            }
        } else if (arg == "--delay-max-ms") {
            const auto text = value("--delay-max-ms");
            if (!text || !parse_int(*text, options.delay_max_ms)) {
                error = "invalid --delay-max-ms";
                return std::nullopt;
            }
        } else if (arg == "--step-size") {
            const auto text = value("--step-size");
            if (!text || !parse_double(*text, options.step_size)) {
                error = "invalid --step-size";
                return std::nullopt;
            }
        } else if (arg == "--dt-start") {
            const auto text = value("--dt-start");
            if (!text || !parse_double(*text, options.dt_start)) {
                error = "invalid --dt-start";
                return std::nullopt;
            }
            options.dt_start_set = true;
        } else if (arg == "--dt-end") {
            const auto text = value("--dt-end");
            if (!text || !parse_double(*text, options.dt_end)) {
                error = "invalid --dt-end";
                return std::nullopt;
            }
            options.dt_end_set = true;
        } else if (arg == "--duration") {
            const auto text = value("--duration");
            if (!text || !parse_double(*text, options.duration)) {
                error = "invalid --duration";
                return std::nullopt;
            }
        } else if (arg == "--delay-schedule") {
            const auto text = value("--delay-schedule");
            if (!text || !parse_schedule(*text, options.delay_steps, error)) {
                if (error.empty()) error = "invalid --delay-schedule";
                return std::nullopt;
            }
        } else {
            error = "unknown option: " + arg;
            return std::nullopt;
        }
    }
    if (explicit_demo && (!options.far_path.empty() || !options.mic_path.empty())) {
        error = "--demo cannot be combined with --far/--mic";
        return std::nullopt;
    }
    if (!options.demo && (options.far_path.empty() || options.mic_path.empty())) {
        error = "external mode requires both --far and --mic";
        return std::nullopt;
    }
    if (options.filter_taps < 32 || options.delay_max_ms <= options.delay_min_ms ||
        options.step_size <= 0.0 || options.step_size > 1.5) {
        error = "invalid AEC configuration";
        return std::nullopt;
    }
    return options;
}

bool load_external(const Options& options, std::vector<float>& far,
                   std::vector<float>& mic, std::vector<float>& near,
                   std::vector<float>& echo, int& sample_rate, std::string& error) {
    aec::MonoAudio far_audio;
    aec::MonoAudio mic_audio;
    if (!aec::read_mono_wav(options.far_path, far_audio, error) ||
        !aec::read_mono_wav(options.mic_path, mic_audio, error)) {
        return false;
    }
    if (far_audio.sample_rate != mic_audio.sample_rate) {
        error = "far and mic sample rates differ";
        return false;
    }
    sample_rate = far_audio.sample_rate;
    const std::size_t count = std::min(far_audio.samples.size(), mic_audio.samples.size());
    far = aec::slice(far_audio.samples, 0, count);
    mic = aec::slice(mic_audio.samples, 0, count);
    if (!options.near_path.empty()) {
        aec::MonoAudio near_audio;
        if (!aec::read_mono_wav(options.near_path, near_audio, error)) {
            return false;
        }
        if (near_audio.sample_rate == sample_rate &&
            near_audio.samples.size() >= count) {
            near = aec::slice(near_audio.samples, 0, count);
        }
    }
    if (!options.echo_path.empty()) {
        aec::MonoAudio echo_audio;
        if (!aec::read_mono_wav(options.echo_path, echo_audio, error)) {
            return false;
        }
        if (echo_audio.sample_rate == sample_rate &&
            echo_audio.samples.size() >= count) {
            echo = aec::slice(echo_audio.samples, 0, count);
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string error;
    const auto parsed = parse_options(argc, argv, error);
    if (!parsed) {
        if (!error.empty()) {
            std::cerr << "error: " << error << "\n\n";
        }
        print_usage();
        return error.empty() ? 0 : 2;
    }
    const Options options = *parsed;
    std::vector<float> far;
    std::vector<float> mic;
    std::vector<float> near;
    std::vector<float> echo;
    std::vector<aec::DelayStep> true_steps;
    int sample_rate = options.sample_rate;

    if (options.demo) {
        aec::DemoConfig demo;
        demo.sample_rate = options.sample_rate;
        demo.duration_seconds = options.duration;
        demo.double_talk_start = options.dt_start;
        demo.double_talk_end = options.dt_end;
        demo.delay_steps = options.delay_steps;
        const aec::DemoData data = aec::synthesize_demo(demo);
        far = data.far;
        near = data.near;
        echo = data.echo;
        mic = data.mic;
        true_steps = data.delay_steps;
        sample_rate = data.sample_rate;
    } else if (!load_external(options, far, mic, near, echo, sample_rate, error)) {
        std::cerr << "error: " << error << "\n";
        return 3;
    }

    aec::AecConfig config;
    config.sample_rate = sample_rate;
    config.filter_taps = options.filter_taps;
    config.delay_min_ms = options.delay_min_ms;
    config.delay_max_ms = options.delay_max_ms;
    config.delay_step_size = options.step_size;
    aec::GroundTruth truth;
    if (near.size() == mic.size()) {
        truth.near_clean = &near;
    }
    if (echo.size() == mic.size()) {
        truth.true_echo = &echo;
    }

    std::cerr << "processing " << far.size() << " samples at " << sample_rate
              << " Hz..." << std::endl;
    const aec::AecResult result = aec::run_aec(far, mic, config, truth);
    if (!options.demo && true_steps.empty()) {
        for (const auto& step : result.delay_steps) {
            true_steps.push_back({step.time_seconds, step.delay_ms});
        }
    }
    double report_dt_start = options.dt_start;
    double report_dt_end = options.dt_end;
    if (!options.demo) {
        if (!options.dt_start_set &&
            std::isfinite(result.double_talk_start_detected)) {
            report_dt_start = result.double_talk_start_detected;
        }
        if (!options.dt_end_set && std::isfinite(result.double_talk_end_detected)) {
            report_dt_end = result.double_talk_end_detected;
        }
    }
    aec::AnalysisSummary summary;
    if (!aec::write_artifact_bundle(options.out_dir, sample_rate, far, near, echo, mic,
                                    result, true_steps, report_dt_start,
                                    report_dt_end, config, summary, error)) {
        std::cerr << "error: " << error << "\n";
        return 4;
    }
    aec::print_analysis(summary, result);
    std::cout << "Artifacts: " << std::filesystem::absolute(options.out_dir) << "\n";
    std::cout << "Listen: report.html, double_talk_ab.wav, double_talk_input.wav, "
                 "double_talk_output.wav\n";
    return 0;
}

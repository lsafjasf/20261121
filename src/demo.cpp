#include "demo.hpp"

#include <array>
#include <random>

namespace aec {
namespace {

struct FormantSet {
    double f1;
    double f2;
    double f3;
};

struct Syllable {
    double start;
    double length;
    double f0;
    FormantSet formants;
    double amplitude;
    double consonant_mix;
};

class Resonator {
public:
    void configure(double frequency, double bandwidth, int sample_rate) {
        const double r = std::exp(-kPi * bandwidth / sample_rate);
        const double theta = 2.0 * kPi * frequency / sample_rate;
        b0_ = 1.0 - r;
        a1_ = -2.0 * r * std::cos(theta);
        a2_ = r * r;
    }

    double process(double input) {
        const double output = b0_ * input - a1_ * y1_ - a2_ * y2_;
        y2_ = y1_;
        y1_ = output;
        return output;
    }

private:
    double b0_ = 0.0;
    double a1_ = 0.0;
    double a2_ = 0.0;
    double y1_ = 0.0;
    double y2_ = 0.0;
};

double soft_clip(double x) {
    return std::tanh(1.15 * x) / std::tanh(1.15);
}

std::vector<Syllable> make_syllables(double duration, double base_f0,
                                     std::mt19937& rng) {
    static constexpr std::array<FormantSet, 8> kVowels = {{
        {730.0, 1090.0, 2440.0},
        {270.0, 2290.0, 3010.0},
        {300.0, 870.0, 2240.0},
        {530.0, 1840.0, 2480.0},
        {570.0, 840.0, 2410.0},
        {660.0, 1720.0, 2410.0},
        {490.0, 1350.0, 1690.0},
        {440.0, 1020.0, 2240.0},
    }};
    std::uniform_real_distribution<double> jitter(0.88, 1.12);
    std::uniform_int_distribution<int> vowel_pick(0, static_cast<int>(kVowels.size()) - 1);
    std::uniform_real_distribution<double> gap(0.0, 0.12);
    std::vector<Syllable> result;
    double t = 0.08;
    int index = 0;
    while (t < duration) {
        const double length = 0.18 + 0.055 * (index % 4) + 0.025 * (index % 3);
        const bool pause = (index % 9 == 8);
        if (!pause) {
            Syllable syllable;
            syllable.start = t;
            syllable.length = std::min(length, duration - t);
            syllable.f0 = base_f0 * jitter(rng) *
                          (1.0 + 0.045 * std::sin(2.0 * kPi * 0.23 * t));
            syllable.formants = kVowels[vowel_pick(rng)];
            syllable.amplitude = 0.72 + 0.18 * jitter(rng);
            syllable.consonant_mix = (index % 3 == 0) ? 0.35 : 0.08;
            if (syllable.length > 0.02) {
                result.push_back(syllable);
            }
        }
        t += length + gap(rng);
        ++index;
    }
    return result;
}

std::vector<float> synthesize_voice(int sample_rate, double duration, double base_f0,
                                    unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, 1.0);
    const auto syllables = make_syllables(duration, base_f0, rng);
    std::vector<float> output(static_cast<std::size_t>(duration * sample_rate), 0.0f);
    std::size_t syllable_index = 0;
    double phase = 0.0;
    Resonator r1;
    Resonator r2;
    Resonator r3;
    double previous_noise = 0.0;
    int configured_index = -1;

    for (std::size_t i = 0; i < output.size(); ++i) {
        const double t = static_cast<double>(i) / sample_rate;
        while (syllable_index + 1 < syllables.size() &&
               t >= syllables[syllable_index + 1].start) {
            ++syllable_index;
        }
        if (syllables.empty() || t < syllables.front().start ||
            t >= syllables.back().start + syllables.back().length) {
            previous_noise = noise(rng);
            continue;
        }
        if (static_cast<int>(syllable_index) != configured_index) {
            configured_index = static_cast<int>(syllable_index);
            const auto& formants = syllables[syllable_index].formants;
            r1.configure(formants.f1, 90.0, sample_rate);
            r2.configure(formants.f2, 120.0, sample_rate);
            r3.configure(formants.f3, 180.0, sample_rate);
        }

        const Syllable& current = syllables[syllable_index];
        const double local = t - current.start;
        if (local < 0.0 || local >= current.length) {
            previous_noise = noise(rng);
            continue;
        }
        const double attack = std::min(1.0, local / 0.018);
        const double release = std::clamp((current.length - local) / 0.055, 0.0, 1.0);
        const double contour = std::sin(kPi * clamp01(local / current.length));
        const double envelope = current.amplitude * attack * release *
                                (0.55 + 0.45 * contour);
        const double expressive_f0 =
            current.f0 * (1.0 + 0.018 * std::sin(2.0 * kPi * 4.7 * t));
        phase += expressive_f0 / sample_rate;
        if (phase >= 1.0) {
            phase -= 1.0;
        }
        const double glottal = std::pow(std::max(0.0, std::sin(kPi * phase)), 2.2);
        const double raw_noise = noise(rng);
        const double high_noise = raw_noise - previous_noise;
        previous_noise = raw_noise;
        const double excitation =
            0.82 * glottal + current.consonant_mix * 0.32 * high_noise;
        const double voiced = r3.process(r2.process(r1.process(excitation)));
        output[i] = static_cast<float>(soft_clip(envelope * voiced * 3.8));
    }
    return output;
}

void fade_edges(std::vector<float>& samples, int sample_rate, double start, double end,
                double fade_seconds = 0.02) {
    const std::size_t fade_length =
        static_cast<std::size_t>(std::max(1.0, fade_seconds * sample_rate));
    const std::size_t begin = static_cast<std::size_t>(std::max(0.0, start * sample_rate));
    const std::size_t finish =
        std::min(samples.size(), static_cast<std::size_t>(end * sample_rate));
    for (std::size_t i = 0; i < fade_length && begin + i < finish; ++i) {
        const float gain = static_cast<float>(i) / static_cast<float>(fade_length);
        samples[begin + i] *= gain;
    }
    for (std::size_t i = 0; i < fade_length && finish > i + begin; ++i) {
        const std::size_t index = finish - 1 - i;
        const float gain = static_cast<float>(i) / static_cast<float>(fade_length);
        samples[index] *= gain;
    }
    if (finish < samples.size()) {
        std::fill(samples.begin() + static_cast<std::ptrdiff_t>(finish), samples.end(),
                  0.0f);
    }
}

void normalize_rms(std::vector<float>& samples, double target_dbfs) {
    long double energy = 0.0;
    std::size_t count = 0;
    for (float value : samples) {
        if (std::abs(value) > 1.0e-6f) {
            energy += static_cast<long double>(value) * value;
            ++count;
        }
    }
    if (count == 0) {
        return;
    }
    const double current = std::sqrt(static_cast<double>(energy / count));
    const double target = std::pow(10.0, target_dbfs / 20.0);
    const double gain = target / std::max(current, 1.0e-12);
    for (float& value : samples) {
        value = static_cast<float>(value * gain);
    }
}

int delay_at(const std::vector<DelayStep>& steps, double time, int sample_rate) {
    int delay_ms = steps.empty() ? 0 : steps.front().delay_ms;
    for (const auto& step : steps) {
        if (time >= step.time_seconds) {
            delay_ms = step.delay_ms;
        }
    }
    return static_cast<int>(std::lrint(delay_ms * sample_rate / 1000.0));
}

}  // namespace

DemoData synthesize_demo(const DemoConfig& config) {
    const int sr = config.sample_rate;
    const std::size_t count =
        static_cast<std::size_t>(std::lrint(config.duration_seconds * sr));
    DemoData data;
    data.sample_rate = sr;
    data.delay_steps = config.delay_steps;
    data.far = synthesize_voice(sr, config.duration_seconds, 112.0, config.seed);
    data.near = synthesize_voice(sr, config.duration_seconds, 196.0, config.seed + 711U);

    const double near_start =
        std::max(0.0, config.double_talk_start - config.near_lead_seconds);
    const double near_end =
        std::min(config.duration_seconds, config.double_talk_end + config.near_tail_seconds);
    for (std::size_t i = 0; i < count; ++i) {
        const double t = static_cast<double>(i) / sr;
        if (t < near_start || t >= near_end) {
            data.near[i] = 0.0f;
        }
    }
    fade_edges(data.near, sr, near_start, near_end, 0.025);
    normalize_rms(data.far, -18.5);
    normalize_rms(data.near, -17.0);

    static constexpr std::array<double, 5> kEchoTaps = {0.72, 0.21, -0.12, 0.055, 0.028};
    static constexpr std::array<int, 5> kEchoOffsets = {0, 3, 11, 29, 61};
    double tap_energy = 0.0;
    for (double tap : kEchoTaps) {
        tap_energy += tap * tap;
    }
    const double echo_gain =
        std::pow(10.0, config.echo_gain_db / 20.0) / std::sqrt(tap_energy);

    data.echo.assign(count, 0.0f);
    data.mic.assign(count, 0.0f);
    data.true_delay_samples.assign(count, 0);
    std::mt19937 rng(config.seed + 19U);
    std::normal_distribution<double> noise(0.0, 1.0);
    const double noise_gain = std::pow(10.0, config.noise_dbfs / 20.0);

    for (std::size_t i = 0; i < count; ++i) {
        const double t = static_cast<double>(i) / sr;
        const int delay = delay_at(config.delay_steps, t, sr);
        data.true_delay_samples[i] = delay;
        double echo = 0.0;
        for (std::size_t tap = 0; tap < kEchoTaps.size(); ++tap) {
            const int source_index =
                static_cast<int>(i) - delay - kEchoOffsets[tap];
            if (source_index >= 0 && static_cast<std::size_t>(source_index) < count) {
                echo += echo_gain * kEchoTaps[tap] *
                        data.far[static_cast<std::size_t>(source_index)];
            }
        }
        data.echo[i] = static_cast<float>(echo);
        data.mic[i] =
            static_cast<float>(echo + data.near[i] + noise_gain * noise(rng));
    }

    const double scale = std::min(quantize_peak_scale(data.mic, -1.2),
                                  quantize_peak_scale(data.near, -1.2));
    for (std::size_t i = 0; i < count; ++i) {
        data.far[i] = static_cast<float>(data.far[i] * scale);
        data.near[i] = static_cast<float>(data.near[i] * scale);
        data.echo[i] = static_cast<float>(data.echo[i] * scale);
        data.mic[i] = static_cast<float>(data.mic[i] * scale);
    }
    return data;
}

}  // namespace aec

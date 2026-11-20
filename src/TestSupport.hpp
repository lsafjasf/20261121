#pragma once

#include "TruePeakLimiter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tpltest {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct StereoBuffer {
    std::vector<float> left;
    std::vector<float> right;

    std::size_t frames() const noexcept { return left.size(); }
};

struct LimitedResult {
    StereoBuffer audio;
    std::vector<float> gain;
};

inline double linearToDb(double linear) {
    return 20.0 * std::log10(std::max(linear, 1.0e-12));
}

inline double samplePeak(const StereoBuffer& buffer) {
    double peak = 0.0;
    for (std::size_t i = 0; i < buffer.frames(); ++i) {
        peak = std::max(peak, std::abs(static_cast<double>(buffer.left[i])));
        peak = std::max(peak, std::abs(static_cast<double>(buffer.right[i])));
    }
    return peak;
}

inline double rms(const StereoBuffer& buffer) {
    double sum = 0.0;
    for (std::size_t i = 0; i < buffer.frames(); ++i) {
        sum += static_cast<double>(buffer.left[i]) * buffer.left[i];
        sum += static_cast<double>(buffer.right[i]) * buffer.right[i];
    }
    return buffer.frames() == 0
               ? 0.0
               : std::sqrt(sum / static_cast<double>(2 * buffer.frames()));
}

inline std::string fmtFixed(double value, int decimals = 3) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(decimals) << value;
    return stream.str();
}

class TruePeakAnalyzer {
public:
    explicit TruePeakAnalyzer(int factor = 16, int halfWidth = 64)
        : factor_(factor), halfWidth_(halfWidth) {
        if (factor_ < 4 || halfWidth_ < 8) {
            throw std::invalid_argument("invalid true-peak analyzer configuration");
        }
    }

    double measure(const StereoBuffer& buffer) const {
        double peak = samplePeak(buffer);
        if (buffer.frames() == 0) {
            return peak;
        }

        const int taps = 2 * halfWidth_;
        std::vector<std::vector<double>> kernels(
            static_cast<std::size_t>(factor_),
            std::vector<double>(static_cast<std::size_t>(taps), 0.0));
        for (int phase = 0; phase < factor_; ++phase) {
            const double fraction = static_cast<double>(phase) /
                                    static_cast<double>(factor_);
            double norm = 0.0;
            for (int tap = 0; tap < taps; ++tap) {
                const double distance =
                    static_cast<double>(tap - halfWidth_ + 1) - fraction;
                const double sinc = std::abs(distance) < 1.0e-12
                                        ? 1.0
                                        : std::sin(kPi * distance) / (kPi * distance);
                const double normalized =
                    distance / static_cast<double>(halfWidth_);
                constexpr double beta = 4.0;
                const double window =
                    modifiedBesselI0(beta * std::sqrt(
                        std::max(0.0, 1.0 - normalized * normalized))) /
                    modifiedBesselI0(beta);
                kernels[static_cast<std::size_t>(phase)][static_cast<std::size_t>(tap)] =
                    sinc * window;
                norm += sinc * window;
            }
            if (std::abs(norm) > 1.0e-9) {
                for (double& value : kernels[static_cast<std::size_t>(phase)]) {
                    value /= norm;
                }
            }
        }

        for (std::size_t base = 0; base < buffer.frames(); ++base) {
            for (int phase = 0; phase < factor_; ++phase) {
                double valueL = 0.0;
                double valueR = 0.0;
                for (int tap = 0; tap < taps; ++tap) {
                    const long long source =
                        static_cast<long long>(base) + tap - halfWidth_ + 1;
                    const double coefficient =
                        kernels[static_cast<std::size_t>(phase)]
                               [static_cast<std::size_t>(tap)];
                    double sampleL = 0.0;
                    double sampleR = 0.0;
                    if (source >= 0 &&
                        source < static_cast<long long>(buffer.frames())) {
                        sampleL = buffer.left[static_cast<std::size_t>(source)];
                        sampleR = buffer.right[static_cast<std::size_t>(source)];
                    }
                    valueL += coefficient * sampleL;
                    valueR += coefficient * sampleR;
                }
                peak = std::max(peak, std::abs(valueL));
                peak = std::max(peak, std::abs(valueR));
            }
        }
        return peak;
    }

    double measureDb(const StereoBuffer& buffer) const {
        return linearToDb(measure(buffer));
    }

private:
    static double modifiedBesselI0(double value) {
        double sum = 1.0;
        double term = 1.0;
        for (int order = 1; order < 64; ++order) {
            term *= (value * value * 0.25) /
                    (static_cast<double>(order) * static_cast<double>(order));
            sum += term;
            if (term < sum * 1.0e-15) {
                break;
            }
        }
        return sum;
    }

    int factor_;
    int halfWidth_;
};

class WavWriter {
public:
    static void writeFloat32(const std::string& path, const StereoBuffer& buffer,
                             int sampleRate) {
        if (sampleRate <= 0) {
            throw std::invalid_argument("WAV sample rate must be positive");
        }
        if (buffer.left.size() != buffer.right.size()) {
            throw std::invalid_argument("stereo WAV channels must have equal length");
        }

        const std::uint32_t channels = 2;
        const std::uint32_t bytesPerSample = 4;
        const std::uint64_t dataBytes64 =
            static_cast<std::uint64_t>(buffer.frames()) * channels * bytesPerSample;
        if (dataBytes64 > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("WAV input is too large for a RIFF file");
        }
        const std::uint32_t dataBytes = static_cast<std::uint32_t>(dataBytes64);

        std::ofstream output(path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot open WAV file for writing: " + path);
        }

        writeTag(output, "RIFF");
        writeU32(output, 4u + (8u + 16u) + (8u + 4u) + (8u + dataBytes));
        writeTag(output, "WAVE");
        writeTag(output, "fmt ");
        writeU32(output, 16u);
        writeU16(output, 3u);
        writeU16(output, static_cast<std::uint16_t>(channels));
        writeU32(output, static_cast<std::uint32_t>(sampleRate));
        writeU32(output, static_cast<std::uint32_t>(sampleRate) * channels * bytesPerSample);
        writeU16(output, static_cast<std::uint16_t>(channels * bytesPerSample));
        writeU16(output, static_cast<std::uint16_t>(bytesPerSample * 8));
        writeTag(output, "fact");
        writeU32(output, 4u);
        writeU32(output, static_cast<std::uint32_t>(buffer.frames()));
        writeTag(output, "data");
        writeU32(output, dataBytes);
        for (std::size_t i = 0; i < buffer.frames(); ++i) {
            writeFloat32Value(output, buffer.left[i]);
            writeFloat32Value(output, buffer.right[i]);
        }
    }

    static void writePcm24(const std::string& path, const StereoBuffer& buffer,
                           int sampleRate) {
        if (sampleRate <= 0) {
            throw std::invalid_argument("WAV sample rate must be positive");
        }
        if (buffer.left.size() != buffer.right.size()) {
            throw std::invalid_argument("stereo WAV channels must have equal length");
        }

        const std::uint32_t channels = 2;
        const std::uint32_t bytesPerSample = 3;
        const std::uint64_t dataBytes64 =
            static_cast<std::uint64_t>(buffer.frames()) * channels * bytesPerSample;
        if (dataBytes64 > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("WAV input is too large for a RIFF file");
        }
        const std::uint32_t dataBytes = static_cast<std::uint32_t>(dataBytes64);

        std::ofstream output(path, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot open WAV file for writing: " + path);
        }

        writeTag(output, "RIFF");
        writeU32(output, 4u + (8u + 16u) + (8u + dataBytes));
        writeTag(output, "WAVE");
        writeTag(output, "fmt ");
        writeU32(output, 16u);
        writeU16(output, 1u);
        writeU16(output, static_cast<std::uint16_t>(channels));
        writeU32(output, static_cast<std::uint32_t>(sampleRate));
        writeU32(output, static_cast<std::uint32_t>(sampleRate) * channels * bytesPerSample);
        writeU16(output, static_cast<std::uint16_t>(channels * bytesPerSample));
        writeU16(output, static_cast<std::uint16_t>(bytesPerSample * 8));
        writeTag(output, "data");
        writeU32(output, dataBytes);
        for (std::size_t i = 0; i < buffer.frames(); ++i) {
            writePcm24Value(output, buffer.left[i]);
            writePcm24Value(output, buffer.right[i]);
        }
    }

private:
    static void writeTag(std::ofstream& output, const char* tag) {
        output.write(tag, 4);
    }

    static void writeU16(std::ofstream& output, std::uint16_t value) {
        const char bytes[2] = {
            static_cast<char>(value & 0xffu),
            static_cast<char>((value >> 8u) & 0xffu)};
        output.write(bytes, 2);
    }

    static void writeU32(std::ofstream& output, std::uint32_t value) {
        const char bytes[4] = {
            static_cast<char>(value & 0xffu),
            static_cast<char>((value >> 8u) & 0xffu),
            static_cast<char>((value >> 16u) & 0xffu),
            static_cast<char>((value >> 24u) & 0xffu)};
        output.write(bytes, 4);
    }

    static void writeFloat32Value(std::ofstream& output, float value) {
        std::uint32_t bits = 0;
        static_assert(sizeof(float) == sizeof(bits), "32-bit float required");
        std::memcpy(&bits, &value, sizeof(bits));
        writeU32(output, bits);
    }

    static void writePcm24Value(std::ofstream& output, float value) {
        const double clamped = std::clamp(static_cast<double>(value), -1.0, 1.0);
        const std::int32_t scaled =
            static_cast<std::int32_t>(std::lround(clamped * 8388607.0));
        const std::uint32_t bits = static_cast<std::uint32_t>(scaled) & 0x00ffffffu;
        const char bytes[3] = {
            static_cast<char>(bits & 0xffu),
            static_cast<char>((bits >> 8u) & 0xffu),
            static_cast<char>((bits >> 16u) & 0xffu)};
        output.write(bytes, 3);
    }
};

inline LimitedResult runLimiter(const StereoBuffer& input,
                                const tpl::PeakLimiterConfig& config) {
    tpl::TruePeakLimiter limiter(config);
    LimitedResult result{input, std::vector<float>(input.frames(), 1.0f)};
    limiter.process(result.audio.left.data(), result.audio.right.data(),
                    result.audio.frames(), result.gain.data());
    return result;
}

}  // namespace tpltest

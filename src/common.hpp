#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace aec {

constexpr double kPi = 3.14159265358979323846;
constexpr double kFloorDb = -120.0;

struct MonoAudio {
    int sample_rate = 0;
    std::vector<float> samples;
};

struct StereoAudio {
    int sample_rate = 0;
    std::vector<float> left;
    std::vector<float> right;
};

struct WavAudio {
    int sample_rate = 0;
    int channels = 0;
    std::vector<float> interleaved;

    std::size_t frames() const {
        return channels > 0 ? interleaved.size() / static_cast<std::size_t>(channels) : 0;
    }
};

inline double clamp01(double x) {
    return std::clamp(x, 0.0, 1.0);
}

inline double db_from_power(double power) {
    constexpr double kFloor = 1.0e-12;
    return 10.0 * std::log10(std::max(power, kFloor));
}

inline double power_to_db(double x, double reference = 1.0) {
    return 20.0 * std::log10(std::max(std::abs(x) / reference, 1.0e-6));
}

inline double rms(const std::vector<float>& x, std::size_t begin, std::size_t end) {
    begin = std::min(begin, x.size());
    end = std::min(end, x.size());
    if (end <= begin) {
        return 0.0;
    }
    long double sum = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
        sum += static_cast<long double>(x[i]) * x[i];
    }
    return std::sqrt(static_cast<double>(sum / static_cast<long double>(end - begin)));
}

inline double dot(const std::vector<float>& a, const std::vector<float>& b,
                  std::size_t begin, std::size_t end) {
    begin = std::min(begin, std::min(a.size(), b.size()));
    end = std::min(end, std::min(a.size(), b.size()));
    long double sum = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
        sum += static_cast<long double>(a[i]) * b[i];
    }
    return static_cast<double>(sum);
}

inline std::vector<float> slice(const std::vector<float>& x,
                                std::size_t begin, std::size_t end) {
    begin = std::min(begin, x.size());
    end = std::min(std::max(end, begin), x.size());
    return std::vector<float>(x.begin() + static_cast<std::ptrdiff_t>(begin),
                              x.begin() + static_cast<std::ptrdiff_t>(end));
}

inline double quantize_peak_scale(const std::vector<float>& x, double peak_dbfs = -1.0) {
    double peak = 0.0;
    for (float v : x) {
        peak = std::max(peak, static_cast<double>(std::abs(v)));
    }
    if (peak <= std::numeric_limits<double>::epsilon()) {
        return 1.0;
    }
    const double target = std::pow(10.0, peak_dbfs / 20.0);
    return target / peak;
}

}  // namespace aec

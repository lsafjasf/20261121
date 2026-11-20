#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tpl {

struct PeakLimiterConfig {
    double sampleRate = 48000.0;
    double ceilingDbTP = -1.0;
    double lookaheadMs = 3.0;
    double releaseMs = 90.0;
    double safetyMarginDb = 0.50;
    int oversampleFactor = 8;
    int detectorHalfWidth = 48;
    bool bypass = false;
};

class TruePeakLimiter {
public:
    explicit TruePeakLimiter(const PeakLimiterConfig& config = {}) {
        if (!configure(config)) {
            throw std::invalid_argument("invalid TruePeakLimiter configuration");
        }
    }

    bool configure(const PeakLimiterConfig& config) {
        if (!validConfig(config)) {
            return false;
        }
        config_ = config;
        rebuild();
        return true;
    }

    bool prepare(double sampleRate, int maxBlockFrames = 0) {
        (void)maxBlockFrames;
        if (!std::isfinite(sampleRate) || sampleRate < 8000.0 || sampleRate > 768000.0) {
            return false;
        }
        config_.sampleRate = sampleRate;
        rebuild();
        return true;
    }

    bool setSampleRate(double sampleRate) {
        if (!std::isfinite(sampleRate) || sampleRate < 8000.0 || sampleRate > 768000.0) {
            return false;
        }
        config_.sampleRate = sampleRate;
        rebuild();
        return true;
    }

    void reset() {
        std::fill(detectorL_.begin(), detectorL_.end(), 0.0f);
        std::fill(detectorR_.begin(), detectorR_.end(), 0.0f);
        std::fill(delayL_.begin(), delayL_.end(), 0.0f);
        std::fill(delayR_.begin(), delayR_.end(), 0.0f);
        std::fill(requiredGainRing_.begin(), requiredGainRing_.end(), 1.0f);
        minimumGains_.clear();
        inputCounter_ = 0;
        gain_ = 1.0f;
        targetGain_ = 1.0f;
        gainPrimed_ = false;
    }

    void setBypass(bool enabled) noexcept { bypass_ = enabled; }
    bool bypass() const noexcept { return bypass_; }

    bool setCeilingDbTP(double ceilingDb) {
        if (!std::isfinite(ceilingDb) || ceilingDb > 0.0 || ceilingDb < -60.0) {
            return false;
        }
        config_.ceilingDbTP = ceilingDb;
        ceilingLinear_ = dbToLinear(ceilingDb - config_.safetyMarginDb);
        return true;
    }

    double sampleRate() const noexcept { return config_.sampleRate; }
    double ceilingDbTP() const noexcept { return config_.ceilingDbTP; }
    int oversampleFactor() const noexcept { return config_.oversampleFactor; }
    int detectorHalfWidth() const noexcept { return config_.detectorHalfWidth; }
    int lookaheadSamples() const noexcept { return lookaheadSamples_; }
    int detectorDelaySamples() const noexcept { return config_.detectorHalfWidth; }
    int latencySamples() const noexcept { return latencySamples_; }
    double latencyMs() const noexcept {
        return 1000.0 * static_cast<double>(latencySamples_) / config_.sampleRate;
    }
    float currentGain() const noexcept { return gain_; }

    void process(float* left, float* right, std::size_t frames) {
        process(left, right, frames, nullptr);
    }

    void process(float* left, float* right, std::size_t frames, float* gainOut) {
        if (left == nullptr || right == nullptr) {
            if (frames != 0) {
                throw std::invalid_argument("TruePeakLimiter::process requires stereo buffers");
            }
            return;
        }

        for (std::size_t frame = 0; frame < frames; ++frame) {
            const float inL = left[frame];
            const float inR = right[frame];
            writeInput(inL, inR);

            const std::uint64_t signalIndex = inputCounter_ - static_cast<std::uint64_t>(config_.detectorHalfWidth);
            if (inputCounter_ >= static_cast<std::uint64_t>(config_.detectorHalfWidth)) {
                const float peak = detectTruePeakAtCurrentOutput();
                const float required = peak > ceilingLinear_ ? ceilingLinear_ / peak : 1.0f;
                requiredGainRing_[signalIndex % requiredGainRing_.size()] = required;
                pushMinimum(signalIndex, required);
            }

            float appliedGain = 1.0f;
            float outL = 0.0f;
            float outR = 0.0f;
            if (inputCounter_ >= static_cast<std::uint64_t>(latencySamples_)) {
                const std::uint64_t outputIndex =
                    inputCounter_ - static_cast<std::uint64_t>(latencySamples_);
                const std::size_t delaySlot = static_cast<std::size_t>(
                    outputIndex % static_cast<std::uint64_t>(delayL_.size()));
                outL = delayL_[delaySlot];
                outR = delayR_[delaySlot];
                appliedGain = updateGain(signalIndex, outputIndex);
            }

            left[frame] = bypass_ ? outL : outL * appliedGain;
            right[frame] = bypass_ ? outR : outR * appliedGain;

            if (gainOut != nullptr) {
                gainOut[frame] = appliedGain;
            }
            ++inputCounter_;
        }
    }

private:
    static constexpr double kPi = 3.141592653589793238462643383279502884;

    static double dbToLinear(double db) {
        return std::pow(10.0, db / 20.0);
    }

    static float dbToLinearF(double db) {
        return static_cast<float>(dbToLinear(db));
    }

    static bool validConfig(const PeakLimiterConfig& config) {
        return std::isfinite(config.sampleRate) &&
               config.sampleRate >= 8000.0 && config.sampleRate <= 768000.0 &&
               std::isfinite(config.ceilingDbTP) &&
               config.ceilingDbTP <= 0.0 && config.ceilingDbTP >= -60.0 &&
               std::isfinite(config.lookaheadMs) &&
               config.lookaheadMs >= 0.25 && config.lookaheadMs <= 50.0 &&
               std::isfinite(config.releaseMs) &&
               config.releaseMs >= 1.0 && config.releaseMs <= 2000.0 &&
               std::isfinite(config.safetyMarginDb) &&
               config.safetyMarginDb >= 0.0 && config.safetyMarginDb <= 3.0 &&
               (config.oversampleFactor == 4 || config.oversampleFactor == 8) &&
               config.detectorHalfWidth >= 4 && config.detectorHalfWidth <= 128;
    }

    void rebuild() {
        const int factor = config_.oversampleFactor;
        const int halfWidth = config_.detectorHalfWidth;
        const int phaseCount = factor;
        const int taps = 2 * halfWidth + 1;

        coefficients_.assign(static_cast<std::size_t>(phaseCount), {});
        detectorL_.assign(static_cast<std::size_t>(2 * halfWidth + 1), 0.0f);
        detectorR_.assign(static_cast<std::size_t>(2 * halfWidth + 1), 0.0f);

        for (int phase = 0; phase < phaseCount; ++phase) {
            auto& row = coefficients_[static_cast<std::size_t>(phase)];
            row.resize(static_cast<std::size_t>(taps), 0.0);
            const double fraction =
                static_cast<double>(phase) / static_cast<double>(factor);
            double sum = 0.0;
            for (int tap = 0; tap < taps; ++tap) {
                const double distance =
                    static_cast<double>(tap - halfWidth) - fraction;
                const double sinc = nearlyEqual(distance, 0.0)
                                        ? 1.0
                                        : std::sin(kPi * distance) /
                                              (kPi * distance);
                const double windowPosition =
                    distance / static_cast<double>(halfWidth);
                const double radial =
                    std::sqrt(std::max(0.0, 1.0 - windowPosition * windowPosition));
                constexpr double windowBeta = 4.0;
                const double window = modifiedBesselI0(windowBeta * radial) /
                                      modifiedBesselI0(windowBeta);
                const double coefficient = sinc * window;
                row[static_cast<std::size_t>(tap)] = coefficient;
                sum += coefficient;
            }
            if (sum != 0.0) {
                for (double& value : row) {
                    value /= sum;
                }
            }
        }

        lookaheadSamples_ = std::max(
            1, static_cast<int>(std::lround(config_.lookaheadMs * config_.sampleRate / 1000.0)));
        latencySamples_ = lookaheadSamples_ + halfWidth;
        requiredGainRing_.assign(
            static_cast<std::size_t>(lookaheadSamples_ + 2), 1.0f);
        minimumGains_.assign(static_cast<std::size_t>(lookaheadSamples_ + 2));
        delayL_.assign(static_cast<std::size_t>(latencySamples_ + 1), 0.0f);
        delayR_.assign(static_cast<std::size_t>(latencySamples_ + 1), 0.0f);
        ceilingLinear_ = dbToLinearF(config_.ceilingDbTP - config_.safetyMarginDb);
        attackCoefficient_ = static_cast<float>(
            std::exp(-1.0 / (0.00025 * config_.sampleRate)));
        releaseCoefficient_ = static_cast<float>(
            std::exp(-1.0 / (config_.releaseMs * 0.001 * config_.sampleRate)));
        bypass_ = config_.bypass;
        reset();
    }

    static bool nearlyEqual(double a, double b) {
        return std::abs(a - b) < 1.0e-12;
    }

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

    void writeInput(float inL, float inR) {
        const std::size_t detectorSlot =
            static_cast<std::size_t>(inputCounter_ % detectorL_.size());
        detectorL_[detectorSlot] = inL;
        detectorR_[detectorSlot] = inR;
        const std::size_t delaySlot =
            static_cast<std::size_t>(inputCounter_ % delayL_.size());
        delayL_[delaySlot] = inL;
        delayR_[delaySlot] = inR;
    }

    float detectTruePeakAtCurrentOutput() const {
        float peak = 0.0f;
        const int halfWidth = config_.detectorHalfWidth;
        const int factor = config_.oversampleFactor;
        const std::size_t ringSize = detectorL_.size();
        const std::uint64_t signalIndex =
            inputCounter_ - static_cast<std::uint64_t>(halfWidth);

        for (int phase = 0; phase < factor; ++phase) {
            const auto& row = coefficients_[static_cast<std::size_t>(phase)];
            float valueL = 0.0f;
            float valueR = 0.0f;
            for (int tap = 0; tap <= 2 * halfWidth; ++tap) {
                if (inputCounter_ + static_cast<std::uint64_t>(tap) <
                    static_cast<std::uint64_t>(2 * halfWidth)) {
                    continue;
                }
                const std::uint64_t sampleIndex =
                    inputCounter_ + static_cast<std::uint64_t>(tap) -
                    static_cast<std::uint64_t>(2 * halfWidth);
                const std::size_t slot =
                    static_cast<std::size_t>(sampleIndex % ringSize);
                const float coefficient = static_cast<float>(row[static_cast<std::size_t>(tap)]);
                valueL += coefficient * detectorL_[slot];
                valueR += coefficient * detectorR_[slot];
            }
            peak = std::max(peak, std::abs(valueL));
            peak = std::max(peak, std::abs(valueR));
        }

        const std::size_t centerSlot = static_cast<std::size_t>(
            signalIndex % ringSize);
        peak = std::max(peak, std::abs(detectorL_[centerSlot]));
        peak = std::max(peak, std::abs(detectorR_[centerSlot]));
        return peak;
    }

    void pushMinimum(std::uint64_t index, float value) {
        while (!minimumGains_.empty() && minimumGains_.back().value >= value) {
            minimumGains_.popBack();
        }
        minimumGains_.pushBack({index, value});
        const std::uint64_t oldest =
            index >= static_cast<std::uint64_t>(lookaheadSamples_)
                ? index - static_cast<std::uint64_t>(lookaheadSamples_)
                : 0;
        while (!minimumGains_.empty() && minimumGains_.front().index < oldest) {
            minimumGains_.popFront();
        }
    }

    float updateGain(std::uint64_t signalIndex, std::uint64_t outputIndex) {
        const float futureMinimum = minimumGains_.empty()
                                        ? 1.0f
                                        : minimumGains_.front().value;
        targetGain_ = futureMinimum;
        const float requiredNow =
            requiredGainRing_[outputIndex % requiredGainRing_.size()];
        if (!gainPrimed_) {
            gain_ = std::min(targetGain_, requiredNow);
            gainPrimed_ = true;
        } else if (targetGain_ < gain_) {
            gain_ = attackCoefficient_ * gain_ + (1.0f - attackCoefficient_) * targetGain_;
        } else {
            gain_ = releaseCoefficient_ * gain_ + (1.0f - releaseCoefficient_) * targetGain_;
        }

        gain_ = std::clamp(gain_, 0.0f, 1.0f);
        (void)signalIndex;
        return gain_;
    }

    PeakLimiterConfig config_{};
    bool bypass_ = false;
    float ceilingLinear_ = 0.0f;
    float attackCoefficient_ = 0.0f;
    float releaseCoefficient_ = 0.0f;
    float gain_ = 1.0f;
    float targetGain_ = 1.0f;
    bool gainPrimed_ = false;
    int lookaheadSamples_ = 0;
    int latencySamples_ = 0;
    std::uint64_t inputCounter_ = 0;
    std::vector<std::vector<double>> coefficients_;
    std::vector<float> detectorL_;
    std::vector<float> detectorR_;
    std::vector<float> delayL_;
    std::vector<float> delayR_;
    std::vector<float> requiredGainRing_;

    struct MinimumEntry {
        std::uint64_t index;
        float value;
    };

    class MinimumQueue {
    public:
        void assign(std::size_t capacity) {
            data_.assign(capacity == 0 ? 1 : capacity, {});
            head_ = 0;
            count_ = 0;
        }

        void clear() noexcept {
            head_ = 0;
            count_ = 0;
        }

        bool empty() const noexcept { return count_ == 0; }

        const MinimumEntry& front() const noexcept {
            return data_[head_];
        }

        const MinimumEntry& back() const noexcept {
            return data_[(head_ + count_ - 1) % data_.size()];
        }

        void popFront() noexcept {
            head_ = (head_ + 1) % data_.size();
            --count_;
        }

        void popBack() noexcept { --count_; }

        void pushBack(const MinimumEntry& entry) {
            data_[(head_ + count_) % data_.size()] = entry;
            ++count_;
        }

    private:
        std::vector<MinimumEntry> data_;
        std::size_t head_ = 0;
        std::size_t count_ = 0;
    };

    MinimumQueue minimumGains_;
};

}  // namespace tpl

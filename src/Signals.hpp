#pragma once

#include "TestSupport.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

namespace tpltest {

inline std::size_t framesFor(double sampleRate, double seconds) {
    return static_cast<std::size_t>(std::llround(sampleRate * seconds));
}

inline StereoBuffer makeNearNyquistProbe(double sampleRate, double seconds = 1.25) {
    const std::size_t frames = framesFor(sampleRate, seconds);
    StereoBuffer buffer{std::vector<float>(frames), std::vector<float>(frames)};
    const double f1 = 0.484375 * sampleRate;
    const double f2 = 0.453125 * sampleRate;
    const double phase1 = 3.4033920413889427;
    const double phase2 = 3.9269908169872414;
    const double scale = 1.25;
    const double fadeFrames = 0.035 * sampleRate;

    for (std::size_t i = 0; i < frames; ++i) {
        const double time = static_cast<double>(i) / sampleRate;
        const double envelope = std::min(
            1.0, std::min(static_cast<double>(i) / fadeFrames,
                          static_cast<double>(frames - 1 - i) / fadeFrames));
        const double left =
            scale * envelope *
            (0.80 * std::sin(2.0 * kPi * f1 * time + phase1) +
             0.52 * std::sin(2.0 * kPi * f2 * time + phase2));
        const double right = left;
        buffer.left[i] = static_cast<float>(left);
        buffer.right[i] = static_cast<float>(right);
    }
    return buffer;
}

inline StereoBuffer makeAlternatingHits(double sampleRate, double seconds = 2.55) {
    const std::size_t frames = framesFor(sampleRate, seconds);
    StereoBuffer buffer{std::vector<float>(frames), std::vector<float>(frames)};
    const double pilotLeftHz = 997.0;
    const double pilotRightHz = 1493.0;
    const double hitFrequency = 73.0;
    const double firstHit = 0.24;
    const double hitSpacing = 0.29;
    const double hitDecay = 0.055;
    const double hitAttack = 0.0025;
    const double hitAmplitude = 2.05;

    for (std::size_t i = 0; i < frames; ++i) {
        const double time = static_cast<double>(i) / sampleRate;
        double left = 0.045 * std::sin(2.0 * kPi * pilotLeftHz * time);
        double right = 0.040 * std::sin(2.0 * kPi * pilotRightHz * time);
        if (time >= firstHit) {
            const double hitPosition = (time - firstHit) / hitSpacing;
            const long long hitNumber = static_cast<long long>(std::floor(hitPosition));
            const double localTime = time - (firstHit + hitNumber * hitSpacing);
            if (localTime < 0.22) {
                const double attack = std::min(1.0, localTime / hitAttack);
                const double envelope =
                    attack * std::exp(-localTime / hitDecay);
                double hit =
                    hitAmplitude * envelope *
                    std::sin(2.0 * kPi * hitFrequency * localTime);
                hit += 0.22 * hitAmplitude * envelope *
                       std::sin(2.0 * kPi * 431.0 * localTime);
                if ((hitNumber & 1LL) == 0) {
                    left += hit;
                } else {
                    right += hit;
                }
            }
        }
        buffer.left[i] = static_cast<float>(left);
        buffer.right[i] = static_cast<float>(right);
    }
    return buffer;
}

inline StereoBuffer makeAlignmentProbe(double sampleRate, double seconds = 0.35) {
    const std::size_t frames = framesFor(sampleRate, seconds);
    StereoBuffer buffer{std::vector<float>(frames), std::vector<float>(frames)};
    const std::size_t impulse1 = frames / 5;
    const std::size_t impulse2 = frames * 3 / 5;

    for (std::size_t i = 0; i < frames; ++i) {
        const double time = static_cast<double>(i) / sampleRate;
        double left = 0.12 * std::sin(2.0 * kPi * 113.0 * time) +
                      0.05 * std::sin(2.0 * kPi * 887.0 * time + 0.2);
        double right = 0.14 * std::sin(2.0 * kPi * 257.0 * time + 0.7) +
                       0.04 * std::sin(2.0 * kPi * 1297.0 * time);
        if (i == impulse1) {
            left += 0.62;
            right -= 0.43;
        }
        if (i == impulse2) {
            left -= 0.39;
            right += 0.58;
        }
        buffer.left[i] = static_cast<float>(left);
        buffer.right[i] = static_cast<float>(right);
    }
    return buffer;
}

inline StereoBuffer delayedCopy(const StereoBuffer& input, int latencySamples) {
    StereoBuffer output{std::vector<float>(input.frames()),
                        std::vector<float>(input.frames())};
    for (std::size_t i = 0; i < input.frames(); ++i) {
        if (i >= static_cast<std::size_t>(latencySamples)) {
            output.left[i] = input.left[i - static_cast<std::size_t>(latencySamples)];
            output.right[i] = input.right[i - static_cast<std::size_t>(latencySamples)];
        }
    }
    return output;
}

inline double maxAbsDifference(const StereoBuffer& a, const StereoBuffer& b) {
    if (a.frames() != b.frames()) {
        return std::numeric_limits<double>::infinity();
    }
    double difference = 0.0;
    for (std::size_t i = 0; i < a.frames(); ++i) {
        difference = std::max(
            difference,
            std::abs(static_cast<double>(a.left[i]) - b.left[i]));
        difference = std::max(
            difference,
            std::abs(static_cast<double>(a.right[i]) - b.right[i]));
    }
    return difference;
}

}  // namespace tpltest

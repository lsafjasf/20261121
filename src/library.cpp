#include "footbatch/library.hpp"

#include <algorithm>
#include <cmath>
#include "footbatch/constants.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace footbatch {
namespace {

uint64_t splitMix64(uint64_t& state) {
    state += 0x9e3779b97f4a7c15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

double unitRandom(uint64_t& state) {
    return static_cast<double>(splitMix64(state) >> 11) / static_cast<double>(1ull << 53);
}

uint64_t seedFor(const std::string& tag) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (char c : tag) {
        h ^= static_cast<unsigned char>(c);
        h *= 0x100000001b3ull;
    }
    return h;
}

double lowpassCoeff(double cutoffHz, int sampleRate) {
    double rc = 1.0 / (2.0 * kPi * std::clamp(cutoffHz, 20.0, 20000.0));
    double dt = 1.0 / sampleRate;
    return dt / (rc + dt);
}

// Surface families change the spectral character of the synthesized step.
struct SurfaceTone {
    double bodyHz;      // tonal body frequency
    double noiseMix;   // 0 = tonal, 1 = white noise
    double decay;      // amplitude decay factor per second
};

SurfaceTone toneFor(const std::string& surface) {
    if (surface == "concrete") return {180.0, 0.75, 7.0};
    if (surface == "wood")     return {320.0, 0.45, 5.5};
    if (surface == "grass")    return {90.0,  0.92, 9.0};
    if (surface == "metal")    return {680.0, 0.30, 4.0};
    if (surface == "water")    return {140.0, 0.85, 3.0};
    return {220.0, 0.6, 6.0};
}

WavData synthesizeStep(const StrategyConfig& cfg, const SurfaceConfig& surface,
                       const VelocityLayer& layer, int variant) {
    const double durationSec = 0.6;
    const size_t frames = static_cast<size_t>(cfg.sampleRate * durationSec);
    uint64_t rng = seedFor(surface.name + "|" + layer.name + "|" + std::to_string(variant));
    SurfaceTone tone = toneFor(surface.name);
    double alpha = lowpassCoeff(surface.lowpassHz, cfg.sampleRate);

    WavData audio;
    audio.sampleRate = cfg.sampleRate;
    audio.channels = 1;
    audio.interleaved.resize(frames);

    double phase = 0.0;
    double filtered = 0.0;
    double velocityLevel = 0.25 + 0.75 * ((layer.minVelocity + layer.maxVelocity) * 0.5);
    for (size_t n = 0; n < frames; ++n) {
        double t = static_cast<double>(n) / cfg.sampleRate;
        double white = unitRandom(rng) * 2.0 - 1.0;
        double body = std::sin(phase);
        phase += 2.0 * kPi * tone.bodyHz / cfg.sampleRate;
        double mixed = white * tone.noiseMix + body * (1.0 - tone.noiseMix);
        filtered += alpha * (mixed - filtered);
        double attack = std::min(1.0, t / 0.004);
        double envelope = attack * std::exp(-tone.decay * t);
        audio.interleaved[n] = static_cast<float>(filtered * envelope * velocityLevel * 0.8);
    }
    return audio;
}

std::string samplePath(const std::string& root, const std::string& surface,
                       const std::string& layer, int variant) {
    return root + "/" + surface + "/" + layer + "_v" + std::to_string(variant) + ".wav";
}

}  // namespace

void generateLibrary(const StrategyConfig& cfg) {
    std::filesystem::create_directories(cfg.libraryRoot);
    for (const SurfaceConfig& surface : cfg.surfaces) {
        std::filesystem::create_directories(cfg.libraryRoot + "/" + surface.name);
        for (const VelocityLayer& layer : cfg.layers) {
            for (int variant = 0; variant < surface.variantsPerLayer; ++variant) {
                WavData audio = synthesizeStep(cfg, surface, layer, variant);
                writeWavPcm16(samplePath(cfg.libraryRoot, surface.name, layer.name, variant),
                              audio, 0);
            }
        }
    }
}

void SampleLibrary::load(const StrategyConfig& cfg) {
    for (const SurfaceConfig& surface : cfg.surfaces) {
        for (const VelocityLayer& layer : cfg.layers) {
            for (int variant = 0; variant < surface.variantsPerLayer; ++variant) {
                SampleKey key{surface.name, layer.name, variant};
                std::string path =
                    samplePath(cfg.libraryRoot, surface.name, layer.name, variant);
                Sample sample;
                sample.key = key;
                sample.path = path;
                sample.audio = readWav(path);
                if (sample.audio.sampleRate != cfg.sampleRate) {
                    throw std::runtime_error("sample rate mismatch in " + path);
                }
                samples_.emplace(key, std::move(sample));
            }
        }
    }
}

const Sample& SampleLibrary::get(const SampleKey& key) const {
    auto it = samples_.find(key);
    if (it == samples_.end()) throw std::runtime_error("missing sample: " + key.id());
    return it->second;
}

}  // namespace footbatch

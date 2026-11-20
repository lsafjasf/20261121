#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "footbatch/strategy.hpp"
#include "footbatch/wav.hpp"

namespace footbatch {

struct SampleKey {
    std::string surface;
    std::string layer;
    int variant = 0;

    std::string id() const { return surface + "/" + layer + "#" + std::to_string(variant); }
    bool operator==(const SampleKey& other) const {
        return surface == other.surface && layer == other.layer && variant == other.variant;
    }
};

struct SampleKeyHash {
    size_t operator()(const SampleKey& k) const {
        size_t h = std::hash<std::string>{}(k.surface);
        h ^= std::hash<std::string>{}(k.layer) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(k.variant) * 0x9e3779b9u;
        return h;
    }
};

struct Sample {
    SampleKey key;
    std::string path;
    WavData audio;
};

class SampleLibrary {
public:
    void load(const StrategyConfig& cfg);
    const Sample& get(const SampleKey& key) const;
    size_t size() const { return samples_.size(); }

private:
    std::unordered_map<SampleKey, Sample, SampleKeyHash> samples_;
};

// Builds a deterministic synthetic footstep library on disk.
void generateLibrary(const StrategyConfig& cfg);

}  // namespace footbatch

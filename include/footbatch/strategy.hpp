#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "footbatch/json.hpp"

namespace footbatch {

struct VelocityLayer {
    std::string name;
    double minVelocity = 0.0;  // inclusive lower bound, range 0..1
    double maxVelocity = 1.0;  // inclusive upper bound
    double gainDb = 0.0;
};

struct SurfaceConfig {
    std::string name;
    double pitchSemitones = 0.0;   // surface-level pitch offset
    double lowpassHz = 20000.0;    // one-pole filter cutoff
    int variantsPerLayer = 1;
};

struct StrategyConfig {
    int sampleRate = 48000;
    int blockFrames = 256;
    std::string channelLayout = "stereo";  // "mono" or "stereo"
    std::string libraryRoot;
    std::vector<VelocityLayer> layers;
    std::vector<SurfaceConfig> surfaces;

    // Continuous-trigger interval encoding.
    double overlapWindowSec = 0.18;
    double overlapDuckDb = -4.0;       // applied to the new voice on overlap
    double tailPaddingSec = 0.05;      // always render this much tail after last event
    int maxVoices = 32;

    // Replayable deformation ranges (per event).
    double pitchJitterSemitones = 0.25;
    double gainJitterDb = 1.5;
    double panJitter = 0.15;
    uint64_t baseSeed = 0x464f4f5453454544ull;  // "FOOTSEED"

    int channels() const { return channelLayout == "mono" ? 1 : 2; }
    const VelocityLayer* findLayer(double velocity) const;
    const SurfaceConfig* findSurface(const std::string& name) const;
};

struct FootEvent {
    int index = 0;            // position in the input array
    double timeSec = 0.0;
    double velocity = 0.0;
    std::string surface;
    std::string foot;         // "", "L", "R"
};

struct ClipSpec {
    std::string clipId;
    std::vector<FootEvent> events;
};

struct BatchSpec {
    std::vector<ClipSpec> clips;
};

StrategyConfig loadStrategy(const std::string& path);
BatchSpec loadBatch(const std::string& path);
std::vector<FootEvent> loadEvents(const std::string& path);

// Stable order used by the renderer: time first, original input order second.
std::vector<FootEvent> sortEvents(std::vector<FootEvent> events);

}  // namespace footbatch

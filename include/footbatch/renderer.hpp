#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "footbatch/json.hpp"
#include "footbatch/library.hpp"
#include "footbatch/strategy.hpp"

namespace footbatch {

struct SelectionParams {
    SampleKey sample;
    std::string foot;
    double gapSec = 0.0;
    bool overlapped = false;
    double pitchRatio = 1.0;
    double totalGain = 1.0;
    double lowpassAlpha = 1.0;
    double pan = 0.0;          // -1 left, +1 right
    double duckGain = 1.0;
    uint64_t rngSeed = 0;
};

struct RenderResult {
    std::string wavPath;
    std::string manifestPath;
    int eventCount = 0;
    int peakVoices = 0;
    bool resumed = false;
    int checkpointResumes = 0;
};

class Renderer {
public:
    Renderer(StrategyConfig cfg, const SampleLibrary& library, std::string outDir);

    RenderResult renderClip(const std::string& clipId, std::vector<FootEvent> events,
                            const std::string& checkpointPath, bool resume);

private:
    struct Voice {
        bool active = false;
        const Sample* sample = nullptr;
        SelectionParams params;
        double position = 0.0;
        double filterState = 0.0;
        size_t startFrame = 0;
    };

    SelectionParams select(const ClipSpec& clip, const FootEvent& event,
                           const FootEvent* previous, int chronologicalIndex) const;
    std::vector<float> renderBlock(size_t blockStart, size_t blockFrames);
    Json buildManifest(const std::string& clipId, size_t totalFrames);
    void writeCheckpoint(const std::string& path, size_t nextFrame,
                         int nextEvent, int peakVoices, int resumes);

    StrategyConfig cfg_;
    const SampleLibrary& library_;
    std::string outDir_;
    std::string clipId_;
    std::vector<FootEvent> events_;
    std::vector<Voice> voices_;
    std::vector<SelectionParams> selections_;
    size_t nextFrame_ = 0;
    int nextEvent_ = 0;
    int peakVoices_ = 0;
    bool voiceLimitExceeded_ = false;
};

}  // namespace footbatch

#include "footbatch/strategy.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace footbatch {
namespace {

std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw JsonError("cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

const VelocityLayer* StrategyConfig::findLayer(double velocity) const {
    const VelocityLayer* fallback = nullptr;
    for (const VelocityLayer& layer : layers) {
        if (velocity >= layer.minVelocity && velocity <= layer.maxVelocity) {
            return &layer;
    }
        if (velocity >= layer.minVelocity && (!fallback || layer.maxVelocity > fallback->maxVelocity)) {
            fallback = &layer;
    }
    }
    return fallback;
}

const SurfaceConfig* StrategyConfig::findSurface(const std::string& name) const {
    for (const SurfaceConfig& surface : surfaces) {
        if (surface.name == name) return &surface;
    }
    return nullptr;
}

StrategyConfig loadStrategy(const std::string& path) {
    Json j = Json::parse(readFile(path));
    StrategyConfig cfg;
    cfg.sampleRate = j.contains("sample_rate") ? j.at("sample_rate").asInt() : 48000;
    cfg.blockFrames = j.contains("block_frames") ? j.at("block_frames").asInt() : 256;
    cfg.channelLayout = j.stringOr("channel_layout", "stereo");
    if (cfg.channelLayout != "mono" && cfg.channelLayout != "stereo") {
        throw JsonError("channel_layout must be mono or stereo");
    }
    cfg.libraryRoot = j.at("library_root").asString();

    if (j.contains("layers")) {
        for (const Json& lj : j.at("layers").array()) {
            VelocityLayer layer;
            layer.name = lj.at("name").asString();
            layer.minVelocity = lj.at("min_velocity").asNumber();
            layer.maxVelocity = lj.at("max_velocity").asNumber();
            layer.gainDb = lj.numberOr("gain_db", 0.0);
            cfg.layers.push_back(layer);
    }
    }
    if (j.contains("surfaces")) {
        for (const Json& sj : j.at("surfaces").array()) {
            SurfaceConfig surface;
            surface.name = sj.at("name").asString();
            surface.pitchSemitones = sj.numberOr("pitch_semitones", 0.0);
            surface.lowpassHz = sj.numberOr("lowpass_hz", 20000.0);
            surface.variantsPerLayer = sj.contains("variants_per_layer")
                                           ? sj.at("variants_per_layer").asInt()
                                           : 1;
            cfg.surfaces.push_back(surface);
    }
    }

    if (j.contains("interval")) {
        const Json& ij = j.at("interval");
        cfg.overlapWindowSec = ij.numberOr("overlap_window_sec", cfg.overlapWindowSec);
        cfg.overlapDuckDb = ij.numberOr("overlap_duck_db", cfg.overlapDuckDb);
        cfg.tailPaddingSec = ij.numberOr("tail_padding_sec", cfg.tailPaddingSec);
        cfg.maxVoices = ij.contains("max_voices") ? ij.at("max_voices").asInt() : cfg.maxVoices;
    }
    if (j.contains("deformation")) {
        const Json& dj = j.at("deformation");
        cfg.pitchJitterSemitones = dj.numberOr("pitch_jitter_semitones", cfg.pitchJitterSemitones);
        cfg.gainJitterDb = dj.numberOr("gain_jitter_db", cfg.gainJitterDb);
        cfg.panJitter = dj.numberOr("pan_jitter", cfg.panJitter);
        if (dj.contains("base_seed")) cfg.baseSeed = static_cast<uint64_t>(dj.at("base_seed").asNumber());
    }
    if (cfg.layers.empty()) throw JsonError("strategy must define at least one velocity layer");
    if (cfg.surfaces.empty()) throw JsonError("strategy must define at least one surface");
    return cfg;
}

std::vector<FootEvent> parseEvents(const Json& arr) {
    std::vector<FootEvent> events;
    int index = 0;
    for (const Json& ej : arr.array()) {
        FootEvent e;
        e.index = index++;
        e.timeSec = ej.at("time_sec").asNumber();
        e.velocity = ej.at("velocity").asNumber();
        e.surface = ej.at("surface").asString();
        e.foot = ej.stringOr("foot", "");
        if (e.velocity < 0.0 || e.velocity > 1.0) {
            throw JsonError("event velocity out of range [0,1]");
    }
        events.push_back(e);
    }
    return events;
}

std::vector<FootEvent> loadEvents(const std::string& path) {
    Json j = Json::parse(readFile(path));
    const Json& arr = j.contains("events") ? j.at("events") : j;
    if (!arr.isArray()) throw JsonError("event file must contain an events array");
    return parseEvents(arr);
}

BatchSpec loadBatch(const std::string& path) {
    Json j = Json::parse(readFile(path));
    BatchSpec batch;
    for (const Json& cj : j.at("clips").array()) {
        ClipSpec clip;
        clip.clipId = cj.at("clip_id").asString();
        clip.events = parseEvents(cj.at("events"));
        batch.clips.push_back(std::move(clip));
    }
    if (batch.clips.empty()) throw JsonError("batch must contain at least one clip");
    return batch;
}

std::vector<FootEvent> sortEvents(std::vector<FootEvent> events) {
    std::stable_sort(events.begin(), events.end(),
                     [](const FootEvent& a, const FootEvent& b) {
                         if (a.timeSec != b.timeSec) return a.timeSec < b.timeSec;
                         return a.index < b.index;
                     });
    return events;
}

}  // namespace footbatch

#include "footbatch/renderer.hpp"

#include <algorithm>
#include <cmath>
#include "footbatch/constants.hpp"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace footbatch {
namespace {

constexpr char kCheckpointMagic[8] = {'F', 'B', 'C', 'K', 'P', 'T', '1', '0'};

uint64_t splitMix64(uint64_t& state) {
    state += 0x9e3779b97f4a7c15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

double uniform(uint64_t& state) {
    return static_cast<double>(splitMix64(state) >> 11) /
           static_cast<double>(1ull << 53);
}

double dbToGain(double db) { return std::pow(10.0, db / 20.0); }

uint64_t hashString(const std::string& s) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (char c : s) {
        h ^= static_cast<unsigned char>(c);
        h *= 0x100000001b3ull;
    }
    return h;
}

double lowpassAlpha(double cutoffHz, int sampleRate) {
    double rc = 1.0 / (2.0 * kPi * std::clamp(cutoffHz, 20.0, 20000.0));
    double dt = 1.0 / sampleRate;
    return dt / (rc + dt);
}

void appendRaw(const std::string& path, const float* data, size_t count) {
    std::ofstream f(path, std::ios::binary | std::ios::app);
 if (!f) throw std::runtime_error("cannot write raw sidecar: " + path);
    f.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(count * sizeof(float)));
}

}  // namespace

Renderer::Renderer(StrategyConfig cfg, const SampleLibrary& library, std::string outDir)
    : cfg_(std::move(cfg)), library_(library), outDir_(std::move(outDir)) {
    std::filesystem::create_directories(outDir_);
}

SelectionParams Renderer::select(const ClipSpec& clip, const FootEvent& event,
                                 const FootEvent* previous, int chronologicalIndex) const {
    const VelocityLayer* layer = cfg_.findLayer(event.velocity);
    if (!layer) throw std::runtime_error("no velocity layer matches velocity");
    const SurfaceConfig* surface = cfg_.findSurface(event.surface);
    if (!surface) throw std::runtime_error("unknown surface type: " + event.surface);

    SelectionParams p;
    p.sample.surface = surface->name;
    p.sample.layer = layer->name;

    uint64_t rng = cfg_.baseSeed ^ hashString(clip.clipId) ^
                   (static_cast<uint64_t>(event.index) * 0x9e3779b97f4a7c15ull);
    p.rngSeed = rng;
    if (surface->variantsPerLayer > 1) {
        p.sample.variant = static_cast<int>(uniform(rng) * surface->variantsPerLayer);
        p.sample.variant = std::min(p.sample.variant, surface->variantsPerLayer - 1);
    }

    // Alternating foot order: explicit marker wins, otherwise L/R alternation
    // keyed on chronological position (stable under resume).
    if (!event.foot.empty()) {
        p.foot = event.foot;
    } else {
        p.foot = (chronologicalIndex % 2 == 0) ? "L" : "R";
    }

    p.gapSec = previous ? event.timeSec - previous->timeSec : 0.0;
    p.overlapped = previous && p.gapSec < cfg_.overlapWindowSec;
    p.duckGain = p.overlapped ? dbToGain(cfg_.overlapDuckDb) : 1.0;

    double pitchJitter = (uniform(rng) * 2.0 - 1.0) * cfg_.pitchJitterSemitones;
    double gainJitterDb = (uniform(rng) * 2.0 - 1.0) * cfg_.gainJitterDb;
    double jitter = (uniform(rng) * 2.0 - 1.0) * cfg_.panJitter;

    double semitones = surface->pitchSemitones + pitchJitter;
    p.pitchRatio = std::pow(2.0, semitones / 12.0);
    p.totalGain = dbToGain(layer->gainDb + gainJitterDb) * p.duckGain;
    p.lowpassAlpha = lowpassAlpha(surface->lowpassHz, cfg_.sampleRate);
    p.pan = std::clamp((p.foot == "R" ? 0.25 : -0.25) + jitter, -1.0, 1.0);
    return p;
}

std::vector<float> Renderer::renderBlock(size_t blockStart, size_t blockFrames) {
    const int channels = cfg_.channels();
    std::vector<float> buffer(blockFrames * static_cast<size_t>(channels), 0.0f);

    // Activate every event whose frame lands in this block. Simultaneous events
    // all start on the same frame and share the mix without eviction.
    while (nextEvent_ < static_cast<int>(events_.size())) {
        const FootEvent& event = events_[nextEvent_];
        size_t frame = static_cast<size_t>(std::llround(event.timeSec * cfg_.sampleRate));
        if (frame >= blockStart + blockFrames) break;

        ClipSpec clip;
        clip.clipId = clipId_;
        const FootEvent* previous = nextEvent_ > 0 ? &events_[nextEvent_ - 1] : nullptr;
        SelectionParams params = select(clip, event, previous, nextEvent_);
        selections_.push_back(params);

        Voice voice;
        voice.active = true;
        voice.sample = &library_.get(params.sample);
        voice.params = params;
        voice.position = 0.0;
        voice.filterState = 0.0;
        voice.startFrame = frame;
        voices_.push_back(voice);
        ++nextEvent_;
    }

    int activeNow = 0;
    for (Voice& voice : voices_) {
        if (!voice.active) continue;
        const WavData& src = voice.sample->audio;
        for (size_t n = 0; n < blockFrames; ++n) {
            size_t globalFrame = blockStart + n;
            if (globalFrame < voice.startFrame) continue;
            size_t idx = static_cast<size_t>(voice.position);
            if (idx >= src.frames()) {
                voice.active = false;
                break;
            }
            double frac = voice.position - idx;
            size_t next = std::min(idx + 1, src.frames() - 1);
            float s0 = src.interleaved[idx];
            float s1 = src.interleaved[next];
            float input = static_cast<float>(s0 + (s1 - s0) * frac);
            voice.filterState += voice.params.lowpassAlpha *
                                 (static_cast<double>(input) - voice.filterState);
            float value = static_cast<float>(voice.filterState) *
                          static_cast<float>(voice.params.totalGain);
            if (channels == 1) {
                buffer[n] += value;
            } else {
                float left = static_cast<float>(std::clamp(1.0 - voice.params.pan, 0.0, 1.0));
                float right = static_cast<float>(std::clamp(1.0 + voice.params.pan, 0.0, 1.0));
                buffer[n * 2] += value * left;
                buffer[n * 2 + 1] += value * right;
            }
            voice.position += voice.params.pitchRatio;
        }
        if (voice.active) ++activeNow;
    }
    peakVoices_ = std::max(peakVoices_, activeNow);
    if (activeNow > cfg_.maxVoices) voiceLimitExceeded_ = true;
    return buffer;
}

void Renderer::writeCheckpoint(const std::string& path, size_t nextFrame,
                               int nextEvent, int peakVoices, int resumes) {
    std::ofstream f(path + ".tmp", std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot write checkpoint: " + path);
    auto u32 = [&](uint32_t v) {
        char b[4] = {static_cast<char>(v), static_cast<char>(v >> 8),
                     static_cast<char>(v >> 16), static_cast<char>(v >> 24)};
        f.write(b, 4);
    };
    auto u64 = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i) f.put(static_cast<char>(v >> (8 * i)));
    };
    f.write(kCheckpointMagic, 8);
    u32(1);  // version
    u64(nextFrame);
    u32(static_cast<uint32_t>(nextEvent));
    u32(static_cast<uint32_t>(peakVoices));
    u32(static_cast<uint32_t>(resumes));
    u64(selections_.size());
    if (!f) throw std::runtime_error("checkpoint write failed: " + path);
    f.close();
    std::filesystem::rename(path + ".tmp", path);
}

namespace {

uint32_t readU32(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
           (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
}

uint64_t readU64(std::ifstream& f) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(f.get()) << (8 * i);
    return v;
}

Json selectionToJson(const SelectionParams& p) {
    Json j = Json::makeObject();
    Json src = Json::makeObject();
    src.set("surface", Json::makeString(p.sample.surface));
    src.set("layer", Json::makeString(p.sample.layer));
    src.set("variant", Json::makeNumber(p.sample.variant));
    j.set("sample", src);
    j.set("foot", Json::makeString(p.foot));
    j.set("gap_sec", Json::makeNumber(p.gapSec));
    j.set("overlapped", Json::makeBool(p.overlapped));
    j.set("pitch_ratio", Json::makeNumber(p.pitchRatio));
    j.set("total_gain", Json::makeNumber(p.totalGain));
    j.set("duck_gain", Json::makeNumber(p.duckGain));
    j.set("lowpass_alpha", Json::makeNumber(p.lowpassAlpha));
    j.set("pan", Json::makeNumber(p.pan));
    j.set("rng_seed", Json::makeNumber(static_cast<double>(p.rngSeed)));
    return j;
}

}  // namespace

RenderResult Renderer::renderClip(const std::string& clipId,
                                  std::vector<FootEvent> events,
                                  const std::string& checkpointPath, bool resume) {
    clipId_ = clipId;
    events_ = sortEvents(std::move(events));
    voices_.clear();
    selections_.clear();
    nextFrame_ = 0;
    nextEvent_ = 0;
    peakVoices_ = 0;
    voiceLimitExceeded_ = false;

    // Validate everything before touching any output file.
    for (const FootEvent& event : events_) {
        if (!cfg_.findSurface(event.surface)) {
            throw std::runtime_error("event references unknown surface '" +
                                     event.surface + "' in clip " + clipId);
        }
        if (!cfg_.findLayer(event.velocity)) {
            throw std::runtime_error("event velocity matches no layer in clip " + clipId);
        }
    }

    const std::string rawPath = outDir_ + "/" + clipId + ".f32le";
    const std::string ndjsonPath = outDir_ + "/" + clipId + ".selections.ndjson";
    int resumes = 0;
    bool resumed = false;

    if (resume && std::filesystem::exists(checkpointPath)) {
        std::ifstream cp(checkpointPath, std::ios::binary);
        char magic[8];
        cp.read(magic, 8);
        if (!cp || std::memcmp(magic, kCheckpointMagic, 8) != 0) {
            throw std::runtime_error("corrupt checkpoint: " + checkpointPath);
        }
        if (readU32(cp) != 1) throw std::runtime_error("unsupported checkpoint version");
        nextFrame_ = static_cast<size_t>(readU64(cp));
        nextEvent_ = static_cast<int>(readU32(cp));
        peakVoices_ = static_cast<int>(readU32(cp));
        resumes = static_cast<int>(readU32(cp)) + 1;
        size_t storedSelections = static_cast<size_t>(readU64(cp));
        size_t block = static_cast<size_t>(cfg_.blockFrames);
        bool blockAligned = nextFrame_ % block == 0;
        if (!std::filesystem::exists(rawPath) || !std::filesystem::exists(ndjsonPath)) {
            throw std::runtime_error("checkpoint sidecars missing for " + clipId);
        }
        size_t rawFrames =
            std::filesystem::file_size(rawPath) / (sizeof(float) * cfg_.channels());
        if (!blockAligned && rawFrames != nextFrame_) {
            throw std::runtime_error("checkpoint is not block aligned");
        }
        if (rawFrames != nextFrame_) {
            throw std::runtime_error("raw sidecar does not match checkpoint frame count");
        }
        // Replay prior selections deterministically so RNG stream and voices
        // continue identically to a one-shot render.
        ClipSpec clip;
        clip.clipId = clipId;
        for (int i = 0; i < nextEvent_; ++i) {
            const FootEvent* previous = i > 0 ? &events_[i - 1] : nullptr;
            selections_.push_back(select(clip, events_[i], previous, i));
        }
        if (selections_.size() != storedSelections) {
            throw std::runtime_error("checkpoint selection count mismatch");
        }
        resumed = true;
    } else {
        std::filesystem::remove(rawPath);
        std::filesystem::remove(ndjsonPath);
        std::filesystem::remove(checkpointPath);
    }

    size_t totalFrames = 0;
    if (!events_.empty()) {
        size_t lastFrame =
            static_cast<size_t>(std::llround(events_.back().timeSec * cfg_.sampleRate));
        double longest = 0.0;
        ClipSpec clip;
        clip.clipId = clipId;
        for (int i = 0; i < static_cast<int>(events_.size()); ++i) {
            const FootEvent* previous = i > 0 ? &events_[i - 1] : nullptr;
            SelectionParams p = select(clip, events_[i], previous, i);
            longest = std::max(longest,
                               library_.get(p.sample).audio.frames() / p.pitchRatio);
        }
        totalFrames = lastFrame + static_cast<size_t>(longest) +
                      static_cast<size_t>(cfg_.tailPaddingSec * cfg_.sampleRate);
    }

    std::ofstream ndjson(ndjsonPath, resumed ? std::ios::app : std::ios::trunc);
    if (!ndjson) throw std::runtime_error("cannot write selections: " + ndjsonPath);

    size_t blockSize = static_cast<size_t>(cfg_.blockFrames);
    size_t flushedSelections = resumed ? selections_.size() : 0;
    for (size_t start = nextFrame_; start < totalFrames; start += blockSize) {
        size_t count = std::min(blockSize, totalFrames - start);
        std::vector<float> block = renderBlock(start, count);
        appendRaw(rawPath, block.data(), block.size());
        while (flushedSelections < selections_.size()) {
            const SelectionParams& p = selections_.at(flushedSelections);
            ndjson << selectionToJson(p).dump(0) << "\n";
            ++flushedSelections;
        }
        ndjson.flush();
        nextFrame_ = start + count;
        writeCheckpoint(checkpointPath, nextFrame_, nextEvent_, peakVoices_, resumes);
    }
    ndjson.close();

    // Materialize the final WAV (float sidecar keeps mixing bit-exact across
    // one-shot and resumed runs; WAV is quantized once at the end).
    WavData wav;
    wav.sampleRate = cfg_.sampleRate;
    wav.channels = cfg_.channels();
    wav.interleaved.resize(totalFrames * static_cast<size_t>(cfg_.channels()));
    std::ifstream raw(rawPath, std::ios::binary);
    if (!raw) throw std::runtime_error("missing raw sidecar: " + rawPath);
    raw.read(reinterpret_cast<char*>(wav.interleaved.data()),
             static_cast<std::streamsize>(wav.interleaved.size() * sizeof(float)));
    if (static_cast<size_t>(raw.gcount()) != wav.interleaved.size() * sizeof(float)) {
        throw std::runtime_error("raw sidecar size mismatch for " + clipId);
    }
    uint32_t mask = cfg_.channels() == 1 ? 0u : 0x3u;  // mono | FL+FR
    std::string wavPath = outDir_ + "/" + clipId + ".wav";
    writeWavPcm16(wavPath, wav, mask);

    std::string manifestPath = outDir_ + "/" + clipId + ".manifest.json";
    std::ofstream mf(manifestPath, std::ios::trunc);
    mf << buildManifest(clipId, totalFrames).dump(2);

    RenderResult result;
    result.wavPath = wavPath;
    result.manifestPath = manifestPath;
    result.eventCount = static_cast<int>(events_.size());
    result.peakVoices = peakVoices_;
    result.resumed = resumed;
    result.checkpointResumes = resumes;
    return result;
}

Json Renderer::buildManifest(const std::string& clipId, size_t totalFrames) {
    Json root = Json::makeObject();
    root.set("clip_id", Json::makeString(clipId));
    root.set("sample_rate", Json::makeNumber(cfg_.sampleRate));
    root.set("total_frames", Json::makeNumber(static_cast<double>(totalFrames)));
    root.set("duration_sec",
             Json::makeNumber(static_cast<double>(totalFrames) / cfg_.sampleRate));
    Json layout = Json::makeObject();
    if (cfg_.channels() == 1) {
        layout.set("channels", Json::makeNumber(1));
        layout.set("labels", Json::makeArray());
        layout.at("labels").array().push_back(Json::makeString("mono"));
        layout.set("wav_channel_mask", Json::makeNumber(0));
    } else {
        layout.set("channels", Json::makeNumber(2));
        Json labels = Json::makeArray();
        labels.push(Json::makeString("FL"));
        labels.push(Json::makeString("FR"));
        layout.set("labels", labels);
        layout.set("wav_channel_mask", Json::makeNumber(0x3));
    }
    root.set("channel_layout", layout);

    Json records = Json::makeArray();
    for (size_t i = 0; i < selections_.size(); ++i) {
        Json record = Json::makeObject();
        record.set("event_index", Json::makeNumber(events_[i].index));
        record.set("time_sec", Json::makeNumber(events_[i].timeSec));
        record.set("velocity", Json::makeNumber(events_[i].velocity));
        record.set("surface", Json::makeString(events_[i].surface));
        record.set("selection", selectionToJson(selections_[i]));
        records.push(record);
    }
    root.set("events", records);
    root.set("peak_voices", Json::makeNumber(peakVoices_));
    Json warnings = Json::makeArray();
    if (voiceLimitExceeded_) {
        warnings.push(Json::makeString("peak voice count exceeded max_voices=" +
                                       std::to_string(cfg_.maxVoices) +
                                       "; no voices were evicted"));
    }
    root.set("warnings", warnings);
    return root;
}

}  // namespace footbatch

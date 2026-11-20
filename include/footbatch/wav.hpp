#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace footbatch {

struct WavData {
    int sampleRate = 0;
    int channels = 0;
    std::vector<float> interleaved;  // planar index: frame * channels + ch

    size_t frames() const {
        return channels == 0 ? 0 : interleaved.size() / static_cast<size_t>(channels);
    }
    float sample(size_t frame, int ch) const {
        return interleaved[frame * static_cast<size_t>(channels) + ch];
    }
};

class WavError : public std::runtime_error {
public:
    explicit WavError(const std::string& msg) : std::runtime_error(msg) {}
};

WavData readWav(const std::string& path);

// Writes PCM-16 WAVEFORMATEXTENSIBLE with an explicit speaker channel mask.
// mask 0 => mono, 0x3 => stereo (front left + front right).
void writeWavPcm16(const std::string& path, const WavData& data, uint32_t channelMask);

}  // namespace footbatch

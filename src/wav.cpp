#include "footbatch/wav.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace footbatch {
namespace {

uint16_t rd16(const unsigned char* p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t rd32(const unsigned char* p) {
    return static_cast<uint32_t>(p[0] | (p[1] << 8) | (p[2] << 16) |
                                  (static_cast<uint32_t>(p[3]) << 24));
}

void wr16(std::ofstream& f, uint16_t v) {
    char b[2] = {static_cast<char>(v & 0xFF), static_cast<char>(v >> 8)};
    f.write(b, 2);
}

void wr32(std::ofstream& f, uint32_t v) {
    char b[4] = {static_cast<char>(v & 0xFF), static_cast<char>((v >> 8) & 0xFF),
                 static_cast<char>((v >> 16) & 0xFF), static_cast<char>(v >> 24)};
    f.write(b, 4);
}

void wrTag(std::ofstream& f, const char tag[4]) {
    f.write(tag, 4);
}

std::vector<unsigned char> readAll(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw WavError("cannot open wav: " + path);
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(f)),
                                      std::istreambuf_iterator<char>());
}

// KSDATAFORMAT_SUBTYPE_PCM, byte for byte.
const unsigned char kSubtypePcm[16] = {
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
    0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71};

}  // namespace

WavData readWav(const std::string& path) {
    std::vector<unsigned char> b = readAll(path);
    if (b.size() < 12 || std::memcmp(b.data(), "RIFF", 4) != 0 ||
        std::memcmp(b.data() + 8, "WAVE", 4) != 0) {
        throw WavError("not a RIFF/WAVE file: " + path);
    }

    int formatTag = 0;
    int channels = 0;
    int sampleRate = 0;
    int bits = 0;
    const unsigned char* data = nullptr;
    uint32_t dataSize = 0;

    size_t pos = 12;
    while (pos + 8 <= b.size()) {
        const unsigned char* chunk = b.data() + pos;
        uint32_t size = rd32(chunk + 4);
        size_t body = pos + 8;
        if (body + size > b.size()) throw WavError("truncated wav chunk: " + path);
        if (std::memcmp(chunk, "fmt ", 4) == 0) {
            if (size < 16) throw WavError("bad fmt chunk: " + path);
            formatTag = rd16(b.data() + body);
            channels = rd16(b.data() + body + 2);
            sampleRate = static_cast<int>(rd32(b.data() + body + 4));
            bits = rd16(b.data() + body + 14);
            if (formatTag == 0xFFFE) {
                if (size < 40 || std::memcmp(b.data() + body + 24, kSubtypePcm, 16) != 0) {
                    throw WavError("only PCM subformat is supported: " + path);
                }
                formatTag = 1;  // WAVEFORMATEXTENSIBLE PCM behaves as PCM-16
            } else if (formatTag != 1 && formatTag != 3) {
                throw WavError("unsupported wav format tag: " + path);
            }
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            data = b.data() + body;
            dataSize = size;
        }
        pos = body + size + (size & 1u);
    }
    if (!data || channels <= 0 || sampleRate <= 0) {
        throw WavError("wav missing fmt/data: " + path);
    }

    WavData out;
    out.sampleRate = sampleRate;
    out.channels = channels;

    if (formatTag == 1 && bits == 16) {
        size_t frames = dataSize / (2 * static_cast<uint32_t>(channels));
        out.interleaved.resize(frames * static_cast<size_t>(channels));
        for (size_t i = 0; i < out.interleaved.size(); ++i) {
            int16_t v = static_cast<int16_t>(rd16(data + i * 2));
            out.interleaved[i] = static_cast<float>(v) / 32768.0f;
        }
    } else if (formatTag == 3 && bits == 32) {
        size_t frames = dataSize / (4 * static_cast<uint32_t>(channels));
        out.interleaved.resize(frames * static_cast<size_t>(channels));
        for (size_t i = 0; i < out.interleaved.size(); ++i) {
            uint32_t bits32 = rd32(data + i * 4);
            float v;
            std::memcpy(&v, &bits32, 4);
            out.interleaved[i] = v;
        }
    } else {
        throw WavError("only PCM16 and float32 wav are supported: " + path);
    }
    return out;
}

void writeWavPcm16(const std::string& path, const WavData& data, uint32_t channelMask) {
    if (data.channels <= 0 || data.sampleRate <= 0) {
        throw WavError("cannot write empty wav: " + path);
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw WavError("cannot write wav: " + path);

    const int channels = data.channels;
    const uint32_t dataBytes =
        static_cast<uint32_t>(data.interleaved.size() * sizeof(int16_t));
    const uint32_t fmtBytes = 40;  // WAVEFORMATEXTENSIBLE
    const uint32_t riffSize = 4 + (8 + fmtBytes) + (8 + dataBytes);

    wrTag(f, "RIFF");
    wr32(f, riffSize);
    wrTag(f, "WAVE");

    wrTag(f, "fmt ");
    wr32(f, fmtBytes);
    wr16(f, 0xFFFE);               // WAVE_FORMAT_EXTENSIBLE
    wr16(f, static_cast<uint16_t>(channels));
    wr32(f, static_cast<uint32_t>(data.sampleRate));
    wr32(f, static_cast<uint32_t>(data.sampleRate) * channels * 2);  // byte rate
    wr16(f, static_cast<uint16_t>(channels * 2));                    // block align
    wr16(f, 16);                                                     // bits per sample
    wr16(f, 22);                                                     // cbSize
    wr16(f, 16);                                                     // valid bits
    wr32(f, channelMask);
    f.write(reinterpret_cast<const char*>(kSubtypePcm), 16);

    wrTag(f, "data");
    wr32(f, dataBytes);
    for (float v : data.interleaved) {
        v = std::clamp(v, -1.0f, 1.0f);
        int32_t scaled = static_cast<int32_t>(std::lround(v * 32767.0f));
        wr16(f, static_cast<int16_t>(std::clamp(scaled, -32768, 32767)));
    }
    if (!f) throw WavError("failed while writing wav: " + path);
}

}  // namespace footbatch

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct WavData {
    int sampleRate = 0;
    int channels = 0;
    std::vector<float> samples; // interleaved, normalized to [-1, 1]

    int64_t frames() const { return channels > 0 ? (int64_t)samples.size() / channels : 0; }
};

static uint16_t rdU16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rdU32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

inline WavData readWav(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t)size);
    if (fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        fclose(f);
        throw std::runtime_error("cannot read " + path);
    }
    fclose(f);

    if (buf.size() < 12 || memcmp(buf.data(), "RIFF", 4) != 0 ||
        memcmp(buf.data() + 8, "WAVE", 4) != 0)
        throw std::runtime_error("not a RIFF/WAVE file: " + path);

    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    const uint8_t* data = nullptr;
    uint32_t dataSize = 0;
    size_t off = 12;
    while (off + 8 <= buf.size()) {
        const uint8_t* ch = buf.data() + off;
        uint32_t csz = rdU32(ch + 4);
        if (memcmp(ch, "fmt ", 4) == 0 && csz >= 16) {
            format = rdU16(ch + 8);
            channels = rdU16(ch + 10);
            rate = rdU32(ch + 12);
            bits = rdU16(ch + 22);
        } else if (memcmp(ch, "data", 4) == 0) {
            data = ch + 8;
            dataSize = csz;
        }
        off += 8 + csz + (csz & 1);
    }
    if (!data || channels == 0 || rate == 0)
        throw std::runtime_error("missing fmt/data chunk in " + path);
    if (format != 1 && format != 3)
        throw std::runtime_error("unsupported WAV format (need PCM int or IEEE float)");

    WavData w;
    w.sampleRate = (int)rate;
    w.channels = (int)channels;
    uint32_t bytesPerSample = bits / 8;
    uint32_t frameSize = bytesPerSample * channels;
    if (frameSize == 0) throw std::runtime_error("bad bit depth");
    uint32_t frames = dataSize / frameSize;
    w.samples.resize((size_t)frames * channels);

    for (uint32_t i = 0; i < frames * channels; ++i) {
        const uint8_t* p = data + (size_t)i * bytesPerSample;
        float v = 0.0f;
        if (format == 3 && bits == 32) {
            float fv;
            memcpy(&fv, p, 4);
            v = fv;
        } else if (format == 1 && bits == 16) {
            v = (float)((int16_t)rdU16(p)) / 32768.0f;
        } else if (format == 1 && bits == 24) {
            int32_t s = (int32_t)(p[0] | (p[1] << 8) | (p[2] << 16));
            if (s & 0x800000) s |= ~0xFFFFFF;
            v = (float)s / 8388608.0f;
        } else if (format == 1 && bits == 32) {
            v = (float)(int32_t)rdU32(p) / 2147483648.0f;
        } else {
            throw std::runtime_error("unsupported bit depth/format combo");
        }
        w.samples[i] = v;
    }
    return w;
}

inline void writeWavFloat(const std::string& path, const std::vector<float>& samples,
                          int channels, int sampleRate) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    uint32_t dataSize = (uint32_t)(samples.size() * 4);
    uint32_t byteRate = (uint32_t)sampleRate * channels * 4;
    uint16_t blockAlign = (uint16_t)(channels * 4);

    uint32_t riffSize = 36 + dataSize;
    fwrite("RIFF", 1, 4, f);
    fwrite(&riffSize, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    uint32_t fmtSize = 16;
    uint16_t fmt = 3, ch = (uint16_t)channels, bits = 32;
    fwrite("fmt ", 1, 4, f);
    fwrite(&fmtSize, 4, 1, f);
    fwrite(&fmt, 2, 1, f);
    fwrite(&ch, 2, 1, f);
    fwrite(&sampleRate, 4, 1, f);
    fwrite(&byteRate, 4, 1, f);
    fwrite(&blockAlign, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&dataSize, 4, 1, f);
    fwrite(samples.data(), 4, samples.size(), f);
    fclose(f);
}

inline std::vector<float> extractChannel(const WavData& w, int ch) {
    std::vector<float> out((size_t)w.frames());
    for (int64_t i = 0; i < w.frames(); ++i)
        out[(size_t)i] = w.samples[(size_t)i * w.channels + ch];
    return out;
}

inline void writeWavPcm24(const std::string& path, const std::vector<float>& samples,
                          int channels, int sampleRate) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    uint32_t dataSize = (uint32_t)(samples.size() * 3);
    uint32_t byteRate = (uint32_t)sampleRate * channels * 3;
    uint16_t blockAlign = (uint16_t)(channels * 3);
    uint32_t riffSize = 36 + dataSize;
    uint32_t fmtSize = 16;
    uint16_t fmt = 1, ch = (uint16_t)channels, bits = 24;
    fwrite("RIFF", 1, 4, f);
    fwrite(&riffSize, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    fwrite(&fmtSize, 4, 1, f);
    fwrite(&fmt, 2, 1, f);
    fwrite(&ch, 2, 1, f);
    fwrite(&sampleRate, 4, 1, f);
    fwrite(&byteRate, 4, 1, f);
    fwrite(&blockAlign, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&dataSize, 4, 1, f);
    std::vector<uint8_t> buf(samples.size() * 3);
    for (size_t i = 0; i < samples.size(); ++i) {
        float v = samples[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        int32_t s = (int32_t)lrintf(v * 8388608.0f);
        if (s > 8388607) s = 8388607;
        if (s < -8388608) s = -8388608;
        buf[3 * i] = (uint8_t)(s & 0xFF);
        buf[3 * i + 1] = (uint8_t)((s >> 8) & 0xFF);
        buf[3 * i + 2] = (uint8_t)((s >> 16) & 0xFF);
    }
    fwrite(buf.data(), 1, buf.size(), f);
    fclose(f);
}

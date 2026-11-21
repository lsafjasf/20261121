#include "wav_io.hpp"

#include <cstring>
#include <fstream>
#include <numeric>

namespace aec {
namespace {

uint16_t read_u16(const unsigned char* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

uint32_t read_u32(const unsigned char* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

void put_u16(std::vector<unsigned char>& out, uint16_t v) {
    out.push_back(static_cast<unsigned char>(v & 0xff));
    out.push_back(static_cast<unsigned char>((v >> 8) & 0xff));
}

void put_u32(std::vector<unsigned char>& out, uint32_t v) {
    out.push_back(static_cast<unsigned char>(v & 0xff));
    out.push_back(static_cast<unsigned char>((v >> 8) & 0xff));
    out.push_back(static_cast<unsigned char>((v >> 16) & 0xff));
    out.push_back(static_cast<unsigned char>((v >> 24) & 0xff));
}

bool write_all(const std::filesystem::path& path, const std::vector<unsigned char>& bytes,
               std::string& error) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        error = "cannot open output WAV: " + path.string();
        return false;
    }
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        error = "failed while writing WAV: " + path.string();
        return false;
    }
    return true;
}

std::vector<float> decode_pcm16(const unsigned char* data, std::size_t bytes) {
    const std::size_t count = bytes / 2;
    std::vector<float> samples(count);
    for (std::size_t i = 0; i < count; ++i) {
        int16_t value = static_cast<int16_t>(read_u16(data + 2 * i));
        samples[i] = static_cast<float>(value) / 32768.0f;
    }
    return samples;
}

std::vector<unsigned char> encode_pcm16(const std::vector<float>& samples) {
    std::vector<unsigned char> bytes;
    bytes.reserve(samples.size() * 2);
    for (float sample : samples) {
        const double clamped = std::clamp(static_cast<double>(sample), -1.0, 1.0);
        const int value = static_cast<int>(std::lrint(clamped * 32767.0));
        put_u16(bytes, static_cast<uint16_t>(static_cast<int16_t>(value)));
    }
    return bytes;
}

std::vector<unsigned char> make_wav_header(int sample_rate, int channels,
                                           uint32_t data_bytes) {
    std::vector<unsigned char> header;
    header.reserve(44);
    const auto tag = [&header](const char* text) {
        header.insert(header.end(), text, text + 4);
    };
    const uint16_t block_align = static_cast<uint16_t>(channels * 2);
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate * block_align);
    tag("RIFF");
    put_u32(header, 36 + data_bytes);
    tag("WAVE");
    tag("fmt ");
    put_u32(header, 16);
    put_u16(header, 1);
    put_u16(header, static_cast<uint16_t>(channels));
    put_u32(header, static_cast<uint32_t>(sample_rate));
    put_u32(header, byte_rate);
    put_u16(header, block_align);
    put_u16(header, 16);
    tag("data");
    put_u32(header, data_bytes);
    return header;
}

bool write_pcm16(const std::filesystem::path& path, int sample_rate, int channels,
                 const std::vector<float>& interleaved, std::string& error) {
    if (sample_rate <= 0 || (channels != 1 && channels != 2)) {
        error = "invalid WAV output format";
        return false;
    }
    if (interleaved.size() % static_cast<std::size_t>(channels) != 0) {
        error = "interleaved sample count is not divisible by channel count";
        return false;
    }
    const auto payload = encode_pcm16(interleaved);
    auto bytes = make_wav_header(sample_rate, channels, static_cast<uint32_t>(payload.size()));
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    return write_all(path, bytes, error);
}

}  // namespace

bool read_wav(const std::filesystem::path& path, WavAudio& audio, std::string& error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        error = "cannot open input WAV: " + path.string();
        return false;
    }
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(stream)),
                                     std::istreambuf_iterator<char>());
    if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
        std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        error = "not a RIFF/WAVE file: " + path.string();
        return false;
    }

    uint16_t format = 0;
    uint16_t channels = 0;
    uint16_t bits_per_sample = 0;
    uint32_t sample_rate = 0;
    const unsigned char* pcm = nullptr;
    std::size_t pcm_bytes = 0;

    std::size_t offset = 12;
    while (offset + 8 <= bytes.size()) {
        const unsigned char* chunk = bytes.data() + offset;
        const uint32_t size = read_u32(chunk + 4);
        const std::size_t body = offset + 8;
        if (body + size > bytes.size()) {
            error = "truncated WAV chunk: " + path.string();
            return false;
        }
        if (std::memcmp(chunk, "fmt ", 4) == 0 && size >= 16) {
            format = read_u16(bytes.data() + body);
            channels = read_u16(bytes.data() + body + 2);
            sample_rate = read_u32(bytes.data() + body + 4);
            bits_per_sample = read_u16(bytes.data() + body + 14);
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            pcm = bytes.data() + body;
            pcm_bytes = size;
        }
        offset = body + size + (size & 1U);
    }

    if (format != 1 || bits_per_sample != 16 || (channels != 1 && channels != 2) ||
        sample_rate == 0 || pcm == nullptr) {
        error = "only 16-bit PCM mono/stereo WAV is supported: " + path.string();
        return false;
    }

    auto decoded = decode_pcm16(pcm, pcm_bytes);
    const std::size_t frame_count =
        decoded.size() - decoded.size() % static_cast<std::size_t>(channels);
    decoded.resize(frame_count);

    audio.sample_rate = static_cast<int>(sample_rate);
    audio.channels = static_cast<int>(channels);
    audio.interleaved = std::move(decoded);
    return true;
}

bool read_mono_wav(const std::filesystem::path& path, MonoAudio& audio, std::string& error) {
    WavAudio wav;
    if (!read_wav(path, wav, error)) {
        return false;
    }
    audio.sample_rate = wav.sample_rate;
    if (wav.channels == 1) {
        audio.samples = std::move(wav.interleaved);
        return true;
    }
    audio.samples.resize(wav.frames());
    for (std::size_t i = 0; i < audio.samples.size(); ++i) {
        audio.samples[i] = 0.5f * (wav.interleaved[2 * i] + wav.interleaved[2 * i + 1]);
    }
    return true;
}

bool write_mono_wav(const std::filesystem::path& path, const MonoAudio& audio,
                    std::string& error) {
    return write_pcm16(path, audio.sample_rate, 1, audio.samples, error);
}

bool write_stereo_wav(const std::filesystem::path& path, const StereoAudio& audio,
                      std::string& error) {
    if (audio.left.size() != audio.right.size()) {
        error = "stereo channels must have equal length";
        return false;
    }
    std::vector<float> interleaved(audio.left.size() * 2);
    for (std::size_t i = 0; i < audio.left.size(); ++i) {
        interleaved[2 * i] = audio.left[i];
        interleaved[2 * i + 1] = audio.right[i];
    }
    return write_pcm16(path, audio.sample_rate, 2, interleaved, error);
}

}  // namespace aec

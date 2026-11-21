#pragma once

#include "common.hpp"

#include <filesystem>
#include <string>

namespace aec {

bool read_wav(const std::filesystem::path& path, WavAudio& audio, std::string& error);

bool read_mono_wav(const std::filesystem::path& path, MonoAudio& audio, std::string& error);

bool write_mono_wav(const std::filesystem::path& path, const MonoAudio& audio,
                    std::string& error);

bool write_stereo_wav(const std::filesystem::path& path, const StereoAudio& audio,
                      std::string& error);

}  // namespace aec

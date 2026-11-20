#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "footbatch/json.hpp"
#include "footbatch/library.hpp"
#include "footbatch/renderer.hpp"
#include "footbatch/strategy.hpp"
#include "footbatch/wav.hpp"

namespace {

using namespace footbatch;

struct Args {
    std::map<std::string, std::string> values;
    std::vector<std::string> positional;
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) == 0) {
            std::string key = a.substr(2);
            std::string value;
            auto eq = key.find('=');
            if (eq != std::string::npos) {
                value = key.substr(eq + 1);
                key = key.substr(0, eq);
            } else if (i + 1 < argc) {
                value = argv[++i];
            }
            args.values[key] = value;
        } else {
            args.positional.push_back(a);
        }
    }
    return args;
}

std::string need(const Args& args, const std::string& key) {
    auto it = args.values.find(key);
    if (it == args.values.end()) throw std::runtime_error("missing --" + key);
    return it->second;
}

int runGenLib(const Args& args) {
    StrategyConfig cfg = loadStrategy(need(args, "strategy"));
    generateLibrary(cfg);
    std::cout << "generated library at " << cfg.libraryRoot << "\n";
    return 0;
}

int runRender(const Args& args) {
    StrategyConfig cfg = loadStrategy(need(args, "strategy"));
    SampleLibrary library;
    library.load(cfg);
    std::string outDir = need(args, "out");
    bool resume = args.values.count("resume") != 0;

    std::vector<ClipSpec> clips;
    if (args.values.count("batch")) {
        clips = loadBatch(need(args, "batch")).clips;
    } else {
        ClipSpec clip;
        clip.clipId = args.values.count("clip-id") ? need(args, "clip-id") : "clip";
        clip.events = loadEvents(need(args, "events"));
        clips.push_back(std::move(clip));
    }

    Renderer renderer(cfg, library, outDir);
    for (const ClipSpec& clip : clips) {
        std::string cp = outDir + "/" + clip.clipId + ".checkpoint";
        RenderResult r = renderer.renderClip(clip.clipId, clip.events, cp, resume);
        std::cout << clip.clipId << ": " << r.eventCount << " events, peak "
                  << r.peakVoices << " voices -> " << r.wavPath
                  << (r.resumed ? " (resumed)\n" : "\n");
    }
    return 0;
}

int runVerify(const Args& args) {
    WavData a = readWav(need(args, "a"));
    WavData b = readWav(need(args, "b"));
    if (a.sampleRate != b.sampleRate || a.channels != b.channels ||
        a.interleaved.size() != b.interleaved.size()) {
        std::cerr << "verify: wav shape mismatch\n";
        return 2;
    }
    int diffs = 0;
    for (size_t i = 0; i < a.interleaved.size(); ++i) {
        if (a.interleaved[i] != b.interleaved[i]) ++diffs;
    }
    if (diffs != 0) {
        std::cerr << "verify: " << diffs << " differing samples\n";
        return 2;
    }
    std::cout << "verify: bit-identical (" << a.interleaved.size() << " samples)\n";
    return 0;
}

void usage() {
    std::cout
        << "footbatch - deterministic batch footstep renderer\n\n"
        << "subcommands:\n"
        << "  genlib  --strategy FILE\n"
        << "  render  --strategy FILE --out DIR (--events FILE [--clip-id ID] | --batch FILE)\n"
        << "          [--resume]\n"
        << "  verify  --a WAV --b WAV\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    std::string cmd = argv[1];
    Args args = parseArgs(argc - 2, argv + 2);
    try {
        if (cmd == "genlib") return runGenLib(args);
        if (cmd == "render") return runRender(args);
        if (cmd == "verify") return runVerify(args);
        usage();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "footbatch: " << e.what() << "\n";
        return 1;
    }
}

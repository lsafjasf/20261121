// check_master: verify the merged master against the ground truth.
//   - no discontinuities anywhere, especially around reported switch points
//   - per-segment fidelity (ramp / repeated clip touches / quiet tail)
//   - quiet-tail noise floor stays at the high-gain track's level and does
//     not rise toward the (amplified) low-gain track's noise
//
// Usage: check_master <master.wav> <truth.wav> <stereo.wav> <report.txt>

#include "dsp.h"
#include "wav.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Report {
    double gain = 1.0;
    double delay = 0.0;
    int lowChannel = 0;
    std::vector<std::vector<int64_t>> regions; // fadeInStart, coreStart, coreEnd, fadeOutEnd
};

Report readReport(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) throw std::runtime_error("cannot open report " + path);
    Report rep;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (!strncmp(line, "gain=", 5)) rep.gain = atof(line + 5);
        else if (!strncmp(line, "delay=", 6)) rep.delay = atof(line + 6);
        else if (!strncmp(line, "low_channel=", 12)) rep.lowChannel = atoi(line + 12);
        else if (!strncmp(line, "region=", 7)) {
            std::vector<int64_t> r(4);
            sscanf(line + 7, "%lld,%lld,%lld,%lld", (long long*)&r[0],
                   (long long*)&r[1], (long long*)&r[2], (long long*)&r[3]);
            rep.regions.push_back(r);
        }
    }
    fclose(f);
    return rep;
}

double maxSlope(const std::vector<float>& x, int64_t n0, int64_t n1) {
    double m = 0.0;
    if (n0 < 1) n0 = 1;
    if (n1 > (int64_t)x.size()) n1 = (int64_t)x.size();
    for (int64_t i = n0; i < n1; ++i)
        m = std::max(m, (double)std::fabs(x[i] - x[i - 1]));
    return m;
}

double rmsErr(const std::vector<float>& a, const std::vector<float>& b, int64_t n0,
              int64_t n1) {
    double s = 0.0;
    for (int64_t i = n0; i < n1; ++i) {
        double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return std::sqrt(s / (double)(n1 - n0));
}

double rms(const std::vector<float>& a, int64_t n0, int64_t n1) {
    double s = 0.0;
    for (int64_t i = n0; i < n1; ++i) s += (double)a[i] * a[i];
    return std::sqrt(s / (double)(n1 - n0));
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: check_master <master.wav> <truth.wav> <stereo.wav> "
                "<report.txt>\n");
        return 2;
    }
    WavData master = readWav(argv[1]);
    WavData truth = readWav(argv[2]);
    WavData stereo = readWav(argv[3]);
    Report rep = readReport(argv[4]);

    const int fs = master.sampleRate;
    const int64_t n = std::min(master.frames(), truth.frames());
    int fails = 0;
    auto check = [&](bool ok, const char* name, const char* detail) {
        printf("[%s] %-34s %s\n", ok ? "PASS" : "FAIL", name, detail);
        if (!ok) ++fails;
    };
    char buf[160];

    // 1. Global continuity: master slope must be comparable to truth slope.
    double sm = maxSlope(master.samples, 1, n);
    double st = maxSlope(truth.samples, 1, n);
    snprintf(buf, sizeof(buf), "max|dx| master=%.5f truth=%.5f ratio=%.2f", sm, st,
             sm / st);
    check(sm < 2.0 * st, "global continuity", buf);

    // 2. Switch-point continuity: +-10 ms around every reported region.
    double worst = 0.0;
    for (auto& r : rep.regions) {
        int64_t a = std::max<int64_t>(1, r[0] - (int64_t)(0.010 * fs));
        int64_t b = std::min<int64_t>(n, r[3] + (int64_t)(0.010 * fs));
        double rm = maxSlope(master.samples, a, b);
        double rt = maxSlope(truth.samples, a, b);
        worst = std::max(worst, rm / std::max(rt, 1e-9));
    }
    snprintf(buf, sizeof(buf), "worst local slope ratio=%.2f over %zu regions",
             worst, rep.regions.size());
    check(worst < 2.0, "switch-point continuity", buf);

    // 3. Per-segment fidelity vs truth (ramp / clip touches / quiet tail).
    const char* segNames[3] = {"ramp", "clip-touches", "quiet-tail"};
    for (int s = 0; s < 3; ++s) {
        int64_t a = (int64_t)(s * 3.0 * fs) + (int64_t)(0.05 * fs);
        int64_t b = std::min<int64_t>(n, (int64_t)((s + 1) * 3.0 * fs));
        double e = rmsErr(master.samples, truth.samples, a, b);
        double ref = rms(truth.samples, a, b);
        char name[64];
        snprintf(name, sizeof(name), "segment fidelity (%s)", segNames[s]);
        snprintf(buf, sizeof(buf), "rms_err=%.6f seg_rms=%.6f rel=%.4f", e, ref,
                 e / ref);
        check(e < 0.05 * ref, name, buf);
    }

    // 4. Quiet-tail noise floor (last 1.5 s): master must track the high-gain
    //    track's noise, not the amplified low-gain noise.
    int64_t a = (int64_t)(7.5 * fs), b = n;
    std::vector<float> high(n), lowScaled(n);
    int lc = rep.lowChannel;
    std::vector<float> low = extractChannel(stereo, lc);
    for (int64_t i = 0; i < n; ++i) {
        high[i] = stereo.samples[2 * i + (1 - lc)];
        lowScaled[i] = (float)(rep.gain *
                               sincInterp(low, (double)i + rep.delay));
    }
    double nMaster = rmsErr(master.samples, truth.samples, a, b);
    double nHigh = rmsErr(high, truth.samples, a, b);
    double nLow = rmsErr(lowScaled, truth.samples, a, b);
    snprintf(buf, sizeof(buf), "noise master=%.2e high=%.2e low*gain=%.2e", nMaster,
             nHigh, nLow);
    check(nMaster < 3.0 * nHigh && nMaster < nLow / 3.0, "quiet-tail noise floor",
          buf);

    printf("%s\n", fails == 0 ? "ALL CHECKS PASSED" : "CHECKS FAILED");
    return fails == 0 ? 0 : 1;
}

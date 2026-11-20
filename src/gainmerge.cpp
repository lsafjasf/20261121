// gainmerge: merge a dual-gain stereo recording (one mic, low-gain + high-gain
// track) into a continuous unclipped floating-point master.
//
// Usage: gainmerge <in-stereo.wav> <out-master.wav> [options]
//   --low <0|1>     force low-gain channel index (default: auto by RMS)
//   --report <path> write estimation/switch report (default: <out>.report.txt)

#include "dsp.h"
#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Estimate {
    double delay = 0.0; // samples; high[n] ~= gain * low_interp(n + delay)
    double gain = 1.0;  // linear, high = gain * low
};

// Pick a window of clean, loud samples on the high track for estimation.
void pickWindow(const std::vector<float>& h, double clipRef, int64_t& n0, int64_t& n1) {
    const int64_t n = (int64_t)h.size();
    const int64_t win = std::min<int64_t>(n, 32768);
    double bestScore = -1.0;
    int64_t bestStart = 0;
    const int64_t step = 4096;
    for (int64_t s = 0; s + win <= n; s += step) {
        double sum = 0.0;
        bool clipped = false;
        for (int64_t i = s; i < s + win; ++i) {
            double a = std::fabs(h[i]);
            if (a > 0.85 * clipRef) {
                clipped = true;
                break;
            }
            sum += a;
        }
        if (!clipped && sum > bestScore) {
            bestScore = sum;
            bestStart = s;
        }
    }
    n0 = bestStart;
    n1 = bestStart + win;
}

Estimate estimateDelayAndGain(const std::vector<float>& h, const std::vector<float>& l,
                              double clipRef) {
    Estimate est;
    int64_t n0, n1;
    pickWindow(h, clipRef, n0, n1);

    // Integer lag via cross-correlation: R[k] = sum h[n] * l[n+k], peak near delay.
    const int KMAX = 64;
    std::vector<double> R(2 * KMAX + 1);
    for (int k = -KMAX; k <= KMAX; ++k) R[k + KMAX] = corrAtLag(h, l, n0, n1, k);
    int kbest = -KMAX;
    for (int k = -KMAX + 1; k <= KMAX - 1; ++k)
        if (R[k + KMAX] > R[kbest + KMAX]) kbest = k;

    // Parabolic refinement around the integer peak.
    double ym = R[kbest - 1 + KMAX], y0 = R[kbest + KMAX], yp = R[kbest + 1 + KMAX];
    double denom = ym - 2.0 * y0 + yp;
    double frac = (std::fabs(denom) > 1e-30) ? 0.5 * (ym - yp) / denom : 0.0;
    frac = std::max(-1.0, std::min(1.0, frac));
    double d0 = kbest + frac;

    // Fine search with sinc-interpolated low track, normalized by the
    // interpolated energy to avoid bias from interpolator amplitude ripple,
    // then parabolic refinement of the grid peak.
    const double step = 0.005;
    std::vector<double> ds, cs;
    for (double d = d0 - 0.15; d <= d0 + 0.1501; d += step) {
        double num = 0.0, den = 0.0;
        for (int64_t i = n0; i < n1; ++i) {
            double la = sincInterp(l, (double)i + d);
            num += (double)h[i] * la;
            den += la * la;
        }
        ds.push_back(d);
        cs.push_back(num / std::sqrt(std::max(den, 1e-30)));
    }
    size_t bi = 0;
    for (size_t i = 1; i < cs.size(); ++i)
        if (cs[i] > cs[bi]) bi = i;
    double refined = ds[bi];
    if (bi > 0 && bi + 1 < cs.size()) {
        double ym = cs[bi - 1], y0 = cs[bi], yp = cs[bi + 1];
        double dn = ym - 2.0 * y0 + yp;
        if (std::fabs(dn) > 1e-30) {
            double off = 0.5 * (ym - yp) / dn;
            if (std::fabs(off) <= 1.0) refined = ds[bi] + off * step;
        }
    }
    est.delay = refined;

    // Least-squares gain on clean samples: h ~= g * l_aligned.
    double num = 0.0, den = 0.0;
    const int64_t n = (int64_t)h.size();
    for (int64_t i = 0; i < n; ++i) {
        double a = std::fabs(h[i]);
        if (a < 0.02 || a > 0.6 * clipRef) continue;
        double la = sincInterp(l, (double)i + est.delay);
        num += (double)h[i] * la;
        den += la * la;
    }
    if (den > 0.0) est.gain = num / den;
    return est;
}

} // namespace

struct Region {
    int64_t fadeInStart;  // weight ramps 0 -> 1 over [fadeInStart, coreStart]
    int64_t coreStart;
    int64_t coreEnd;      // weight == 1 over [coreStart, coreEnd]
    int64_t fadeOutEnd;   // weight ramps 1 -> 0 over [coreEnd, fadeOutEnd]
};

// Find clipped samples, expand by a safety margin, merge nearby regions,
// then attach raised-cosine fades fully outside the expanded cores.
std::vector<Region> findSwitchRegions(const std::vector<float>& h, double clipThresh,
                                      int sampleRate) {
    const int64_t n = (int64_t)h.size();
    const int64_t margin = (int64_t)(0.004 * sampleRate);  // 4 ms each side
    const int64_t fade = (int64_t)(0.003 * sampleRate);    // 3 ms raised cosine
    const int64_t mergeGap = (int64_t)(0.040 * sampleRate); // merge gaps < 40 ms

    std::vector<std::pair<int64_t, int64_t>> cores;
    int64_t i = 0;
    while (i < n) {
        if (std::fabs(h[i]) >= clipThresh) {
            int64_t s = i;
            while (i < n && std::fabs(h[i]) >= clipThresh) ++i;
            cores.push_back({s, i - 1});
        } else {
            ++i;
        }
    }

    // Expand cores by the safety margin, then merge overlapping/nearby ones.
    std::vector<std::pair<int64_t, int64_t>> merged;
    for (auto& c : cores) {
        int64_t s = std::max<int64_t>(0, c.first - margin);
        int64_t e = std::min<int64_t>(n - 1, c.second + margin);
        if (!merged.empty() && s - merged.back().second < mergeGap) {
            merged.back().second = std::max(merged.back().second, e);
        } else {
            merged.push_back({s, e});
        }
    }

    std::vector<Region> regions;
    for (auto& mc : merged) {
        Region r;
        r.coreStart = mc.first;
        r.coreEnd = mc.second;
        r.fadeInStart = std::max<int64_t>(0, mc.first - fade);
        r.fadeOutEnd = std::min<int64_t>(n - 1, mc.second + fade);
        regions.push_back(r);
    }
    // Fix overlaps introduced by fades of merged neighbors.
    for (size_t k = 1; k < regions.size(); ++k) {
        if (regions[k].fadeInStart < regions[k - 1].fadeOutEnd)
            regions[k].fadeInStart = regions[k - 1].fadeOutEnd;
    }
    return regions;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: gainmerge <in-stereo.wav> <out-master.wav> "
                "[--low 0|1] [--report path]\n");
        return 2;
    }
    std::string inPath = argv[1], outPath = argv[2];
    int lowCh = -1;
    std::string reportPath;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--low") && i + 1 < argc) {
            lowCh = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--report") && i + 1 < argc) {
            reportPath = argv[++i];
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        }
    }
    if (reportPath.empty()) reportPath = outPath + ".report.txt";

    WavData w = readWav(inPath);
    if (w.channels != 2) {
        fprintf(stderr, "error: input must be stereo (low-gain + high-gain)\n");
        return 1;
    }
    const int64_t n = w.frames();
    std::vector<float> ch[2];
    ch[0].resize(n);
    ch[1].resize(n);
    for (int64_t i = 0; i < n; ++i) {
        ch[0][i] = w.samples[2 * i];
        ch[1][i] = w.samples[2 * i + 1];
    }

    if (lowCh < 0) {
        double rms[2] = {0, 0};
        for (int c = 0; c < 2; ++c)
            for (int64_t i = 0; i < n; ++i) rms[c] += (double)ch[c][i] * ch[c][i];
        lowCh = (rms[0] <= rms[1]) ? 0 : 1;
    }
    const std::vector<float>& low = ch[lowCh];
    const std::vector<float>& high = ch[1 - lowCh];

    double maxAbs = 0.0;
    for (int64_t i = 0; i < n; ++i) maxAbs = std::max(maxAbs, (double)std::fabs(high[i]));
    const double clipRef = 1.0;
    if (maxAbs < 0.90) {
        fprintf(stderr,
                "warning: high-gain track peaks at %.3f; no clipping likely\n", maxAbs);
    }
    const double clipThresh = 0.995;

    Estimate est = estimateDelayAndGain(high, low, clipRef);
    printf("low-gain channel : %d\n", lowCh);
    printf("gain difference  : %.6f linear (%.2f dB)\n", est.gain,
           20.0 * log10(est.gain));
    printf("sub-sample delay : %.4f samples (high = gain * low[n + delay])\n",
           est.delay);

    std::vector<Region> regions = findSwitchRegions(high, clipThresh, w.sampleRate);
    printf("switch regions   : %zu\n", regions.size());

    // Build the master: weight 0 = high track, 1 = aligned+scaled low track.
    std::vector<float> out(n);
    size_t r = 0;
    for (int64_t i = 0; i < n; ++i) {
        while (r < regions.size() && i > regions[r].fadeOutEnd) ++r;
        double wgt = 0.0;
        if (r < regions.size() && i >= regions[r].fadeInStart) {
            const Region& rg = regions[r];
            if (i < rg.coreStart) {
                double t = (double)(i - rg.fadeInStart) / (double)(rg.coreStart - rg.fadeInStart);
                wgt = 0.5 - 0.5 * cos(M_PI * t);
            } else if (i <= rg.coreEnd) {
                wgt = 1.0;
            } else {
                double t = (double)(i - rg.coreEnd) / (double)(rg.fadeOutEnd - rg.coreEnd);
                wgt = 0.5 + 0.5 * cos(M_PI * t);
            }
        }
        double la = (wgt > 0.0) ? est.gain * sincInterp(low, (double)i + est.delay) : 0.0;
        out[i] = (float)((1.0 - wgt) * (double)high[i] + wgt * la);
    }

    writeWavFloat(outPath, out, 1, w.sampleRate);

    FILE* rep = fopen(reportPath.c_str(), "w");
    if (rep) {
        fprintf(rep, "sample_rate=%d\n", w.sampleRate);
        fprintf(rep, "frames=%lld\n", (long long)n);
        fprintf(rep, "low_channel=%d\n", lowCh);
        fprintf(rep, "gain=%.8f\n", est.gain);
        fprintf(rep, "delay=%.6f\n", est.delay);
        fprintf(rep, "clip_thresh=%.6f\n", clipThresh);
        fprintf(rep, "regions=%zu\n", regions.size());
        for (const Region& rg : regions)
            fprintf(rep, "region=%lld,%lld,%lld,%lld\n", (long long)rg.fadeInStart,
                    (long long)rg.coreStart, (long long)rg.coreEnd,
                    (long long)rg.fadeOutEnd);
        fclose(rep);
    }
    printf("master written   : %s (float32, mono)\n", outPath.c_str());
    printf("report written   : %s\n", reportPath.c_str());
    return 0;
}


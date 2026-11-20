// make_test: synthesize a dual-gain stereo test WAV plus the unclipped
// ground-truth master, with three segments:
//   0-3 s  amplitude ramp that ends up clipping the high-gain track
//   3-6 s  level hovering around full scale, repeatedly touching the clip
//   6-9 s  quiet decaying tail near the noise floor
//
// Usage: make_test <out-stereo.wav> <out-truth.wav> [--pcm24] [--swap]

#include "wav.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <random>
#include <vector>

namespace {

const int FS = 48000;
const double DUR = 9.0;
const double GAIN = 4.0;        // high = GAIN * low (12 dB)
const double DELAY = 0.3125;    // sub-sample delay of high track, in samples
const double CLIP = 0.999;      // converter clip level of high track
const double NOISE_RMS = 8e-5;  // per-track independent noise

// Ground-truth signal in high-gain scale. Continuous phase, per-segment
// amplitude/frequency plan.
double trueSignal(double t) {
    double amp, f1, f2;
    if (t < 3.0) {
        amp = 0.05 + (1.7 - 0.05) * (t / 3.0); // ramp into clipping
        f1 = 440.0;
        f2 = 1320.0;
    } else if (t < 6.0) {
        amp = 1.02 + 0.18 * sin(2.0 * M_PI * 1.5 * (t - 3.0)); // touches clip
        f1 = 330.0;
        f2 = 990.0;
    } else {
        amp = 0.02 * exp(-1.5 * (t - 6.0)); // quiet tail
        f1 = 440.0;
        f2 = 1320.0;
    }
    return amp * (0.7 * sin(2.0 * M_PI * f1 * t) + 0.3 * sin(2.0 * M_PI * f2 * t));
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: make_test <out-stereo.wav> <out-truth.wav> [--pcm24] "
                "[--swap]\n");
        return 2;
    }
    bool pcm24 = false, swap = false;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--pcm24")) pcm24 = true;
        else if (!strcmp(argv[i], "--swap")) swap = true;
    }
    const int64_t n = (int64_t)(DUR * FS);
    std::mt19937_64 rng(12345);
    std::normal_distribution<double> gauss(0.0, NOISE_RMS);

    std::vector<float> stereo(2 * n), truth(n);
    for (int64_t i = 0; i < n; ++i) {
        double t = (double)i / FS;
        double x = trueSignal(t);               // low track sees x(n)
        double xh = trueSignal(t + DELAY / FS); // high track sees x(n + delay)
        double low = x / GAIN + gauss(rng);
        double high = xh + gauss(rng);
        if (high > CLIP) high = CLIP;
        if (high < -CLIP) high = -CLIP;
        stereo[2 * i + (swap ? 1 : 0)] = (float)low;
        stereo[2 * i + (swap ? 0 : 1)] = (float)high;
        truth[i] = (float)xh; // ideal master: unclipped signal on high timeline
    }

    if (pcm24)
        writeWavPcm24(argv[1], stereo, 2, FS);
    else
        writeWavFloat(argv[1], stereo, 2, FS);
    writeWavFloat(argv[2], truth, 1, FS);
    printf("wrote %s (stereo: L=low gain, R=high gain, gain=%.1f, delay=%.4f "
           "samples)\n",
           argv[1], GAIN, DELAY);
    printf("wrote %s (unclipped ground truth, high-gain scale)\n", argv[2]);
    return 0;
}

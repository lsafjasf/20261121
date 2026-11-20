#pragma once

#include <cmath>
#include <vector>

// Windowed-sinc interpolation of x at fractional position pos.
// Half-length 16 taps, Hann window. Returns 0 outside valid range.
inline double sincInterp(const std::vector<float>& x, double pos) {
    const int H = 16;
    int64_t n = (int64_t)x.size();
    int64_t i0 = (int64_t)std::floor(pos);
    double frac = pos - (double)i0;
    double sum = 0.0;
    for (int k = -H + 1; k <= H; ++k) {
        int64_t idx = i0 + k;
        if (idx < 0 || idx >= n) continue;
        double t = (double)k - frac; // distance from sample to pos
        double s;
        if (std::fabs(t) < 1e-9) {
            s = 1.0;
        } else {
            s = std::sin(M_PI * t) / (M_PI * t);
        }
        // Hann window over [-H, H]
        double w = 0.5 + 0.5 * std::cos(M_PI * t / H);
        sum += (double)x[idx] * s * w;
    }
    return sum;
}

// Normalized cross-correlation of a and b at integer lag k:
// sum_n a[n] * b_interp? No -- plain integer-lag correlation over [n0, n1).
inline double corrAtLag(const std::vector<float>& a, const std::vector<float>& b,
                        int64_t n0, int64_t n1, int lag) {
    double sum = 0.0;
    int64_t n = (int64_t)a.size();
    for (int64_t i = n0; i < n1; ++i) {
        int64_t j = i + lag;
        if (j < 0 || j >= n) continue;
        sum += (double)a[i] * (double)b[j];
    }
    return sum;
}

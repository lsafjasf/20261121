#pragma once

#include "common.hpp"

#include <vector>

namespace aec {

struct DelayStep {
    double time_seconds = 0.0;
    int delay_ms = 0;
};

struct DemoConfig {
    int sample_rate = 16000;
    double duration_seconds = 26.0;
    double double_talk_start = 10.0;
    double double_talk_end = 14.0;
    double near_lead_seconds = 0.0;
    double near_tail_seconds = 0.0;
    double echo_gain_db = -7.0;
    double noise_dbfs = -62.0;
    std::vector<DelayStep> delay_steps = {
        {0.0, 48}, {5.0, 112}, {16.0, 72}, {21.0, 144}};
    unsigned int seed = 20261121U;
};

struct DemoData {
    int sample_rate = 0;
    std::vector<float> far;
    std::vector<float> near;
    std::vector<float> echo;
    std::vector<float> mic;
    std::vector<int> true_delay_samples;
    std::vector<DelayStep> delay_steps;
};

DemoData synthesize_demo(const DemoConfig& config);

}  // namespace aec

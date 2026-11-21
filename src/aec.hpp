#pragma once

#include "common.hpp"
#include "demo.hpp"

#include <optional>
#include <vector>

namespace aec {

struct AecConfig {
    int sample_rate = 16000;
    int filter_taps = 640;
    int block_size = 160;
    int delay_min_ms = 0;
    int delay_max_ms = 240;
    double delay_step_size = 0.50;
    double delay_estimate_interval = 0.10;
    double delay_window = 0.60;
    double dtd_enter_correlation = 0.56;
    double dtd_exit_correlation = 0.80;
    int dtd_enter_blocks = 2;
    int dtd_exit_blocks = 40;
    double dtd_release_hold = 0.45;
    double recovery_warmup = 0.35;
    double near_active_correlation_window = 0.020;
};

struct AecFrameMetric {
    double time_seconds = 0.0;
    double far_dbfs = kFloorDb;
    double near_dbfs = kFloorDb;
    double echo_dbfs = kFloorDb;
    double mic_dbfs = kFloorDb;
    double output_dbfs = kFloorDb;
    double residual_echo_dbfs = kFloorDb;
    double erle_db = std::numeric_limits<double>::quiet_NaN();
    double near_attenuation_db = std::numeric_limits<double>::quiet_NaN();
    double delay_ms = 0.0;
    int double_talk = 0;
    double double_talk_fraction = 0.0;
};

struct DetectedDelayStep {
    double time_seconds = 0.0;
    int delay_ms = 0;
    int delay_samples = 0;
    double correlation = 0.0;
};

struct GroundTruth {
    const std::vector<float>* near_clean = nullptr;
    const std::vector<float>* true_echo = nullptr;
};

struct AecResult {
    std::vector<float> output;
    std::vector<AecFrameMetric> metrics;
    std::vector<DetectedDelayStep> delay_steps;
    std::vector<uint8_t> double_talk_samples;
    double double_talk_start_detected = std::numeric_limits<double>::quiet_NaN();
    double double_talk_end_detected = std::numeric_limits<double>::quiet_NaN();
};

AecResult run_aec(const std::vector<float>& far, const std::vector<float>& mic,
                  const AecConfig& config, const GroundTruth& truth = {});

}  // namespace aec

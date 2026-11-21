#pragma once

#include "aec.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace aec {

struct WindowStats {
    std::string name;
    double start_seconds = 0.0;
    double end_seconds = 0.0;
    double erle_db = std::numeric_limits<double>::quiet_NaN();
    double residual_echo_dbfs = kFloorDb;
    double near_attenuation_db = std::numeric_limits<double>::quiet_NaN();
    double double_talk_fraction = 0.0;
    double delay_ms = 0.0;
    int frame_count = 0;
};

struct ReconvergenceEvent {
    std::string label;
    double event_seconds = 0.0;
    bool found = false;
    double healthy_start_seconds = std::numeric_limits<double>::quiet_NaN();
    double confirmed_seconds = std::numeric_limits<double>::quiet_NaN();
    double erle_before_db = std::numeric_limits<double>::quiet_NaN();
    double erle_after_db = std::numeric_limits<double>::quiet_NaN();
    double residual_after_dbfs = kFloorDb;
    double delay_ms = 0.0;
};

struct AnalysisSummary {
    double duration_seconds = 0.0;
    double output_peak_dbfs = kFloorDb;
    double detector_release_seconds = std::numeric_limits<double>::quiet_NaN();
    std::vector<WindowStats> windows;
    std::vector<ReconvergenceEvent> reconvergence;
};

AnalysisSummary analyze_result(const AecResult& result,
                               const std::vector<DelayStep>& true_delay_steps,
                               double dt_start, double dt_end, double duration_seconds);

bool write_artifact_bundle(const std::filesystem::path& out_dir, int sample_rate,
                           const std::vector<float>& far,
                           const std::vector<float>& near,
                           const std::vector<float>& echo,
                           const std::vector<float>& mic,
                           const AecResult& result,
                           const std::vector<DelayStep>& true_delay_steps,
                           double dt_start, double dt_end,
                           const AecConfig& aec_config,
                           AnalysisSummary& summary, std::string& error);

void print_analysis(const AnalysisSummary& summary, const AecResult& result);

}  // namespace aec

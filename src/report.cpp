#include "report.hpp"

#include "wav_io.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace aec {
namespace {

double median(std::vector<double> values) {
    values.erase(std::remove_if(values.begin(), values.end(),
                                [](double value) { return !std::isfinite(value); }),
                 values.end());
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle),
                     values.end());
    if (values.size() % 2 == 1) {
        return values[middle];
    }
    const double upper = values[middle];
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(middle - 1),
                     values.end());
    return 0.5 * (upper + values[middle - 1]);
}

double mean_power_from_db(const std::vector<AecFrameMetric>& frames,
                          double AecFrameMetric::*member) {
    double sum = 0.0;
    int count = 0;
    for (const auto& frame : frames) {
        const double db = frame.*member;
        if (std::isfinite(db) && db > -110.0) {
            sum += std::pow(10.0, db / 10.0);
            ++count;
        }
    }
    return count > 0 ? db_from_power(sum / count) : kFloorDb;
}

std::vector<const AecFrameMetric*> frames_in_window(
    const std::vector<AecFrameMetric>& metrics, double start, double end) {
    std::vector<const AecFrameMetric*> result;
    for (const auto& metric : metrics) {
        const double center = metric.time_seconds + 0.025;
        if (center >= start && center < end) {
            result.push_back(&metric);
        }
    }
    return result;
}

WindowStats summarize_window(const std::vector<AecFrameMetric>& metrics,
                             const std::string& name, double start, double end) {
    WindowStats stats;
    stats.name = name;
    stats.start_seconds = start;
    stats.end_seconds = end;
    const auto frames = frames_in_window(metrics, start, end);
    stats.frame_count = static_cast<int>(frames.size());
    if (frames.empty()) {
        return stats;
    }
    std::vector<double> erle;
    std::vector<double> near_att;
    std::vector<double> delay;
    std::vector<AecFrameMetric> dereferenced;
    double dt_sum = 0.0;
    for (const auto* frame : frames) {
        if (std::isfinite(frame->erle_db)) {
            erle.push_back(frame->erle_db);
        }
        if (std::isfinite(frame->near_attenuation_db)) {
            near_att.push_back(frame->near_attenuation_db);
        }
        if (frame->far_dbfs > -55.0) {
            delay.push_back(frame->delay_ms);
        }
        dt_sum += frame->double_talk_fraction;
        dereferenced.push_back(*frame);
    }
    stats.erle_db = median(erle);
    stats.near_attenuation_db = median(near_att);
    stats.residual_echo_dbfs =
        mean_power_from_db(dereferenced, &AecFrameMetric::residual_echo_dbfs);
    stats.double_talk_fraction = dt_sum / frames.size();
    stats.delay_ms = median(delay);
    return stats;
}

ReconvergenceEvent find_reconvergence(const std::vector<AecFrameMetric>& metrics,
                                      const std::string& label, double event_time,
                                      double search_start, double search_end) {
    ReconvergenceEvent event;
    event.label = label;
    event.event_seconds = event_time;
    const auto before =
        frames_in_window(metrics, std::max(0.0, event_time - 1.6),
                         std::max(0.0, event_time - 0.05));
    std::vector<double> before_erle;
    for (const auto* frame : before) {
        if (std::isfinite(frame->erle_db)) {
            before_erle.push_back(frame->erle_db);
        }
    }
    event.erle_before_db = median(before_erle);
    const double baseline =
        std::isfinite(event.erle_before_db) ? event.erle_before_db : 8.0;
    const double threshold = std::max(6.0, std::min(12.0, baseline - 5.0));
    const auto candidates =
        frames_in_window(metrics, search_start, std::min(search_end, 1.0e9));
    constexpr std::size_t kWindowFrames = 10;
    for (std::size_t start = 0; start + kWindowFrames <= candidates.size(); ++start) {
        int healthy_count = 0;
        int echo_active_count = 0;
        for (std::size_t i = start; i < start + kWindowFrames; ++i) {
            const auto* frame = candidates[i];
            const bool healthy =
                frame->double_talk_fraction < 0.25 &&
                frame->residual_echo_dbfs < -38.0 &&
                ((std::isfinite(frame->erle_db) && frame->erle_db >= threshold) ||
                 frame->echo_dbfs < -55.0);
            healthy_count += healthy ? 1 : 0;
            echo_active_count += frame->echo_dbfs > -45.0 ? 1 : 0;
        }
        if (healthy_count >= 8 && echo_active_count >= 2) {
            std::size_t first_healthy = start;
            while (first_healthy < start + kWindowFrames &&
                   !(candidates[first_healthy]->double_talk_fraction < 0.25 &&
                     candidates[first_healthy]->residual_echo_dbfs < -38.0 &&
                     ((std::isfinite(candidates[first_healthy]->erle_db) &&
                       candidates[first_healthy]->erle_db >= threshold) ||
                      candidates[first_healthy]->echo_dbfs < -55.0))) {
                ++first_healthy;
            }
            event.found = true;
            event.healthy_start_seconds = candidates[first_healthy]->time_seconds;
            event.confirmed_seconds = event.healthy_start_seconds + 0.50;
            std::vector<AecFrameMetric> settled;
            std::vector<double> settled_erle;
            for (std::size_t i = first_healthy; i < start + kWindowFrames; ++i) {
                settled.push_back(*candidates[i]);
                if (std::isfinite(candidates[i]->erle_db) &&
                    candidates[i]->echo_dbfs > -50.0) {
                    settled_erle.push_back(candidates[i]->erle_db);
                }
            }
            if (settled_erle.empty()) {
                const std::size_t extra_end = std::min(
                    candidates.size(), start + kWindowFrames + 30);
                for (std::size_t i = start + kWindowFrames; i < extra_end; ++i) {
                    if (std::isfinite(candidates[i]->erle_db) &&
                        candidates[i]->echo_dbfs > -50.0) {
                        settled_erle.push_back(candidates[i]->erle_db);
                    }
                }
            }
            event.erle_after_db = median(settled_erle);
            event.residual_after_dbfs =
                mean_power_from_db(settled, &AecFrameMetric::residual_echo_dbfs);
            event.delay_ms = settled.back().delay_ms;
            break;
        }
    }
    return event;
}

std::string json_number(double value, int precision = 6) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream stream;
    stream << std::setprecision(precision) << std::fixed << value;
    return stream.str();
}

std::string json_escape(const std::string& text) {
    std::string result;
    for (char ch : text) {
        if (ch == '\\' || ch == '"') {
            result.push_back('\\');
        }
        result.push_back(ch);
    }
    return result;
}

bool write_metrics_csv(const std::filesystem::path& path, const AecResult& result,
                       std::string& error) {
    std::ofstream stream(path);
    if (!stream) {
        error = "cannot write " + path.string();
        return false;
    }
    stream << "time_s,far_dbfs,near_dbfs,echo_dbfs,mic_dbfs,output_dbfs,"
              "residual_echo_dbfs,erle_db,near_attenuation_db,delay_ms,"
              "double_talk,double_talk_fraction\n";
    stream << std::fixed << std::setprecision(5);
    const auto number = [](double value) {
        if (!std::isfinite(value)) {
            return std::string("nan");
        }
        std::ostringstream text;
        text << std::fixed << std::setprecision(5) << value;
        return text.str();
    };
    for (const auto& metric : result.metrics) {
        stream << number(metric.time_seconds) << ',' << number(metric.far_dbfs) << ','
               << number(metric.near_dbfs) << ',' << number(metric.echo_dbfs) << ','
               << number(metric.mic_dbfs) << ',' << number(metric.output_dbfs) << ','
               << number(metric.residual_echo_dbfs) << ','
               << number(metric.erle_db) << ','
               << number(metric.near_attenuation_db) << ','
               << number(metric.delay_ms) << ',' << metric.double_talk << ','
               << number(metric.double_talk_fraction) << '\n';
    }
    return true;
}

bool write_summary_json(const std::filesystem::path& path, const AnalysisSummary& summary,
                        const AecResult& result, const AecConfig& config,
                        const std::vector<DelayStep>& true_steps, double dt_start,
                        double dt_end, bool ground_truth_metrics, std::string& error) {
    std::ofstream stream(path);
    if (!stream) {
        error = "cannot write " + path.string();
        return false;
    }
    stream << "{\n";
    stream << "  \"sample_rate\": " << config.sample_rate << ",\n";
    stream << "  \"duration_seconds\": " << json_number(summary.duration_seconds) << ",\n";
    stream << "  \"ground_truth_metrics\": "
           << (ground_truth_metrics ? "true" : "false") << ",\n";
    stream << "  \"output_peak_dbfs\": " << json_number(summary.output_peak_dbfs) << ",\n";
    stream << "  \"double_talk\": {\"start\": " << json_number(dt_start)
           << ", \"end\": " << json_number(dt_end)
           << ", \"detector_start\": "
           << json_number(result.double_talk_start_detected)
           << ", \"detector_release\": "
           << json_number(result.double_talk_end_detected) << "},\n";
    stream << "  \"aec\": {\"filter_taps\": " << config.filter_taps
           << ", \"delay_min_ms\": " << config.delay_min_ms
           << ", \"delay_max_ms\": " << config.delay_max_ms
           << ", \"step_size\": " << json_number(config.delay_step_size) << "},\n";
    stream << "  \"windows\": [\n";
    for (std::size_t i = 0; i < summary.windows.size(); ++i) {
        const auto& window = summary.windows[i];
        stream << "    {\"name\": \"" << json_escape(window.name)
               << "\", \"start\": " << json_number(window.start_seconds)
               << ", \"end\": " << json_number(window.end_seconds)
               << ", \"erle_db\": " << json_number(window.erle_db)
               << ", \"residual_echo_dbfs\": "
               << json_number(window.residual_echo_dbfs)
               << ", \"near_attenuation_db\": "
               << json_number(window.near_attenuation_db)
               << ", \"double_talk_fraction\": "
               << json_number(window.double_talk_fraction)
               << ", \"delay_ms\": " << json_number(window.delay_ms) << "}";
        stream << (i + 1 == summary.windows.size() ? "\n" : ",\n");
    }
    stream << "  ],\n";
    stream << "  \"reconvergence\": [\n";
    for (std::size_t i = 0; i < summary.reconvergence.size(); ++i) {
        const auto& event = summary.reconvergence[i];
        stream << "    {\"label\": \"" << json_escape(event.label)
               << "\", \"event_seconds\": " << json_number(event.event_seconds)
               << ", \"found\": " << (event.found ? "true" : "false")
               << ", \"healthy_start_seconds\": "
               << json_number(event.healthy_start_seconds)
               << ", \"confirmed_seconds\": "
               << json_number(event.confirmed_seconds)
               << ", \"erle_before_db\": " << json_number(event.erle_before_db)
               << ", \"erle_after_db\": " << json_number(event.erle_after_db)
               << ", \"residual_after_dbfs\": "
               << json_number(event.residual_after_dbfs)
               << ", \"delay_ms\": " << json_number(event.delay_ms) << "}";
        stream << (i + 1 == summary.reconvergence.size() ? "\n" : ",\n");
    }
    stream << "  ],\n";
    stream << "  \"true_delay_steps\": [";
    for (std::size_t i = 0; i < true_steps.size(); ++i) {
        stream << (i == 0 ? "" : ", ") << "{\"time\": "
               << json_number(true_steps[i].time_seconds)
               << ", \"delay_ms\": " << true_steps[i].delay_ms << "}";
    }
    stream << "],\n";
    stream << "  \"detected_delay_steps\": [";
    for (std::size_t i = 0; i < result.delay_steps.size(); ++i) {
        const auto& step = result.delay_steps[i];
        stream << (i == 0 ? "" : ", ") << "{\"time\": "
               << json_number(step.time_seconds) << ", \"delay_ms\": " << step.delay_ms
               << ", \"correlation\": " << json_number(step.correlation) << "}";
    }
    stream << "]\n";
    stream << "}\n";
    return true;
}

std::string xml_escape(const std::string& text) {
    std::string result;
    for (char ch : text) {
        switch (ch) {
            case '&':
                result += "&amp;";
                break;
            case '<':
                result += "&lt;";
                break;
            case '>':
                result += "&gt;";
                break;
            default:
                result.push_back(ch);
                break;
        }
    }
    return result;
}

void append_waveform(std::ostringstream& svg, const std::string& title,
                     const std::vector<float>& samples, int sample_rate, double duration,
                     int top, int height, const std::string& color) {
    constexpr double kLeft = 72.0;
    constexpr double kWidth = 1160.0;
    svg << "<text x='8' y='" << top + 16
        << "' font-size='13' font-weight='600' fill='#1f2937'>" << xml_escape(title)
        << "</text>\n";
    svg << "<rect x='" << kLeft << "' y='" << top << "' width='" << kWidth
        << "' height='" << height << "' fill='#fbfdff' stroke='#cbd5e1'/>\n";
    svg << "<line x1='" << kLeft << "' y1='" << top + height / 2.0 << "' x2='"
        << kLeft + kWidth << "' y2='" << top + height / 2.0
        << "' stroke='#e2e8f0'/>\n";
    svg << "<path d='";
    const int pixels = static_cast<int>(kWidth);
    const double center = top + height / 2.0;
    const double amplitude = height / 2.0 - 4.0;
    for (int p = 0; p < pixels; ++p) {
        const std::size_t begin = static_cast<std::size_t>(
            static_cast<double>(p) / pixels * samples.size());
        const std::size_t end = std::min(
            samples.size(), static_cast<std::size_t>(
                                static_cast<double>(p + 1) / pixels * samples.size()) +
                                1);
        float lo = 1.0f;
        float hi = -1.0f;
        for (std::size_t i = begin; i < end; ++i) {
            lo = std::min(lo, samples[i]);
            hi = std::max(hi, samples[i]);
        }
        if (begin >= end) {
            lo = hi = 0.0f;
        }
        const double x = kLeft + p + 0.5;
        const double y1 = center - std::clamp(static_cast<double>(hi), -1.0, 1.0) *
                                       amplitude;
        const double y2 = center - std::clamp(static_cast<double>(lo), -1.0, 1.0) *
                                       amplitude;
        svg << "M" << std::fixed << std::setprecision(2) << x << " "
            << std::setprecision(1) << y1 << "L" << x << " " << y2;
    }
    svg << "' stroke='" << color << "' stroke-width='0.8' fill='none'/>\n";
    svg << "<text x='" << kLeft - 62 << "' y='" << top + height - 6
        << "' font-size='10' fill='#64748b'>-1</text>\n";
    svg << "<text x='" << kLeft - 62 << "' y='" << top + 10
        << "' font-size='10' fill='#64748b'>+1</text>\n";
    (void)sample_rate;
    (void)duration;
}

bool write_timeline_svg(const std::filesystem::path& path,
                        const std::vector<float>& far, const std::vector<float>& near,
                        const std::vector<float>& mic,
                        const std::vector<float>& output,
                        const std::vector<AecFrameMetric>& metrics,
                        const std::vector<DelayStep>& true_steps,
                        const std::vector<ReconvergenceEvent>& reconvergence,
                        double duration, int sample_rate, std::string& error) {
    constexpr double kLeft = 72.0;
    constexpr double kWidth = 1160.0;
    const auto x_of = [&](double t) {
        return kLeft + std::clamp(t / std::max(duration, 1.0e-6), 0.0, 1.0) * kWidth;
    };
    std::ostringstream svg;
    svg << "<svg xmlns='http://www.w3.org/2000/svg' width='1280' height='820' "
           "viewBox='0 0 1280 820' font-family='DejaVu Sans, Arial, sans-serif'>\n";
    svg << "<rect width='1280' height='820' fill='white'/>\n";
    svg << "<text x='14' y='22' font-size='19' font-weight='700' fill='#0f172a'>"
           "Duplex voice-link AEC debug timeline</text>\n";
    svg << "<text x='72' y='40' font-size='11' fill='#475569'>far=blue, "
           "near=green, mic=orange, AEC output=purple; red band=double talk; "
           "green bands=reconvergence</text>\n";
    for (const auto& event : reconvergence) {
        if (!event.found) {
            continue;
        }
        const double x1 = x_of(event.healthy_start_seconds);
        const double x2 = x_of(event.confirmed_seconds);
        svg << "<rect x='" << x1 << "' y='448' width='" << std::max(2.0, x2 - x1)
            << "' height='332' fill='#86efac' opacity='0.25'/>\n";
    }
    const auto& dt_metrics = metrics;
    double dt_start = std::numeric_limits<double>::quiet_NaN();
    double dt_end = dt_start;
    for (const auto& metric : dt_metrics) {
        if (metric.double_talk_fraction > 0.2) {
            if (!std::isfinite(dt_start)) {
                dt_start = metric.time_seconds;
            }
            dt_end = metric.time_seconds + 0.05;
        }
    }
    if (std::isfinite(dt_start)) {
        svg << "<rect x='" << x_of(dt_start) << "' y='48' width='"
            << std::max(2.0, x_of(dt_end) - x_of(dt_start))
            << "' height='700' fill='#fecaca' opacity='0.45'/>\n";
        svg << "<text x='" << x_of(dt_start) + 5
            << "' y='62' font-size='10' fill='#991b1b'>double talk</text>\n";
    }
    for (const auto& step : true_steps) {
        svg << "<line x1='" << x_of(step.time_seconds) << "' y1='48' x2='"
            << x_of(step.time_seconds) << "' y2='752' stroke='#dc2626' "
               "stroke-width='1' stroke-dasharray='4 4' opacity='0.65'/>\n";
        svg << "<text x='" << x_of(step.time_seconds) + 3
            << "' y='750' font-size='9' fill='#991b1b'>" << step.delay_ms
            << " ms</text>\n";
    }
    append_waveform(svg, "far-end reference", far, sample_rate, duration, 58, 86,
                    "#2563eb");
    append_waveform(svg, "near-end clean (truth)", near, sample_rate, duration, 154, 86,
                    "#15803d");
    append_waveform(svg, "microphone = echo + near + noise", mic, sample_rate, duration,
                    250, 86, "#d97706");
    append_waveform(svg, "AEC output", output, sample_rate, duration, 346, 86, "#7c3aed");

    constexpr int kErleTop = 460;
    constexpr int kResidualTop = 570;
    constexpr int kDelayTop = 680;
    constexpr int kPlotHeight = 82;
    const auto plot_frame = [&](const std::string& title, int top,
                                const std::string& unit) {
        svg << "<text x='8' y='" << top + 16
            << "' font-size='13' font-weight='600' fill='#1f2937'>" << title
            << "</text>\n";
        svg << "<rect x='" << kLeft << "' y='" << top << "' width='" << kWidth
            << "' height='" << kPlotHeight
            << "' fill='#fbfdff' stroke='#cbd5e1'/>\n";
        svg << "<text x='" << kLeft - 58 << "' y='" << top + kPlotHeight / 2.0 + 4
            << "' font-size='10' fill='#64748b'>" << unit << "</text>\n";
    };
    plot_frame("ERLE (dB, truth echo / residual)", kErleTop, "dB");
    plot_frame("residual echo (dBFS)", kResidualTop, "dBFS");
    plot_frame("bulk delay (ms)", kDelayTop, "ms");
    for (int db = 0; db <= 30; db += 10) {
        const double y = kErleTop + kPlotHeight -
                         (db + 20.0) / 55.0 * kPlotHeight;
        svg << "<line x1='" << kLeft << "' y1='" << y << "' x2='" << kLeft + kWidth
            << "' y2='" << y << "' stroke='#e2e8f0'/>\n";
        svg << "<text x='" << kLeft + kWidth + 4 << "' y='" << y + 3
            << "' font-size='9' fill='#64748b'>" << db << "</text>\n";
    }
    for (int db = -80; db <= -20; db += 20) {
        const double y = kResidualTop + kPlotHeight -
                         (db + 100.0) / 80.0 * kPlotHeight;
        svg << "<line x1='" << kLeft << "' y1='" << y << "' x2='" << kLeft + kWidth
            << "' y2='" << y << "' stroke='#e2e8f0'/>\n";
        svg << "<text x='" << kLeft + kWidth + 4 << "' y='" << y + 3
            << "' font-size='9' fill='#64748b'>" << db << "</text>\n";
    }
    for (int delay = 0; delay <= 250; delay += 50) {
        const double y =
            kDelayTop + kPlotHeight - delay / 250.0 * kPlotHeight;
        svg << "<line x1='" << kLeft << "' y1='" << y << "' x2='" << kLeft + kWidth
            << "' y2='" << y << "' stroke='#e2e8f0'/>\n";
        svg << "<text x='" << kLeft + kWidth + 4 << "' y='" << y + 3
            << "' font-size='9' fill='#64748b'>" << delay << "</text>\n";
    }

    std::ostringstream erle_path;
    std::ostringstream residual_path;
    std::ostringstream delay_path;
    bool erle_started = false;
    bool residual_started = false;
    bool delay_started = false;
    double last_erle_time = -10.0;
    double last_residual_time = -10.0;
    double last_delay_time = -10.0;
    for (const auto& metric : metrics) {
        const double t = metric.time_seconds + 0.025;
        const double x = x_of(t);
        if (std::isfinite(metric.erle_db)) {
            const double y =
                kErleTop + kPlotHeight -
                std::clamp((metric.erle_db + 20.0) / 55.0, 0.0, 1.0) * kPlotHeight;
            erle_path << ((!erle_started || t - last_erle_time > 0.12) ? "M" : "L")
                      << x << " " << y << " ";
            last_erle_time = t;
            erle_started = true;
        }
        if (std::isfinite(metric.residual_echo_dbfs)) {
            const double y =
                kResidualTop + kPlotHeight -
                std::clamp((metric.residual_echo_dbfs + 100.0) / 80.0, 0.0, 1.0) *
                    kPlotHeight;
            residual_path
                << ((!residual_started || t - last_residual_time > 0.12) ? "M" : "L")
                << x << " " << y << " ";
            last_residual_time = t;
            residual_started = true;
        }
        const double delay_y =
            kDelayTop + kPlotHeight -
            std::clamp(metric.delay_ms / 250.0, 0.0, 1.0) * kPlotHeight;
        delay_path
            << ((!delay_started || t - last_delay_time > 0.12) ? "M" : "L")
            << x << " " << delay_y << " ";
        last_delay_time = t;
        delay_started = true;
    }
    svg << "<path d='" << erle_path.str()
        << "' fill='none' stroke='#0f766e' stroke-width='1.8'/>\n";
    svg << "<path d='" << residual_path.str()
        << "' fill='none' stroke='#be123c' stroke-width='1.6'/>\n";
    svg << "<path d='" << delay_path.str()
        << "' fill='none' stroke='#7c3aed' stroke-width='1.8'/>\n";
    for (int second = 0; second <= static_cast<int>(duration); second += 2) {
        const double x = x_of(second);
        svg << "<line x1='" << x << "' y1='" << kDelayTop + kPlotHeight << "' x2='" << x
            << "' y2='" << kDelayTop + kPlotHeight + 5
            << "' stroke='#94a3b8'/>\n";
        svg << "<text x='" << x << "' y='" << kDelayTop + kPlotHeight + 18
            << "' text-anchor='middle' font-size='10' fill='#475569'>" << second
            << " s</text>\n";
    }
    svg << "</svg>\n";
    std::ofstream stream(path);
    if (!stream) {
        error = "cannot write " + path.string();
        return false;
    }
    stream << svg.str();
    return true;
}

uint32_t crc32_bytes(const unsigned char* data, std::size_t size) {
    uint32_t crc = 0xffffffffU;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1U) ^ (0xedb88320U & (0U - (crc & 1U)));
        }
    }
    return crc ^ 0xffffffffU;
}

void append_u32_be(std::vector<unsigned char>& data, uint32_t value) {
    data.push_back(static_cast<unsigned char>((value >> 24U) & 0xffU));
    data.push_back(static_cast<unsigned char>((value >> 16U) & 0xffU));
    data.push_back(static_cast<unsigned char>((value >> 8U) & 0xffU));
    data.push_back(static_cast<unsigned char>(value & 0xffU));
}

uint32_t adler32_bytes(const std::vector<unsigned char>& data) {
    uint32_t a = 1;
    uint32_t b = 0;
    for (unsigned char value : data) {
        a = (a + value) % 65521U;
        b = (b + a) % 65521U;
    }
    return (b << 16U) | a;
}

void append_png_chunk(std::vector<unsigned char>& png, const char* type,
                      const std::vector<unsigned char>& payload) {
    append_u32_be(png, static_cast<uint32_t>(payload.size()));
    const std::size_t type_start = png.size();
    png.insert(png.end(), type, type + 4);
    png.insert(png.end(), payload.begin(), payload.end());
    const uint32_t crc = crc32_bytes(png.data() + type_start, 4 + payload.size());
    append_u32_be(png, crc);
}

bool write_png_rgb(const std::filesystem::path& path, int width, int height,
                   const std::vector<unsigned char>& rgb, std::string& error) {
    if (width <= 0 || height <= 0 ||
        rgb.size() != static_cast<std::size_t>(width * height * 3)) {
        error = "invalid PNG image buffer";
        return false;
    }
    std::vector<unsigned char> raw;
    raw.reserve(static_cast<std::size_t>(height) *
                (static_cast<std::size_t>(width) * 3 + 1));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        const auto begin = rgb.begin() + static_cast<std::ptrdiff_t>(y) * width * 3;
        raw.insert(raw.end(), begin, begin + width * 3);
    }

    std::vector<unsigned char> zlib = {0x78, 0x01};
    std::size_t offset = 0;
    while (offset < raw.size()) {
        const std::size_t block_size = std::min<std::size_t>(65535, raw.size() - offset);
        const bool final = offset + block_size == raw.size();
        zlib.push_back(final ? 0x01 : 0x00);
        const uint16_t length = static_cast<uint16_t>(block_size);
        const uint16_t inverse = static_cast<uint16_t>(~length);
        zlib.push_back(static_cast<unsigned char>(length & 0xff));
        zlib.push_back(static_cast<unsigned char>(length >> 8));
        zlib.push_back(static_cast<unsigned char>(inverse & 0xff));
        zlib.push_back(static_cast<unsigned char>(inverse >> 8));
        zlib.insert(zlib.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
                    raw.begin() + static_cast<std::ptrdiff_t>(offset + block_size));
        offset += block_size;
    }
    append_u32_be(zlib, adler32_bytes(raw));

    std::vector<unsigned char> png = {137, 80, 78, 71, 13, 10, 26, 10};
    std::vector<unsigned char> ihdr;
    append_u32_be(ihdr, static_cast<uint32_t>(width));
    append_u32_be(ihdr, static_cast<uint32_t>(height));
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});
    append_png_chunk(png, "IHDR", ihdr);
    append_png_chunk(png, "IDAT", zlib);
    append_png_chunk(png, "IEND", {});

    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        error = "cannot write " + path.string();
        return false;
    }
    stream.write(reinterpret_cast<const char*>(png.data()),
                 static_cast<std::streamsize>(png.size()));
    return static_cast<bool>(stream);
}

void fft_in_place(std::vector<std::complex<double>>& values) {
    const std::size_t count = values.size();
    for (std::size_t i = 1, j = 0; i < count; ++i) {
        std::size_t bit = count >> 1U;
        for (; j & bit; bit >>= 1U) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(values[i], values[j]);
        }
    }
    for (std::size_t length = 2; length <= count; length <<= 1U) {
        const double angle = -2.0 * kPi / static_cast<double>(length);
        const std::complex<double> rotation(std::cos(angle), std::sin(angle));
        for (std::size_t offset = 0; offset < count; offset += length) {
            std::complex<double> phase(1.0, 0.0);
            for (std::size_t j = 0; j < length / 2; ++j) {
                const std::complex<double> even = values[offset + j];
                const std::complex<double> odd =
                    values[offset + j + length / 2] * phase;
                values[offset + j] = even + odd;
                values[offset + j + length / 2] = even - odd;
                phase *= rotation;
            }
        }
    }
}

std::array<unsigned char, 3> spectrum_color(double db) {
    const double t = clamp01((db + 85.0) / 75.0);
    const std::array<std::array<double, 3>, 5> stops = {{
        {8, 12, 32}, {24, 72, 150}, {125, 36, 146}, {232, 89, 54}, {255, 224, 120}}};
    const double scaled = t * 4.0;
    const int index = std::min(3, static_cast<int>(scaled));
    const double local = scaled - index;
    std::array<unsigned char, 3> color{};
    for (int channel = 0; channel < 3; ++channel) {
        color[channel] = static_cast<unsigned char>(std::lrint(
            stops[index][channel] * (1.0 - local) +
            stops[index + 1][channel] * local));
    }
    return color;
}

bool write_spectrogram_png(const std::filesystem::path& path,
                           const std::vector<float>& samples, int sample_rate,
                           std::string& error) {
    constexpr int kWidth = 720;
    constexpr int kHeight = 360;
    constexpr int kFftSize = 512;
    constexpr int kHop = 160;
    constexpr int kTopMargin = 6;
    constexpr int kBottomMargin = 8;
    const int plot_height = kHeight - kTopMargin - kBottomMargin;
    std::vector<std::complex<double>> frame(kFftSize);
    std::vector<double> window(kFftSize);
    for (int i = 0; i < kFftSize; ++i) {
        window[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / (kFftSize - 1));
    }
    std::vector<std::vector<double>> magnitudes(
        static_cast<std::size_t>(kWidth), std::vector<double>(plot_height, -85.0));
    const std::size_t frame_count =
        samples.size() > kFftSize ? (samples.size() - kFftSize) / kHop + 1 : 0;
    const double duration =
        static_cast<double>(samples.size()) / std::max(1, sample_rate);
    for (std::size_t f = 0; f < frame_count; ++f) {
        const std::size_t begin = f * kHop;
        for (int i = 0; i < kFftSize; ++i) {
            frame[i] = samples[begin + static_cast<std::size_t>(i)] * window[i];
        }
        fft_in_place(frame);
        const double t = static_cast<double>(begin + kFftSize / 2) / sample_rate;
        const int x = std::clamp(static_cast<int>(t / duration * kWidth), 0, kWidth - 1);
        for (int y = 0; y < plot_height; ++y) {
            const double fraction =
                static_cast<double>(plot_height - 1 - y) / (plot_height - 1);
            const int bin = static_cast<int>(fraction * (kFftSize / 2 - 1));
            const double magnitude = std::abs(frame[static_cast<std::size_t>(bin)]);
            const double db = 20.0 * std::log10(std::max(1.0e-7, magnitude / 16.0));
            magnitudes[x][y] = std::max(magnitudes[x][y], db);
        }
    }
    std::vector<unsigned char> pixels(static_cast<std::size_t>(kWidth * kHeight * 3),
                                      255);
    for (int y = 0; y < plot_height; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            auto color = spectrum_color(magnitudes[x][y]);
            if (x % 72 == 0 || y == plot_height / 2 || y == plot_height / 4 ||
                y == 3 * plot_height / 4) {
                for (auto& channel : color) {
                    channel = static_cast<unsigned char>(0.72 * channel + 0.28 * 210);
                }
            }
            const std::size_t index =
                (static_cast<std::size_t>(y + kTopMargin) * kWidth + x) * 3;
            pixels[index] = color[0];
            pixels[index + 1] = color[1];
            pixels[index + 2] = color[2];
        }
    }
    return write_png_rgb(path, kWidth, kHeight, pixels, error);
}

std::string fixed3(double value) {
    if (!std::isfinite(value)) {
        return "&mdash;";
    }
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

bool write_report_html(const std::filesystem::path& path, const AnalysisSummary& summary,
                       const AecResult& result, const AecConfig& config,
                       double dt_start, double dt_end, bool has_near_truth,
                       bool has_echo_truth, std::string& error) {
    std::ofstream stream(path);
    if (!stream) {
        error = "cannot write " + path.string();
        return false;
    }
    stream << "<!doctype html><meta charset='utf-8'><title>Duplex AEC debug report</title>"
              "<style>body{font-family:system-ui,sans-serif;max-width:1280px;margin:24px "
              "auto;padding:0 18px;color:#172033;background:#f8fafc}h1,h2{color:#0f172a}"
              "table{border-collapse:collapse;width:100%;margin:12px 0 24px;background:white}"
              "th,td{border:1px solid #cbd5e1;padding:7px 9px;text-align:right}"
              "th:first-child,td:first-child{text-align:left}th{background:#e2e8f0}"
              ".card{background:white;border:1px solid #dbe4ef;border-radius:10px;padding:"
              "14px 18px;margin:14px 0}.ok{color:#047857;font-weight:600}.warn{color:#b45309}"
              "audio{width:100%;margin:5px 0}code{background:#e8eef6;padding:1px 4px;"
              "border-radius:3px}</style><body>";
    stream << "<h1>双工语音链路 AEC 调试报告 / Duplex AEC debug report</h1>";
    stream << "<p>" << fixed3(config.sample_rate / 1000.0) << " kHz, "
           << fixed3(summary.duration_seconds) << " s. Delay path steps at";
    for (const auto& step : result.delay_steps) {
        stream << " " << fixed3(step.time_seconds) << " s &rarr; " << step.delay_ms
               << " ms;";
    }
    stream << " double talk " << fixed3(dt_start) << "&ndash;" << fixed3(dt_end)
           << " s. Detection held until " << fixed3(result.double_talk_end_detected)
           << " s.</p>";
    stream << "<div class='card'><h2>试听 / Listen</h2>"
              "<p>双讲片段为 "
           << fixed3(std::max(0.0, dt_start - 1.0)) << "&ndash;"
           << fixed3(std::min(summary.duration_seconds, dt_end + 1.0))
           << " s。A/B 立体声左声道是输入，右声道是 AEC 输出。</p>"
              "<p><b>A/B double-talk input (left) vs output (right)</b></p>"
              "<audio controls preload='none' src='double_talk_ab.wav'></audio>"
              "<p><b>Input microphone</b></p>"
              "<audio controls preload='none' src='double_talk_input.wav'></audio>"
              "<p><b>AEC output</b></p>"
              "<audio controls preload='none' src='double_talk_output.wav'></audio>"
              "<p><a href='mic.wav'>full mic</a> &middot; "
              "<a href='aec_out.wav'>full AEC output</a> &middot; "
              "<a href='far.wav'>far reference</a>";
    if (has_near_truth) {
        stream << " &middot; <a href='near_clean.wav'>near clean</a>";
    }
    if (has_echo_truth) {
        stream << " &middot; <a href='true_echo.wav'>true echo</a>";
    }
    stream << "</p></div>";
    stream << "<h2>Timeline</h2><img src='timeline.svg' style='width:100%;"
              "border:1px solid #dbe4ef;background:white'>";
    stream << "<h2>Spectra</h2><p>Microphone (input) and AEC output. X: 0&ndash;"
           << fixed3(summary.duration_seconds)
           << " s, Y: 0&ndash;8 kHz. Color: &minus;85 to &minus;10 dB.</p>"
              "<img src='input_spectrum.png' width='720' height='360'> "
              "<img src='output_spectrum.png' width='720' height='360'>";
    stream << "<h2>Window metrics</h2><table><tr><th>window</th><th>s</th>"
              "<th>ERLE dB</th><th>residual dBFS</th><th>near att. dB</th>"
              "<th>DT fraction</th><th>delay ms</th></tr>";
    for (const auto& window : summary.windows) {
        stream << "<tr><td>" << xml_escape(window.name) << "</td><td>"
               << fixed3(window.start_seconds) << "&ndash;"
               << fixed3(window.end_seconds) << "</td><td>"
               << fixed3(window.erle_db) << "</td><td>"
               << fixed3(window.residual_echo_dbfs) << "</td><td>"
               << fixed3(window.near_attenuation_db) << "</td><td>"
               << fixed3(window.double_talk_fraction) << "</td><td>"
               << fixed3(window.delay_ms) << "</td></tr>";
    }
    stream << "</table>";
    stream << "<h2>Reconvergence intervals</h2>"
              "<p>“healthy from”是首个满足条件的帧，“confirmed”要求连续 0.5 s "
              "保持低残余回声与稳定 ERLE。绿色带可直接定位算法重新收敛的区间。</p>"
              "<table><tr><th>event</th><th>at s</th><th>status</th>"
              "<th>healthy from</th><th>confirmed</th><th>ERLE before</th>"
              "<th>ERLE after</th><th>residual after</th><th>delay ms</th></tr>";
    for (const auto& event : summary.reconvergence) {
        stream << "<tr><td>" << xml_escape(event.label) << "</td><td>"
               << fixed3(event.event_seconds) << "</td><td>"
               << (event.found ? "<span class='ok'>found</span>"
                               : "<span class='warn'>not found</span>")
               << "</td><td>" << fixed3(event.healthy_start_seconds) << "</td><td>"
               << fixed3(event.confirmed_seconds) << "</td><td>"
               << fixed3(event.erle_before_db) << "</td><td>"
               << fixed3(event.erle_after_db) << "</td><td>"
               << fixed3(event.residual_after_dbfs) << "</td><td>"
               << fixed3(event.delay_ms) << "</td></tr>";
    }
    stream << "</table>";
    stream << "<div class='card'><h2>Algorithm settings</h2><p>NLMS taps <code>"
           << config.filter_taps << "</code>, block <code>" << config.block_size
           << "</code>, step <code>" << fixed3(config.delay_step_size)
           << "</code>, delay search <code>" << config.delay_min_ms << "&ndash;"
           << config.delay_max_ms << " ms</code>. Adaptation is frozen while "
              "double-talk is detected; no near-end suppressor is applied, so the "
              "near talker remains audible.</p>"
              "<p>Raw data: <a href='metrics.csv'>metrics.csv</a> &middot; "
              "<a href='summary.json'>summary.json</a></p></div></body>";
    return true;
}

bool write_mono_file(const std::filesystem::path& path, int sample_rate,
                     const std::vector<float>& samples, std::string& error) {
    MonoAudio audio;
    audio.sample_rate = sample_rate;
    audio.samples = samples;
    return write_mono_wav(path, audio, error);
}

}  // namespace

AnalysisSummary analyze_result(const AecResult& result,
                               const std::vector<DelayStep>& true_delay_steps,
                               double dt_start, double dt_end, double duration_seconds) {
    AnalysisSummary summary;
    summary.duration_seconds = duration_seconds;
    summary.detector_release_seconds = result.double_talk_end_detected;
    for (float value : result.output) {
        summary.output_peak_dbfs =
            std::max(summary.output_peak_dbfs,
                     static_cast<double>(power_to_db(value)));
    }

    summary.windows.push_back(summarize_window(
        result.metrics, "before_double_talk", std::max(0.0, dt_start - 2.0),
        std::max(0.0, dt_start - 0.2)));
    summary.windows.push_back(summarize_window(
        result.metrics, "during_double_talk", dt_start + 0.2, dt_end - 0.2));
    summary.windows.push_back(summarize_window(
        result.metrics, "after_double_talk_hold", dt_end + 0.5,
        std::min(duration_seconds, dt_end + 1.8)));
    for (std::size_t i = 1; i < true_delay_steps.size(); ++i) {
        const double event_time = true_delay_steps[i].time_seconds;
        const double next_event =
            i + 1 < true_delay_steps.size() ? true_delay_steps[i + 1].time_seconds
                                            : duration_seconds;
        if (event_time >= dt_start - 1.0 && event_time < dt_end + 1.0) {
            continue;
        }
        if (event_time + 2.0 < next_event - 0.2) {
            summary.windows.push_back(summarize_window(
                result.metrics,
                "delay_" + std::to_string(true_delay_steps[i].delay_ms) + "ms_settled",
                event_time + 2.0, std::min(next_event - 0.2, event_time + 4.5)));
        }
    }

    for (std::size_t i = 1; i < true_delay_steps.size(); ++i) {
        const auto& step = true_delay_steps[i];
        const auto& previous = true_delay_steps[i - 1];
        if (step.time_seconds >= duration_seconds) {
            continue;
        }
        const double search_end =
            std::min(duration_seconds - 0.05, step.time_seconds + 4.0);
        summary.reconvergence.push_back(find_reconvergence(
            result.metrics,
            "delay step " + std::to_string(previous.delay_ms) + " -> " +
                std::to_string(step.delay_ms) + " ms",
            step.time_seconds, step.time_seconds + 0.05, search_end));
    }
    if (dt_end < duration_seconds - 0.1) {
        const double next_delay = [&]() {
            double next = duration_seconds;
            for (const auto& step : true_delay_steps) {
                if (step.time_seconds > dt_end + 0.05) {
                    next = std::min(next, step.time_seconds);
                }
            }
            return next;
        }();
        summary.reconvergence.push_back(find_reconvergence(
            result.metrics, "double-talk ends", dt_end, dt_end,
            std::min(duration_seconds - 0.05, next_delay)));
        std::sort(summary.reconvergence.begin(), summary.reconvergence.end(),
                  [](const auto& left, const auto& right) {
                      return left.event_seconds < right.event_seconds;
                  });
    }
    return summary;
}

bool write_artifact_bundle(const std::filesystem::path& out_dir, int sample_rate,
                           const std::vector<float>& far,
                           const std::vector<float>& near,
                           const std::vector<float>& echo,
                           const std::vector<float>& mic,
                           const AecResult& result,
                           const std::vector<DelayStep>& true_delay_steps,
                           double dt_start, double dt_end,
                           const AecConfig& aec_config,
                           AnalysisSummary& summary, std::string& error) {
    std::error_code fs_error;
    std::filesystem::create_directories(out_dir, fs_error);
    if (fs_error) {
        error = "cannot create output directory: " + fs_error.message();
        return false;
    }
    const double duration = static_cast<double>(result.output.size()) / sample_rate;
    summary = analyze_result(result, true_delay_steps, dt_start, dt_end, duration);
    const bool has_near_truth = !near.empty() && near.size() == result.output.size();
    const bool has_echo_truth = !echo.empty() && echo.size() == result.output.size();

    if (!write_metrics_csv(out_dir / "metrics.csv", result, error) ||
        !write_summary_json(out_dir / "summary.json", summary, result, aec_config,
                            true_delay_steps, dt_start, dt_end,
                            has_near_truth && has_echo_truth, error)) {
        return false;
    }
    if (!write_mono_file(out_dir / "far.wav", sample_rate, far, error) ||
        !write_mono_file(out_dir / "mic.wav", sample_rate, mic, error) ||
        !write_mono_file(out_dir / "aec_out.wav", sample_rate, result.output, error)) {
        return false;
    }
    if (has_near_truth && !write_mono_file(out_dir / "near_clean.wav", sample_rate,
                                           near, error)) {
        return false;
    }
    if (has_echo_truth && !write_mono_file(out_dir / "true_echo.wav", sample_rate,
                                           echo, error)) {
        return false;
    }

    const std::size_t excerpt_begin = static_cast<std::size_t>(
        std::max(0.0, dt_start - 1.0) * sample_rate);
    const std::size_t excerpt_end = std::min(
        result.output.size(),
        static_cast<std::size_t>(std::min(duration, dt_end + 1.0) * sample_rate));
    const auto input_excerpt = slice(mic, excerpt_begin, excerpt_end);
    const auto output_excerpt = slice(result.output, excerpt_begin, excerpt_end);
    if (!write_mono_file(out_dir / "double_talk_input.wav", sample_rate,
                         input_excerpt, error) ||
        !write_mono_file(out_dir / "double_talk_output.wav", sample_rate,
                         output_excerpt, error)) {
        return false;
    }
    StereoAudio ab;
    ab.sample_rate = sample_rate;
    ab.left = input_excerpt;
    ab.right = output_excerpt;
    if (!write_stereo_wav(out_dir / "double_talk_ab.wav", ab, error)) {
        return false;
    }
    if (!write_timeline_svg(out_dir / "timeline.svg", far, near, mic, result.output,
                            result.metrics, true_delay_steps, summary.reconvergence,
                            duration, sample_rate, error) ||
        !write_spectrogram_png(out_dir / "input_spectrum.png", mic, sample_rate,
                               error) ||
        !write_spectrogram_png(out_dir / "output_spectrum.png", result.output,
                               sample_rate, error) ||
        !write_report_html(out_dir / "report.html", summary, result, aec_config,
                           dt_start, dt_end, has_near_truth, has_echo_truth,
                           error)) {
        return false;
    }
    return true;
}

void print_analysis(const AnalysisSummary& summary, const AecResult& result) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Double-talk detector: start=" << result.double_talk_start_detected
              << " s, release=" << result.double_talk_end_detected << " s\n";
    std::cout << "Detected echo path:\n";
    for (const auto& step : result.delay_steps) {
        std::cout << "  " << std::setw(6) << step.time_seconds << " s  delay="
                  << std::setw(4) << step.delay_ms << " ms  corr="
                  << std::setprecision(3) << step.correlation << "\n";
    }
    std::cout << std::setprecision(2);
    std::cout << "Windows (ERLE / residual / near attenuation):\n";
    for (const auto& window : summary.windows) {
        std::cout << "  " << std::left << std::setw(28) << window.name << std::right
                  << " " << window.start_seconds << "-" << window.end_seconds << " s  "
                  << window.erle_db << " dB  " << window.residual_echo_dbfs
                  << " dBFS  ";
        if (std::isfinite(window.near_attenuation_db)) {
            std::cout << window.near_attenuation_db << " dB";
        } else {
            std::cout << "n/a";
        }
        std::cout << "\n";
    }
    std::cout << "Reconvergence:\n";
    for (const auto& event : summary.reconvergence) {
        std::cout << "  " << std::left << std::setw(30) << event.label << std::right;
        if (event.found) {
            std::cout << " healthy " << event.healthy_start_seconds << " s, confirmed "
                      << event.confirmed_seconds << " s, ERLE " << event.erle_after_db
                      << " dB, residual " << event.residual_after_dbfs << " dBFS\n";
        } else {
            std::cout << " no sustained reconvergence found\n";
        }
    }
}

}  // namespace aec

#include "aec.hpp"

#include <array>
#include <cstdlib>
#include <deque>
#include <iostream>

namespace aec {
namespace {

struct DelayEstimate {
    int lag = -1;
    double correlation = 0.0;
};

DelayEstimate estimate_delay(const std::vector<float>& far, const std::vector<float>& mic,
                             std::size_t end, int sample_rate, int min_lag, int max_lag,
                             double window_seconds) {
    const std::size_t window =
        std::min(end, static_cast<std::size_t>(window_seconds * sample_rate));
    if (window < static_cast<std::size_t>(0.20 * sample_rate)) {
        return {};
    }
    const std::size_t start = end - window;
    const int hop = std::max(8, static_cast<int>(0.002 * sample_rate));
    const int frame_length = hop;
    std::vector<double> far_env;
    std::vector<double> mic_env;
    far_env.reserve(window / hop);
    mic_env.reserve(window / hop);
    for (std::size_t i = 0; i + static_cast<std::size_t>(frame_length) <= window;
         i += static_cast<std::size_t>(hop)) {
        double far_energy = 0.0;
        double mic_energy = 0.0;
        for (int k = 0; k < frame_length; ++k) {
            const double x = far[start + i + static_cast<std::size_t>(k)];
            const double y = mic[start + i + static_cast<std::size_t>(k)];
            far_energy += x * x;
            mic_energy += y * y;
        }
        far_env.push_back(std::log(1.0e-7 +
                                   std::sqrt(far_energy / frame_length)));
        mic_env.push_back(std::log(1.0e-7 +
                                   std::sqrt(mic_energy / frame_length)));
    }
    if (far_env.size() < 64) {
        return {};
    }

    const auto remove_mean = [](std::vector<double>& x) {
        double sum = 0.0;
        for (double value : x) {
            sum += value;
        }
        const double mean = sum / static_cast<double>(x.size());
        for (double& value : x) {
            value -= mean;
        }
    };
    remove_mean(far_env);
    remove_mean(mic_env);

    const int min_frames = std::max(0, min_lag / hop);
    const int max_frames =
        std::min(static_cast<int>(far_env.size()) - 8, max_lag / hop);
    int best_frame = -1;
    double best_corr = -1.0;
    for (int lag = min_frames; lag <= max_frames; ++lag) {
        double xy = 0.0;
        double xx = 0.0;
        double yy = 0.0;
        for (std::size_t j = static_cast<std::size_t>(lag); j < mic_env.size(); ++j) {
            const double x = far_env[j - static_cast<std::size_t>(lag)];
            const double y = mic_env[j];
            xy += x * y;
            xx += x * x;
            yy += y * y;
        }
        const double corr = xy / std::sqrt(std::max(xx * yy, 1.0e-18));
        if (corr > best_corr) {
            best_corr = corr;
            best_frame = lag;
        }
    }
    if (best_frame < 0 || best_corr < 0.20) {
        return {};
    }

    int best_lag = best_frame * hop;
    double refined_corr = -1.0;
    const int refine_radius = hop;
    for (int lag = std::max(min_lag, best_lag - refine_radius);
         lag <= std::min(max_lag, best_lag + refine_radius); ++lag) {
        double xy = 0.0;
        double xx = 0.0;
        double yy = 0.0;
        for (std::size_t i = static_cast<std::size_t>(lag); i < window; ++i) {
            const double x = far[start + i - static_cast<std::size_t>(lag)];
            const double y = mic[start + i];
            xy += x * y;
            xx += x * x;
            yy += y * y;
        }
        const double corr = xy / std::sqrt(std::max(xx * yy, 1.0e-18));
        if (corr > refined_corr) {
            refined_corr = corr;
            best_lag = lag;
        }
    }
    return {best_lag, refined_corr};
}

void compute_frame_metrics(const std::vector<float>& far, const std::vector<float>& mic,
                           const std::vector<float>& output,
                           const std::vector<uint8_t>& dt_flags,
                           const std::vector<int>& delay_samples,
                           const GroundTruth& truth, int sample_rate,
                           std::vector<AecFrameMetric>& metrics) {
    const std::size_t frame_size = static_cast<std::size_t>(0.05 * sample_rate);
    const bool has_near = truth.near_clean != nullptr;
    const bool has_echo = truth.true_echo != nullptr;
    for (std::size_t begin = 0; begin < output.size(); begin += frame_size) {
        const std::size_t end = std::min(output.size(), begin + frame_size);
        if (end - begin < frame_size / 2) {
            break;
        }
        AecFrameMetric metric;
        metric.time_seconds = static_cast<double>(begin) / sample_rate;
        metric.far_dbfs = power_to_db(rms(far, begin, end));
        metric.mic_dbfs = power_to_db(rms(mic, begin, end));
        metric.output_dbfs = power_to_db(rms(output, begin, end));
        if (has_near) {
            metric.near_dbfs = power_to_db(rms(*truth.near_clean, begin, end));
        }
        if (has_echo) {
            metric.echo_dbfs = power_to_db(rms(*truth.true_echo, begin, end));
        }
        int dt_count = 0;
        for (std::size_t i = begin; i < end; ++i) {
            dt_count += dt_flags[i] != 0 ? 1 : 0;
        }
        metric.double_talk_fraction =
            static_cast<double>(dt_count) / static_cast<double>(end - begin);
        metric.double_talk = metric.double_talk_fraction > 0.5 ? 1 : 0;
        metric.delay_ms =
            delay_samples.empty()
                ? 0.0
                : 1000.0 * delay_samples[std::min(end - 1, delay_samples.size() - 1)] /
                      sample_rate;

        double residual_energy = 0.0;
        for (std::size_t i = begin; i < end; ++i) {
            double residual = output[i];
            if (has_near) {
                residual -= (*truth.near_clean)[i];
            }
            residual_energy += residual * residual;
        }
        const double residual_power = residual_energy / static_cast<double>(end - begin);
        metric.residual_echo_dbfs = db_from_power(residual_power);

        if (has_echo) {
            double echo_energy = 0.0;
            for (std::size_t i = begin; i < end; ++i) {
                const double echo = (*truth.true_echo)[i];
                echo_energy += echo * echo;
            }
            if (echo_energy > 1.0e-10 && residual_energy > 1.0e-12) {
                metric.erle_db = 10.0 * std::log10(echo_energy / residual_energy);
            }
        } else if (metric.double_talk_fraction < 0.2) {
            const double mic_energy = dot(mic, mic, begin, end);
            const double out_energy = dot(output, output, begin, end);
            if (mic_energy > 1.0e-10 && out_energy > 1.0e-12) {
                metric.erle_db = 10.0 * std::log10(mic_energy / out_energy);
            }
        }

        if (has_near) {
            const double near_energy =
                dot(*truth.near_clean, *truth.near_clean, begin, end);
            if (near_energy > 1.0e-8) {
                const double projection = dot(output, *truth.near_clean, begin, end);
                const double gain = projection / near_energy;
                metric.near_attenuation_db =
                    -20.0 * std::log10(std::max(std::abs(gain), 1.0e-5));
            }
        }
        metrics.push_back(metric);
    }
}

}  // namespace

AecResult run_aec(const std::vector<float>& far, const std::vector<float>& mic,
                  const AecConfig& config, const GroundTruth& truth) {
    AecResult result;
    const std::size_t count = std::min(far.size(), mic.size());
    result.output.assign(count, 0.0f);
    result.double_talk_samples.assign(count, 0);
    if (count == 0 || config.sample_rate <= 0) {
        return result;
    }

    const int taps = std::max(32, config.filter_taps);
    const int block = std::max(16, config.block_size);
    const double sample_rate = config.sample_rate;
    const int min_lag = std::max(0, config.delay_min_ms * config.sample_rate / 1000);
    const int max_lag = std::max(min_lag + 1,
                                 config.delay_max_ms * config.sample_rate / 1000);
    const std::size_t estimate_interval =
        std::max<std::size_t>(1, static_cast<std::size_t>(
                                     config.delay_estimate_interval * sample_rate));
    const std::size_t correlation_window = std::max<std::size_t>(
        64, static_cast<std::size_t>(config.near_active_correlation_window *
                                     sample_rate));

    std::vector<double> weights(static_cast<std::size_t>(taps), 0.0);
    std::vector<double> stable_weights(static_cast<std::size_t>(taps), 0.0);
    std::vector<double> dt_weight_backup;
    std::vector<double> xblock(static_cast<std::size_t>(block) * taps, 0.0);
    std::vector<double> errors(static_cast<std::size_t>(block), 0.0);
    std::vector<int> delay_samples(count, 0);

    int current_delay = -1;
    std::deque<int> recent_lags;
    int last_median_lag = -1;
    int stable_median_count = 0;
    bool delay_change_pending = false;
    int recovery_blocks = 0;
    int enter_count = 0;
    bool double_talk = false;
    bool dt_started = false;
    bool dt_ended = false;
    double near_absence_seconds = 0.0;
    double last_echo_rms = 0.0;
    double last_output_rms = 0.0;
    std::size_t next_estimate = static_cast<std::size_t>(0.50 * sample_rate);
    const int warmup_blocks = std::max(
        1, static_cast<int>(config.recovery_warmup * sample_rate / block));
    const bool debug = std::getenv("AEC_DEBUG") != nullptr;

    for (std::size_t block_begin = 0; block_begin < count; block_begin += block) {
        const std::size_t block_end = std::min(count, block_begin + block);
        const std::size_t block_length = block_end - block_begin;

        if (block_begin >= next_estimate) {
            next_estimate = block_begin + estimate_interval;
            const DelayEstimate estimate =
                double_talk
                    ? DelayEstimate{}
                    : estimate_delay(far, mic, block_begin, config.sample_rate,
                                     min_lag, max_lag, config.delay_window);
            if (estimate.lag >= 0) {
                if (current_delay >= 0) {
                    delay_change_pending =
                        std::abs(estimate.lag - current_delay) > 8 &&
                        estimate.correlation > 0.65;
                }
                recent_lags.push_back(estimate.lag);
                while (recent_lags.size() > 9) {
                    recent_lags.pop_front();
                }
                std::vector<int> sorted(recent_lags.begin(), recent_lags.end());
                std::nth_element(
                    sorted.begin(),
                    sorted.begin() + static_cast<std::ptrdiff_t>(sorted.size() / 2),
                    sorted.end());
                const int median_lag = sorted[sorted.size() / 2];
                if (debug) {
                    std::cerr << "[delay] t=" << block_begin / sample_rate
                              << " current=" << current_delay
                              << " candidate=" << estimate.lag
                              << " median=" << median_lag
                              << " corr=" << estimate.correlation
                              << " pending=" << delay_change_pending << "\n";
                }
                bool changed = false;
                if (current_delay < 0) {
                    current_delay = median_lag;
                    changed = true;
                } else if (std::abs(median_lag - current_delay) <= 8) {
                    last_median_lag = median_lag;
                    stable_median_count = 0;
                } else {
                    if (last_median_lag >= 0 &&
                        std::abs(median_lag - last_median_lag) <= 8) {
                        ++stable_median_count;
                    } else {
                        stable_median_count = 1;
                    }
                    last_median_lag = median_lag;
                    if (stable_median_count >= 3) {
                        current_delay = median_lag;
                        changed = true;
                        delay_change_pending = false;
                        recent_lags.clear();
                        recent_lags.push_back(median_lag);
                        last_median_lag = median_lag;
                        stable_median_count = 0;
                    }
                }
                if (changed) {
                    std::fill(weights.begin(), weights.end(), 0.0);
                    std::fill(stable_weights.begin(), stable_weights.end(), 0.0);
                    dt_weight_backup.clear();
                    recovery_blocks = warmup_blocks;
                    enter_count = 0;
                    double_talk = false;
                    last_echo_rms = 0.0;
                    last_output_rms = 0.0;
                    result.delay_steps.push_back(
                        {static_cast<double>(block_begin) / sample_rate,
                         static_cast<int>(std::lrint(
                             1000.0 * current_delay / config.sample_rate)),
                         current_delay, estimate.correlation});
                }
            }
        }

        const std::size_t corr_begin =
            block_begin > correlation_window ? block_begin - correlation_window : 0;
        double aligned_energy = 0.0;
        double mic_energy = 0.0;
        double aligned_mic = 0.0;
        if (current_delay >= 0) {
            for (std::size_t i = corr_begin; i < block_begin; ++i) {
                const int source = static_cast<int>(i) - current_delay;
                const double x = source >= 0 ? far[static_cast<std::size_t>(source)] : 0.0;
                const double y = mic[i];
                aligned_energy += x * x;
                mic_energy += y * y;
                aligned_mic += x * y;
            }
        }
        const double correlation =
            aligned_mic / std::sqrt(std::max(aligned_energy * mic_energy, 1.0e-18));
        const double aligned_rms =
            std::sqrt(aligned_energy / static_cast<double>(std::max<std::size_t>(
                          1, block_begin - corr_begin)));
        const double mic_rms =
            std::sqrt(mic_energy / static_cast<double>(std::max<std::size_t>(
                         1, block_begin - corr_begin)));
        const double far_rms = power_to_db(aligned_rms);
        const double mic_rms_db = power_to_db(mic_rms);

        if (recovery_blocks > 0) {
            --recovery_blocks;
        }
        const bool activity =
            current_delay >= 0 && far_rms > -50.0 && mic_rms_db > -47.0 &&
            block_begin - corr_begin >= correlation_window / 2;
        const double relative_level =
            mic_rms / std::max(aligned_rms, 1.0e-6);
        const bool energy_increase =
            mic_rms > 1.20 * (last_echo_rms + 1.0e-5) ||
            mic_rms > 1.8 * (last_output_rms + 1.0e-5);
        const bool correlation_drop =
            correlation < config.dtd_enter_correlation ||
            (relative_level > 1.35 && correlation < 0.90);
        const bool near_candidate =
            activity && recovery_blocks == 0 && correlation_drop &&
            energy_increase && relative_level > 1.35 && !delay_change_pending;
        const bool adaptation_hold = near_candidate;
        if (debug && (near_candidate || correlation_drop)) {
            std::cerr << "[dtd] t=" << block_begin / sample_rate
                      << " corr=" << correlation << " rel=" << relative_level
                      << " mic=" << mic_rms_db << " err=" << last_echo_rms
                      << " candidate=" << near_candidate
                      << " dt=" << double_talk << "\n";
        }

        if (!double_talk) {
            enter_count = near_candidate ? enter_count + 1 : 0;
            if (enter_count >= config.dtd_enter_blocks) {
                double_talk = true;
                dt_weight_backup = stable_weights;
                weights = stable_weights;
                near_absence_seconds = 0.0;
                enter_count = 0;
                if (!dt_started) {
                    result.double_talk_start_detected =
                        static_cast<double>(block_begin) / sample_rate;
                    dt_started = true;
                }
            }
        } else {
            const bool near_evidence =
                mic_rms_db > -48.0 &&
                (last_output_rms > 0.50 * mic_rms + 1.0e-5 ||
                 (activity && (correlation_drop || relative_level > 1.20)));
            if (near_evidence) {
                near_absence_seconds = 0.0;
            } else if (activity) {
                near_absence_seconds +=
                    static_cast<double>(block_length) / sample_rate;
            }
            if (near_absence_seconds >= config.dtd_release_hold) {
                double_talk = false;
                if (!dt_weight_backup.empty()) {
                    weights = dt_weight_backup;
                    dt_weight_backup.clear();
                }
                enter_count = 0;
                near_absence_seconds = 0.0;
                if (!dt_ended) {
                    result.double_talk_end_detected =
                        static_cast<double>(block_begin) / sample_rate;
                    dt_ended = true;
                }
            }
        }

        double block_energy = 0.0;
        double window_energy = 0.0;
        double block_error_energy = 0.0;
        double block_output_energy = 0.0;
        for (std::size_t local = 0; local < block_length; ++local) {
            const std::size_t n = block_begin + local;
            const int source = static_cast<int>(n) - current_delay;
            const double x_val =
                current_delay >= 0 && source >= 0
                    ? far[static_cast<std::size_t>(source)]
                    : 0.0;
            block_energy += x_val * x_val;
            double y_hat = 0.0;
            double sample_energy = 0.0;
            double* row = xblock.data() + local * taps;
            row[0] = x_val;
            sample_energy += x_val * x_val;
            for (int i = 1; i < taps; ++i) {
                const int history_source = source - i;
                const double value =
                    history_source >= 0
                        ? far[static_cast<std::size_t>(history_source)]
                        : 0.0;
                row[i] = value;
                sample_energy += value * value;
                y_hat += weights[static_cast<std::size_t>(i)] * value;
            }
            y_hat += weights[0] * x_val;
            window_energy += sample_energy;
            const double error = mic[n] - y_hat;
            errors[local] = error;
            result.output[n] = static_cast<float>(error);
            result.double_talk_samples[n] = double_talk ? 1 : 0;
            block_error_energy += error * error;
            block_output_energy += error * error;
        }

        const bool can_adapt = current_delay >= 0 && !double_talk &&
                               !adaptation_hold && block_energy > 1.0e-8 &&
                               power_to_db(std::sqrt(
                                   block_energy / static_cast<double>(block_length))) >
                                   -40.0;
        if (!double_talk && !near_candidate && !delay_change_pending &&
            correlation > 0.90 && relative_level < 0.80) {
            stable_weights = weights;
        }
        if (can_adapt) {
            const double normalization =
                config.delay_step_size / (window_energy + 1.0e-8);
            for (int i = 0; i < taps; ++i) {
                double gradient = 0.0;
                for (std::size_t local = 0; local < block_length; ++local) {
                    gradient += errors[local] * xblock[local * taps + i];
                }
                weights[static_cast<std::size_t>(i)] =
                    0.99995 * weights[static_cast<std::size_t>(i)] +
                    normalization * gradient;
            }
        }

        last_echo_rms =
            std::sqrt(block_error_energy / static_cast<double>(block_length));
        last_output_rms =
            std::sqrt(block_output_energy / static_cast<double>(block_length));
        double weight_energy = 0.0;
        for (double weight : weights) {
            weight_energy += weight * weight;
        }
        if (!(weight_energy < 1.0e8)) {
            std::fill(weights.begin(), weights.end(), 0.0);
            recovery_blocks = warmup_blocks;
        }
        for (std::size_t local = 0; local < block_length; ++local) {
            delay_samples[block_begin + local] =
                std::max(0, current_delay);
        }
    }

    compute_frame_metrics(far, mic, result.output, result.double_talk_samples,
                          delay_samples, truth, config.sample_rate, result.metrics);
    return result;
}

}  // namespace aec

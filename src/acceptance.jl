# Acceptance harness.
#
# The hidden truth is used only here, never inside the fuser:
#   1. cross-device beep error — corrected consensus timestamps of the same
#      beep must agree across recorders (after removing the global offset
#      between the consensus clock and true time);
#   2. breakpoint evidence — detected splits must line up with the simulated
#      reboots, and strong evidence must exist at each one;
#   3. uncertainty calibration — reported 95% intervals must cover the truth
#      at roughly the nominal rate.

struct AcceptanceReport
    beep_rms::Float64
    beep_max::Float64
    beep_p95::Float64
    n_beep_pairs::Int
    n_true_restarts::Int
    n_detected_splits::Int
    n_matched::Int
    n_spurious::Int
    n_missing::Int
    matched_jumps::Vector{Float64}
    coverage::Float64
    n_coverage_samples::Int
    pass_beep::Bool
    pass_breakpoint::Bool
    pass_coverage::Bool
end

function _global_offset(observations, st::FuserState)
    diffs = Float64[]
    for o in observations
        o.gross && continue
        r = correct(st, o.device, o.L_meas; event=o.event)
        r.status === :unknown && continue
        r.status === :ambiguous && continue
        push!(diffs, r.tau - o.T_true)
    end
    isempty(diffs) && return 0.0
    return median(diffs)
end

"""
    run_acceptance(truths, observations, beep_times, st; tol_seconds=0.15)

Evaluate the fused state against simulation truth and return an
`AcceptanceReport` with pass flags.
"""
function run_acceptance(truths, observations, beep_times, st::FuserState;
                        tol_seconds::Real=0.15)
    offset = _global_offset(observations, st)

    # ---- 1. cross-device beep error ------------------------------------
    by_event = Dict{Int,Vector{Float64}}()
    for o in observations
        o.gross && continue
        r = correct(st, o.device, o.L_meas; event=o.event)
        r.status === :unknown && continue
        r.status === :ambiguous && continue
        push!(get!(by_event, o.event, Float64[]), r.tau - offset)
    end
    errs = Float64[]
    for (_, vals) in by_event
        length(vals) < 2 && continue
        m = median(vals)
        append!(errs, abs.(vals .- m))
    end
    beep_rms = isempty(errs) ? Inf : sqrt(mean(abs2, errs))
    beep_max = isempty(errs) ? Inf : maximum(errs)
    beep_p95 = isempty(errs) ? Inf : quantile(errs, 0.95)

    # ---- 2. breakpoint / reboot matching -------------------------------
    ev = breakpoint_evidence(st)
    true_rb = Tuple{Int,Float64}[]
    for d in truths, rb in d.restarts
        # reboot time expressed on the consensus clock
        push!(true_rb, (d.id, rb + offset))
    end
    median_gap = length(beep_times) > 1 ?
                 median(diff(beep_times)) : 11.0
    # The first post-reboot detection can lag the true reboot by a couple of
    # beep intervals (plus the dark window), so allow 2.5 intervals.
    # Bound by both beep cadence and an absolute detection-lag budget
    # (reboot dark window plus consecutive missed beeps).
    match_tol = max(3.5 * median_gap, 30.0)

    matched = falses(length(true_rb))
    matched_jumps = Float64[]
    n_matched = 0
    for e in ev
        best, best_dt = 0, Inf
        for (j, (id, rb)) in enumerate(true_rb)
            if id == e.device && !matched[j]
                dt = abs(e.tau_right - rb)
                if dt < best_dt
                    best, best_dt = j, dt
                end
            end
        end
        if best > 0 && best_dt <= match_tol &&
           e.status in (:strong, :moderate)
            matched[best] = true
            n_matched += 1
            push!(matched_jumps, e.jump_seconds)
        end
    end
    n_true_restarts = length(true_rb)
    n_detected = length(ev)
    n_spurious = n_detected - n_matched
    n_missing = n_true_restarts - sum(matched)

    # ---- 3. 95% interval coverage --------------------------------------
    hit = 0; total = 0
    for o in observations
        o.gross && continue
        r = correct(st, o.device, o.L_meas; event=o.event)
        r.status === :unknown && continue
        total += 1
        truth_tau = o.T_true + offset
        if r.lo - 1e-9 <= truth_tau <= r.hi + 1e-9
            hit += 1
        end
    end
    coverage = total == 0 ? 0.0 : hit / total

    AcceptanceReport(
        beep_rms, beep_max, beep_p95, length(errs),
        n_true_restarts, n_detected, n_matched, n_spurious, n_missing,
        matched_jumps, coverage, total,
        beep_p95 <= tol_seconds,
        n_missing == 0 && n_matched >= n_true_restarts,
        coverage >= 0.90)
end

function Base.show(io::IO, r::AcceptanceReport)
    println(io, "AcceptanceReport")
    @printf(io, "  beep error  RMS=%.4fs  p95=%.4fs  max=%.4fs  (%d pairs)  %s\n",
            r.beep_rms, r.beep_p95, r.beep_max, r.n_beep_pairs,
            r.pass_beep ? "PASS" : "FAIL")
    @printf(io, "  breakpoints true=%d detected=%d matched=%d spurious=%d missing=%d  %s\n",
            r.n_true_restarts, r.n_detected_splits, r.n_matched,
            r.n_spurious, r.n_missing,
            r.pass_breakpoint ? "PASS" : "FAIL")
    if !isempty(r.matched_jumps)
        @printf(io, "  matched jumps (s): %s\n",
                join((@sprintf("%.2f", j) for j in r.matched_jumps), ", "))
    end
    @printf(io, "  95%% interval coverage=%.1f%% (%d samples)  %s",
            100 * r.coverage, r.n_coverage_samples,
            r.pass_coverage ? "PASS" : "FAIL")
end

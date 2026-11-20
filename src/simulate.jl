# Forest simulation: independent thermal-drifting clocks, mid-run reboots,
# missed beeps and occasional gross timestamp errors.

"""
    local_clock(era::Era, T::Real)

Local clock reading at true time `T` within `era` (T >= era.T0).
"""
function local_clock(e::Era, T::Real)
    dt = T - e.T0
    drift = (e.w1 / e.omega) * (cos(e.omega * e.T0 + e.phase) -
                                cos(e.omega * T  + e.phase))
    return e.L0 + e.beta * dt + drift
end

"""
    local_clock(dt::DeviceTruth, T::Real)

Local reading at true time `T`, selecting the era containing `T`.
Returns `NaN` if the device is powered off at `T`.
"""
function local_clock(d::DeviceTruth, T::Real)
    idx = 0
    for (k, e) in enumerate(d.eras)
        if T >= e.T0
            idx = k
        end
    end
    idx == 0 && return NaN
    return local_clock(d.eras[idx], T)
end

function _make_era(rng::AbstractRNG, device::Int, T0::Real, L_prev::Float64;
                   reboot::Bool=false)
    ppm     = rand(rng, -180.0:80.0:180.0)
    beta    = 1.0 + ppm * 1e-6
    # Thermal drift: daily-ish plus faster components. w1 is a rate (s/s),
    # tens to a few hundred ppm peak-to-peak amplitude.
    period  = rand(rng, 240.0:60.0:600.0)
    w1      = rand(rng, 60.0:40.0:260.0) * 1e-6
    omega   = 2π / period
    phase   = rand(rng, 0.0:0.1:2π)
    # Fresh offset after a reboot: the new clock starts from an arbitrary
    # small reading instead of continuing the old one.
    L0 = reboot ? rand(rng, 5.0:1.0:60.0) : L_prev
    return Era(device, Float64(T0), L0, beta, w1, omega, phase)
end

"""
    simulate_forest(cfg::SimConfig)

Build `n_devices` independent recorders and a shared beep schedule. Returns
`(truths, observations, beep_times)`. Each device boots at a slightly
different time and suffers 0-2 mid-run reboots. Detections are Bernoulli
thinned and a small fraction get gross timestamp errors.
"""
function simulate_forest(cfg::SimConfig)
    rng = MersenneTwister(cfg.rng_seed)
    n  = cfg.n_devices
    # Common short beeps on a jittered schedule.
    beep_times = Float64[]
    t = cfg.beep_interval
    while t < cfg.T_end
        push!(beep_times, t + rand(rng, -0.35:0.05:0.35))
        t += cfg.beep_interval
    end

    truths = DeviceTruth[]
    for id in 1:n
        boot   = rand(rng, 0.0:1.0:25.0)
        offset = rand(rng, -8.0:0.5:8.0)
        eras   = Era[]
        push!(eras, _make_era(rng, id, boot, offset))
        # 1-2 reboots per device, at least ~250 s apart and away from edges.
        n_reboot = rand(rng, 1:2)
        cand = sort!(rand(rng, 250.0:25.0:(cfg.T_end - 250.0), n_reboot + 2))
        restarts = Float64[]
        last = boot
        for rb in cand
            length(restarts) >= n_reboot && break
            rb - last < 220.0 && continue
            # The clock is dark for a few seconds across a reboot.
            T_back = rb + rand(rng, 3.0:1.0:12.0)
            push!(restarts, rb)
            push!(eras, _make_era(rng, id, T_back, 0.0; reboot=true))
            last = T_back
        end
        push!(truths, DeviceTruth(id, boot, eras, restarts))
    end

    observations = Obs[]
    for (event, Tb) in enumerate(beep_times)
        for d in truths
            L = local_clock(d, Tb)
            isnan(L) && continue                  # powered off
            rand(rng) > cfg.detect_prob && continue  # missed beep
            gross = rand(rng) < cfg.gross_prob
            Lm = if gross
                sign = rand(rng, Bool) ? 1.0 : -1.0
                L + sign * rand(rng, 1.5:0.25:9.0)
            else
                L + randn(rng) * 0.012            # 12 ms detection jitter
            end
            push!(observations, Obs(d.id, event, Tb, Lm, gross))
        end
    end
    return truths, observations, beep_times
end

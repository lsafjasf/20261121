# Core data structures for the forest clock problem.

"""
    Era(device, T0, L0, beta, w0, w1, phase)

One continuous running interval of a device clock. A mid-run reboot creates a
new era with a fresh offset. The local reading at true time `T` is

    L(T) = L0 + beta*(T-T0) + (w1/omega)*(cos(omega*T0+phase)-cos(omega*T+phase))

where `beta = 1 + ppm*1e-6` is the mean rate and the sinusoidal term is the
temperature-driven rate wander (integration of `w1*sin(omega*T+phase)`).
"""
struct Era
    device::Int
    T0::Float64          # true time at which this era starts
    L0::Float64          # local clock reading at T0
    beta::Float64        # mean rate 1 + ppm*1e-6
    w1::Float64          # thermal drift amplitude (rate units, per second)
    omega::Float64       # thermal cycle angular frequency (rad / second)
    phase::Float64
end

"""
    DeviceTruth(id, boot_time, eras, restarts)

Ground truth for one recorder. `restarts` holds the true times at which the
device rebooted (the start time of every era after the first).
"""
struct DeviceTruth
    id::Int
    boot_time::Float64
    eras::Vector{Era}
    restarts::Vector{Float64}
end

"""
    Obs(device, event, T_true, L_meas, gross)

One detected beep. `T_true` is the (hidden) consensus time of the beep, kept
only for validation; the fuser never reads it. `L_meas` is the local
timestamp, `gross` flags a spurious timestamp used to stress robustness.
"""
struct Obs
    device::Int
    event::Int
    T_true::Float64
    L_meas::Float64
    gross::Bool
end

struct SimConfig
    n_devices::Int
    T_end::Float64
    beep_interval::Float64
    detect_prob::Float64       # probability a device records a given beep
    gross_prob::Float64        # probability an inlier detection is corrupted
    rng_seed::Int
end

SimConfig(; n_devices=4, T_end=1800.0, beep_interval=11.0,
          detect_prob=0.92, gross_prob=0.02, rng_seed=20261121) =
    SimConfig(n_devices, T_end, beep_interval, detect_prob, gross_prob, rng_seed)

"""
    Segment(id, L0, tau0, a, b, P, anchors, closed_at,
            first_event, last_event, confirmed)

Fitted piece on one device, mapping local time `L` to consensus time `tau`:

    tau = a + b * (L - L0)

`P` is the 2x2 Kalman covariance of `(a, b)`. `anchors` stores the consensus
anchors `(L, tau, event)` that built the piece. `first_event`/`last_event`
give the beep-index window (used to disambiguate eras after a clock wrap),
and `confirmed` marks pieces that passed the trial-and-clean stage.
"""
mutable struct Segment
    id::Int
    L0::Float64
    tau0::Float64
    a::Float64
    b::Float64
    P::Matrix{Float64}
    anchors::Vector{Tuple{Float64,Float64,Int}}   # (L, tau, event_index)
    closed_at::Float64
    first_event::Int
    last_event::Int
    confirmed::Bool
end

"""
    CorrectionResult(tau, sigma, status, segment_id)

Result of mapping one local timestamp to consensus time.

  * `status = :interpolated` — inside a fitted segment's anchor span.
  * `status = :extrapolated` — outside every fitted span; uncertainty grows
    with the distance to the nearest anchor (process noise).
  * `status = :ambiguous`    — inside a gap between two pieces (e.g. the
    dead window of a reboot); only a conservative interval is available.
  * `status = :unknown`      — no fitted piece exists yet.
"""
struct CorrectionResult
    tau::Float64
    sigma::Float64
    lo::Float64
    hi::Float64
    status::Symbol
    segment_id::Int
end

function Base.show(io::IO, r::CorrectionResult)
    print(io, @sprintf("Correction(tau=%.3f ± %.3f, %s, seg=%d)",
                       r.tau, r.sigma, r.status, r.segment_id))
end

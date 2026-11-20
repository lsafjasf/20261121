# Streaming, reference-free multi-device clock fusion.
#
# Every beep yields one consensus time `tau`, estimated as a variance-weighted
# median of the participating devices' own predictions (so a device that
# jumped cannot drag the consensus with it). Each device then fits its local
# timestamps to `tau` with a piecewise affine Kalman filter; two consecutive
# rejected innovations (or one very large one) close the current piece and
# open a new one — the signature of a reboot.

struct FuserConfig
    q_a::Float64          # intercept process-noise rate (s^2 per local s)
    q_b::Float64          # slope process-noise rate
    R::Float64            # detection timestamp variance (s^2)
    gate_chi2::Float64    # Mahalanobis rejection threshold
    hard_jump::Float64    # absolute error (s) that splits immediately
    confirm_anchors::Int  # anchors a new piece needs to be committed
    sigma_floor::Float64  # minimum honest correction uncertainty (s)
end

FuserConfig(; q_a=2.0e-6, q_b=1.0e-11, R=(0.02)^2,
            gate_chi2=GATE_CHI2, hard_jump=HARD_JUMP,
            confirm_anchors=4, sigma_floor=0.12) =
    FuserConfig(q_a, q_b, R, gate_chi2, hard_jump,
                confirm_anchors, sigma_floor)

mutable struct DeviceState
    id::Int
    segments::Vector{Segment}
    fails::Int
    outliers::Int                            # isolated gross errors, dropped
    pending::Vector{Tuple{Float64,Float64,Int}}  # rejected (L,tau,event)
    last_good_event::Int
    n_anchors::Int
    n_rejected::Int
    n_splits::Int
end

DeviceState(id::Int) =
    DeviceState(id, Segment[], 0, 0, Tuple{Float64,Float64,Int}[],
                0, 0, 0, 0)

mutable struct FuserState
    cfg::FuserConfig
    devices::Dict{Int,DeviceState}
    event_count::Int
    last_tau::Float64
    _boot_L::Vector{Dict{Int,Float64}}
end

FuserState(cfg::FuserConfig=FuserConfig()) =
    FuserState(cfg, Dict{Int,DeviceState}(), 0, NaN,
               Dict{Int,Float64}[])

Base.show(io::IO, s::FuserState) =
    print(io, "FuserState($(length(s.devices)) devices, $(s.event_count) events)")

active(d::DeviceState) = isempty(d.segments) ? nothing : last(d.segments)

# ---- consensus time -------------------------------------------------------

"""
    predict_consensus(st::FuserState, stamp::Dict{Int,Float64})

Variance-weighted median of per-device predictions at one beep. Devices with
fewer than two anchors cannot predict and are ignored until the consensus
exists. Returns `(tau, var_tau)` or `nothing` when nobody can predict.
"""
function predict_consensus(st::FuserState, stamp::Dict{Int,Float64})
    preds = Tuple{Float64,Float64}[]   # (tau_pred, weight)
    for (id, L) in stamp
        d = get(st.devices, id, nothing)
        d === nothing && continue
        seg = active(d)
        seg === nothing && continue
        # Pieces with fewer than two real anchors cannot predict reliably.
        length(seg.anchors) < 2 && continue
        tau_pred, S, = predict_tau(seg, L; q_a=st.cfg.q_a,
                                   q_b=st.cfg.q_b, R=st.cfg.R)
        push!(preds, (tau_pred, 1.0 / max(S, 1e-12)))
    end
    isempty(preds) && return nothing
    tau, var_tau = weighted_median(preds)
    return tau, var_tau
end

"""
    weighted_median(pairs)

Robust central value: sort by prediction and walk the cumulative weight.
"""
function weighted_median(pairs::Vector{Tuple{Float64,Float64}})
    length(pairs) == 1 && return pairs[1][1], 1.0 / pairs[1][2]
    p = sort!(copy(pairs); by=x -> x[1])
    W = sum(w for (_, w) in p)
    acc = 0.0
    tau = p[end][1]
    for (v, w) in p
        acc += w
        if acc >= 0.5 * W
            tau = v
            break
        end
    end
    # Robust scale: weighted MAD of the individual predictions around tau.
    devs = sort!(abs.(v - tau for (v, _) in p))
    mad  = quantile(devs, 0.5)
    var_tau = max((1.4826 * mad)^2, minimum(1.0 / w for (_, w) in p))
    return tau, var_tau
end

# ---- streaming ingestion --------------------------------------------------

"""
    ingest_event!(st, stamps)

Process one shared beep. `stamps` maps device id to its local timestamp.
Returns the consensus time `tau` assigned to the event (or `NaN` if the
event was dropped).
"""
function ingest_event!(st::FuserState, stamps::Dict{Int,Float64})
    for id in keys(stamps)
        haskey(st.devices, id) || (st.devices[id] = DeviceState(id))
    end
    pc = predict_consensus(st, stamps)
    bootstrapping = pc === nothing
    if bootstrapping
        # Cold start: remember raw stamps. The third observed beep becomes
        # the first consensus event, timed by the lowest-id device's local
        # clock (unit slope); every other device then aligns to it.
        push!(st._boot_L, copy(stamps))
        length(st._boot_L) > 3 && popfirst!(st._boot_L)
        if length(st._boot_L) < 3
            tau, var_tau = NaN, 1.0
            st.event_count += 1
            st.last_tau = tau
            return tau
        end
        anchor_id = minimum(keys(stamps))
        # First consensus. Absolute readings carry unknown per-device boot
        # offsets, so anchor the origin on one device and set the inter-beep
        # increment from the robust median of local-time deltas (offset
        # invariant), which rejects a single gross timestamp.
        present_ids = sort([id for id in keys(stamps)
                            if haskey(st._boot_L[2], id)])
        isempty(present_ids) && (present_ids = sort(collect(keys(stamps))))
        anchor_id = present_ids[1]
        deltas = Float64[]
        for (id, Ln) in stamps
            Lp = get(st._boot_L[2], id, nothing)
            Lp === nothing && continue
            push!(deltas, Ln - Lp)
        end
        step = median(deltas)
        origin = get(st._boot_L[2], anchor_id, stamps[anchor_id])
        tau = origin + step
        var_tau = 1.0
        bootstrapping = false
    else
        tau, var_tau = pc
    end
    st.event_count += 1
    st.last_tau = tau
    R_eff = st.cfg.R + var_tau
    bootstrapping && return tau
    for (id, L) in stamps
        _absorb_boot!(st.devices[id], L, tau, R_eff, st.cfg,
                      st._boot_L, st.event_count)
    end
    return tau
end

function _absorb_boot!(d::DeviceState, L::Real, tau::Real, R::Real,
                       cfg::FuserConfig,
                       boot_L::Vector{Dict{Int,Float64}}, ev::Int)
    seg = active(d)
    if seg === nothing
        # First real consensus: back-fill the previous beep via the local
        # time difference (unit slope), then absorb the current one.
        if length(boot_L) >= 2 && haskey(boot_L[1], d.id)
            L_prev = boot_L[1][d.id]
            tau_prev = tau - (L - L_prev)
            seg = init_segment(d.n_splits + 1, L_prev, tau_prev)
            seg.first_event = ev - 1
            seg.last_event = ev
            seg.confirmed = true
            push!(d.segments, seg)
            push!(seg.anchors, (Float64(L_prev), Float64(tau_prev), ev - 1))
            d.n_anchors += 1
            push!(seg.anchors, (Float64(L), Float64(tau), ev))
            d.n_anchors += 1
        else
            seg = init_segment(d.n_splits + 1, L, tau)
            seg.first_event = ev
            seg.last_event = ev
            seg.confirmed = true
            push!(d.segments, seg)
            push!(seg.anchors, (Float64(L), Float64(tau), ev))
            d.n_anchors += 1
        end
        return
    end
    if length(seg.anchors) == 1
        seg.b = 1.0
        seg.a = tau - seg.b * (L - seg.L0)
        seg.P = [R * 4        0.0;
                 0.0     (3e-4)^2]
        push!(seg.anchors, (Float64(L), Float64(tau), ev))
        seg.last_event = ev
        d.n_anchors += 1
        return
    end
    return _absorb!(d, L, tau, R, cfg, ev)
end

"""
    ingest_obs!(st, observations)

Convenience driver: group raw observations by beep event (simulating an
on-device trigger detector that clusters timestamps within 2 s), then feed
the events in order. Missing devices are simply absent — no gap filling.
"""
function ingest_obs!(st::FuserState, observations::AbstractVector{Obs})
    order = sort(observations; by=o -> (o.event, o.device))
    taus = Dict{Int,Float64}()
    cur = -1
    stamps = Dict{Int,Float64}()
    for o in order
        if o.event != cur
            !isempty(stamps) && (taus[cur] = ingest_event!(st, stamps))
            cur = o.event
            stamps = Dict{Int,Float64}()
        end
        stamps[o.device] = o.L_meas
    end
    !isempty(stamps) && (taus[cur] = ingest_event!(st, stamps))
    return taus
end

function _absorb!(d::DeviceState, L::Real, tau::Real, R::Real,
                  cfg::FuserConfig, ev::Int)
    seg = active(d)
    if seg === nothing
        fresh = init_segment(d.n_splits + 1, L, tau)
        fresh.first_event = ev
        fresh.last_event = ev
        fresh.confirmed = true
        push!(d.segments, fresh)
        push!(fresh.anchors, (Float64(L), Float64(tau), ev))
        d.n_anchors += 1
        d.fails = 0
        return
    end
    # An unconfirmed piece is still on trial: an inlier that fits the
    # *previous* piece proves the split was spurious, so roll it back.
    if !seg.confirmed && length(d.segments) >= 2 &&
       length(seg.anchors) <= 2
        prev = d.segments[end - 1]
        d2p, innovp = innovation_test(prev, L, tau, R;
                                      q_a=cfg.q_a, q_b=cfg.q_b)
        if d2p <= cfg.gate_chi2 && abs(innovp) < 0.5 * cfg.hard_jump
            _rollback_split!(d)
            seg = active(d)
        end
    end
    # Once the trial piece has a few anchors, commit to it: absorb every
    # incoming point directly (the KF gate still filters gross errors).
    if !seg.confirmed
        d2, innov = innovation_test(seg, L, tau, R;
                                    q_a=cfg.q_a, q_b=cfg.q_b)
        if d2 <= cfg.gate_chi2 || abs(innov) <= cfg.hard_jump
            _trial_update!(seg, L, tau, ev, cfg)
            d.n_anchors += 1
            seg.last_event = ev
            d.last_good_event = ev
            _maybe_confirm!(d, cfg)
            d.fails = 0
            return
        end
        d.n_rejected += 1
        return
    end
    # Second anchor: close the bootstrap with an exact two-point affine fit,
    # which captures the unknown intercept (boot offset) immediately.
    if length(seg.anchors) == 1
        # Both clocks count physical seconds, so keep unit slope and let the
        # anchor absorb the (unknown) boot offset only.
        seg.b = 1.0
        seg.a = tau - seg.b * (L - seg.L0)
        seg.P = [R*4      0.0;
                 0.0  (3e-4)^2]
        push!(seg.anchors, (Float64(L), Float64(tau), ev))
        seg.last_event = ev
        d.n_anchors += 1
        d.fails = 0
        _maybe_confirm!(d, cfg)
        return
    end
    d2, innov = innovation_test(seg, L, tau, R; q_a=cfg.q_a, q_b=cfg.q_b)
    hard = abs(innov) > cfg.hard_jump
    if d2 <= cfg.gate_chi2 && !hard
        _clear_outlier!(d)
        kf_update!(seg, L, tau, R; q_a=cfg.q_a, q_b=cfg.q_b, ev=ev)
        d.n_anchors += 1
        seg.last_event = ev
        d.fails = 0
        d.last_good_event = ev
        _maybe_confirm!(d, cfg)
        return
    end
    # A reboot jumps the clock by far more than any injected gross error
    # (hundreds of seconds vs <10 s), so one hard innovation splits at once.
    if hard && seg.confirmed
        d.n_rejected += 1
        d.fails += 1
        push!(d.pending, (Float64(L), Float64(tau), ev))
        _split!(d, cfg)
        return
    end
    if !seg.confirmed
        # Keep collecting trial points; do not pile up fail counts that
        # would immediately re-split an unconfirmed piece.
        d.n_rejected += 1
        push!(d.pending, (Float64(L), Float64(tau), ev))
        return
    end
    # Rejected. Isolated gross errors are followed by a fitting point and get
    # discarded; a genuine clock jump keeps rejecting with a consistent sign.
    d.n_rejected += 1
    d.fails += 1
    push!(d.pending, (Float64(L), Float64(tau), ev))
    signs = [round(Int, sign(t - (active(d).a +
                 active(d).b * (Ll - active(d).L0))))
             for (Ll, t, _e) in d.pending]
    consistent = length(signs) >= 2 && all(==(signs[1]), signs)
    # A reboot shifts every subsequent reading, so three consecutive
    # same-sign rejections are required. Isolated gross errors (even large
    # ones) are isolated by the next inlier and never reach this count.
    if d.fails >= 3 && consistent
        _split!(d, cfg)
    end
    return nothing
end

function _maybe_confirm!(d::DeviceState, cfg::FuserConfig)
    seg = active(d)
    if !seg.confirmed && length(seg.anchors) >= cfg.confirm_anchors
        _clean_trial!(seg, cfg)
        seg.confirmed = true
        seg.closed_at = Inf
        prev = length(d.segments) >= 2 ? d.segments[end - 1] : nothing
        prev !== nothing &&
            (prev.closed_at = last(prev.anchors)[1])
        d.n_splits += 1
        empty!(d.pending)
    end
    return nothing
end

# Drop anchors that disagree with the trial piece's own unit-slope model;
# these are boundary gross errors that slipped in around the reboot.
function _clean_trial!(seg::Segment, cfg::FuserConfig)
    length(seg.anchors) <= 3 && return nothing
    L0 = first(seg.anchors)[1]
    resid = [tau - (L - L0) for (L, tau, _e) in seg.anchors]
    med = median(resid)
    mad = median(abs.(resid .- med))
    scale = max(1.4826 * mad, sqrt(cfg.R))
    keep = filter(seg.anchors) do (L, tau, _e)
        abs((tau - (L - L0)) - med) <= 4.0 * scale
    end
    dropped = length(seg.anchors) - length(keep)
    if dropped > 0 && length(keep) >= 2
        seg.anchors = keep
        # Re-estimate intercept at the surviving first anchor.
        L1, tau1, e1 = first(keep)
        seg.L0, seg.tau0 = L1, tau1
        seg.a = median(tau - (L - L1) for (L, tau, _e) in keep)
        seg.b = 1.0
        seg.first_event = e1
    end
    return nothing
end

function _rollback_split!(d::DeviceState)
    trial = pop!(d.segments)
    d.n_anchors -= length(trial.anchors)
    # Trial anchors were rejected against the previous piece; the current
    # inlier (handled by the caller) is the first point after rollback.
    d.outliers += length(trial.anchors)
    d.fails = 0
    empty!(d.pending)
    return nothing
end

# After a successful update, any single buffered rejection was an outlier.
function _clear_outlier!(d::DeviceState)
    if d.fails == 0 && !isempty(d.pending)
        d.outliers += length(d.pending)
        empty!(d.pending)
    end
end

function _split!(d::DeviceState, cfg::FuserConfig)
    # The jump starts at the first rejected anchor whose innovation is both
    # large and same-signed as the confirmed run. Anything before it is a
    # pre-jump gross error and stays out of both fitted pieces.
    queued = splice!(d.pending, 1:length(d.pending))
    seg = active(d)
    scored = map(queued) do (Ll, tl, evl)
        innov = tl - (seg.a + seg.b * (Ll - seg.L0))
        (Ll, tl, evl, innov)
    end
    target_sign = sign(scored[end][4])
    start = findfirst(t -> abs(t[4]) > 0.5 * cfg.hard_jump &&
                           sign(t[4]) == target_sign, scored)
    start === nothing && (start = length(scored))
    kept = [(t[1], t[2], t[3]) for t in scored[start:end]]
    isempty(kept) && (kept = queued[end:end])
    dropped = length(queued) - length(kept)
    d.outliers += dropped
    first_L, first_tau, first_ev = kept[1]
    fresh = init_segment(d.n_splits + 1, first_L, first_tau)
    fresh.first_event = first_ev
    fresh.last_event = first_ev
    fresh.confirmed = false
    fresh.b = 1.0
    push!(d.segments, fresh)
    push!(fresh.anchors, (Float64(first_L), Float64(first_tau), first_ev))
    d.n_anchors += 1
    for (L, tau, ev) in kept[2:end]
        _trial_update!(active(d), L, tau, ev, cfg)
        active(d).last_event = ev
        d.n_anchors += 1
    end
    d.fails = 0
    _maybe_confirm!(d, cfg)
    return nothing
end

# Trial pieces hold unit slope (both clocks count physical seconds); only
# the intercept is adjusted until the piece is confirmed.
function _trial_update!(seg::Segment, L::Real, tau::Real, ev::Int,
                        cfg::FuserConfig)
    seg.b = 1.0
    seg.a = tau - (L - seg.L0)
    push!(seg.anchors, (Float64(L), Float64(tau), ev))
    seg.P[1, 1] = max(seg.P[1, 1], cfg.R * 4)
    return nothing
end

# ---- correction of an arbitrary local timestamp ---------------------------

_span(seg::Segment) =
    isempty(seg.anchors) ? (seg.L0, seg.L0) :
    (first(seg.anchors)[1], last(seg.anchors)[1])

"""
    correct(st, device, L; event=nothing)

Map a local timestamp `L` of one device to consensus time with an honest
uncertainty. When `event` (the beep index) is known, it selects the correct
clock era even if a reboot wrapped local time, so no ambiguity remains.
Without it, overlapping eras return `:ambiguous` with a conservative union
interval; extrapolation beyond the fitted span inflates the variance.
"""
function correct(st::FuserState, device::Int, L::Real; event::Union{Int,Nothing}=nothing)
    d = get(st.devices, device, nothing)
    d === nothing && return CorrectionResult(NaN, Inf, NaN, NaN, :unknown, 0)
    segs = d.segments
    isempty(segs) && return CorrectionResult(NaN, Inf, NaN, NaN, :unknown, 0)

    spans = _span.(segs)
    # Era hint available: pick the piece whose event window contains it.
    if event !== nothing
        candidates = findall(s -> s.first_event <= event <= s.last_event, segs)
        if !isempty(candidates)
            # Boundary events can sit in two windows; the newest era wins.
            return _evaluate(segs[last(candidates)], L, st.cfg, :interpolated)
        end
    end
    # 1) inside one or more fitted spans. A reboot restarts the local clock,
    #    so spans can overlap in local time; multiple matches are ambiguous.
    inside = Segment[]
    for (k, seg) in enumerate(segs)
        lo, hi = spans[k]
        if lo - 1e-9 <= L <= hi + 1e-9
            push!(inside, seg)
        end
    end
    if length(inside) == 1
        return _evaluate(inside[1], L, st.cfg, :interpolated)
    elseif length(inside) > 1
        rs = [_evaluate(s, L, st.cfg, :ambiguous) for s in inside]
        lo = minimum(r.lo for r in rs)
        hi = maximum(r.hi for r in rs)
        mid = 0.5 * (lo + hi)
        return CorrectionResult(mid, 0.5 * (hi - lo), lo, hi, :ambiguous,
                                inside[end].id)
    end
    # 2) gap between consecutive pieces (only when local time did not wrap)
    for k in 1:(length(segs) - 1)
        hi_left  = spans[k][2]
        lo_right = spans[k + 1][1]
        if lo_right > hi_left && hi_left < L < lo_right
            rL = _evaluate(segs[k], L, st.cfg, :ambiguous)
            rR = _evaluate(segs[k + 1], L, st.cfg, :ambiguous)
            mid = 0.5 * (rL.tau + rR.tau)
            lo  = min(rL.lo, rR.lo)
            hi  = max(rL.hi, rR.hi)
            return CorrectionResult(mid, 0.5 * (hi - lo), lo, hi, :ambiguous,
                                    segs[k + 1].id)
        end
    end
    # 3) outside every span -> extrapolation from the nearest piece
    nearest = segs[argmin([min(abs(L - s[1]), abs(L - s[2])) for s in spans])]
    return _evaluate(nearest, L, st.cfg, :extrapolated)
end

function _evaluate(seg::Segment, L::Real, cfg::FuserConfig, status::Symbol)
    dt = L - seg.L0
    H  = [1.0, dt]
    # Extrapolation accumulates process noise over the uncovered distance.
    gap = if status === :interpolated
        0.0
    else
        lo, hi = _span(seg)
        max(lo - L, 0.0, L - hi)
    end
    Q  = [cfg.q_a * gap  0.0;
          0.0           cfg.q_b * max(gap, 1.0)]
    Pv = seg.P + Q
    tau = seg.a + seg.b * dt
    # Floor the uncertainty with the detection-level model error so that
    # over-confident KF covariances never understate real residual risk.
    sigma = max(cfg.sigma_floor, sqrt(dot(H, Pv, H) + cfg.R))
    return CorrectionResult(tau, sigma, tau - 1.96 * sigma,
                            tau + 1.96 * sigma, status, seg.id)
end

"""
    summarize(st)

Per-device fitted-piece counts, anchor counts and rejection statistics.
"""
function summarize(st::FuserState)
    rows = NamedTuple[]
    for id in sort(collect(keys(st.devices)))
        d = st.devices[id]
        push!(rows, (device=id, pieces=length(d.segments),
                     anchors=d.n_anchors, rejected=d.n_rejected,
                     splits=d.n_splits))
    end
    return rows
end

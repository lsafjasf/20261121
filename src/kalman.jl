# Piecewise affine Kalman fit: consensus time tau = a + b*(L - L0).
# A 2-state random-walk model on (intercept, slope) absorbs slow thermal
# drift; large innovations are rejected as outliers / jump candidates.

const GATE_CHI2 = 9.0        # 3-sigma Mahalanobis gate
const HARD_JUMP = 30.0       # seconds: forces a split; above gross-error range

"""
    init_segment(id, L, tau)

Bootstrap a segment from a single anchor. The initial slope is fixed near 1
(local seconds and consensus seconds have the same scale) with a wide prior.
"""
function init_segment(id::Int, L::Real, tau::Real)
    P = [25.0^2  0.0;
         0.0     (3e-4)^2]
    seg = Segment(id, Float64(L), Float64(tau), 0.0, 1.0, P,
                  Tuple{Float64,Float64,Int}[], Inf, 0, 0, false)
    return seg
end

"""
    predict_tau(seg, L; q_a, q_b, R)

Propagate the filter to local time `L` and return
`(tau_pred, innovation_variance)`.
"""
function predict_tau(seg::Segment, L::Real;
                     q_a::Real, q_b::Real, R::Real)
    dt = L - seg.L0
    H  = [1.0, dt]
    # Process noise: intercept variance grows linearly with elapsed local
    # time; slope gets a small independent walk.
    Q  = [q_a*max(dt, 0.0)  0.0;
          0.0              q_b*max(dt, 1.0)]
    Pp = seg.P + Q
    tau_pred = seg.a + seg.b * dt
    S = dot(H, Pp, H) + R
    return tau_pred, S, H, Pp, dt
end

"""
    kf_update!(seg, L, tau, R; q_a, q_b)

Standard Kalman measurement update. Returns the Mahalanobis distance `d2`
of the innovation and the prediction error (before update).
"""
function kf_update!(seg::Segment, L::Real, tau::Real, R::Real;
                    q_a::Real, q_b::Real, ev::Int=0)
    tau_pred, S, H, Pp, dt = predict_tau(seg, L; q_a=q_a, q_b=q_b, R=R)
    innov = tau - tau_pred
    d2 = innov^2 / S
    K  = Pp * H / S
    x  = [seg.a, seg.b] + K * innov
    seg.a, seg.b = x[1], x[2]
    seg.P = (Matrix(I, 2, 2) - K * transpose(H)) * Pp
    # Re-anchor for numerical stability: keep the affine map exact at L.
    _reanchor!(seg, Float64(L), Float64(tau_pred + innov))
    push!(seg.anchors, (Float64(L), Float64(tau), ev))
    seg.last_event = ev
    return d2, innov
end

function _reanchor!(seg::Segment, Lnew::Real, taunew::Real)
    # tau = a + b*(L - L0)  ==>  tau = (a + b*(Lnew-L0)) + b*(L - Lnew)
    new_a = seg.a + seg.b * (Lnew - seg.L0)
    J = [1.0  (Lnew - seg.L0);
         0.0  1.0]
    seg.P = J * seg.P * J'
    seg.L0, seg.tau0 = Lnew, taunew
    seg.a = new_a
    return nothing
end

"""
    innovation_test(seg, L, tau, R; q_a, q_b)

Evaluate a candidate anchor without mutating state. Returns
`(d2, innov, tau_pred, S)`.
"""
function innovation_test(seg::Segment, L::Real, tau::Real, R::Real;
                         q_a::Real, q_b::Real)
    tau_pred, S, H, Pp, dt = predict_tau(seg, L; q_a=q_a, q_b=q_b, R=R)
    innov = tau - tau_pred
    return innov^2 / S, innov, tau_pred, S
end

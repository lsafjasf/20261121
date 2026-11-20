# Breakpoint evidence between adjacent fitted pieces.
#
# For every split we quantify (1) the physical jump in consensus time at the
# boundary, (2) how much better two affine models fit the local anchors than
# one, using residual RMS and BIC, and (3) a Mahalanobis score of the boundary
# mismatch under the left piece's prediction covariance.

struct BreakpointEvidence
    device::Int
    piece::Int                 # id of the new (right) piece
    L_left::Float64
    L_right::Float64
    tau_left::Float64
    tau_right::Float64
    jump_seconds::Float64      # consensus-time discontinuity at the boundary
    slope_change::Float64
    rms_one::Float64
    rms_two::Float64
    bic_one::Float64
    bic_two::Float64
    bic_delta::Float64         # > 0 favors the two-piece model
    mahalanobis::Float64
    status::Symbol             # :strong :moderate :weak
end

function _residual_rms(anchors, a, b, L0)
    isempty(anchors) && return 0.0
    e = [tau - (a + b * (L - L0)) for (L, tau, _ev) in anchors]
    return sqrt(sum(abs2, e) / length(e))
end

"""
    _ols(anchors)

Least-squares affine fit `tau = a + b*(L - L0)` on a set of anchors.
"""
function _ols(anchors::Vector{Tuple{Float64,Float64,Int}})
    n = length(anchors)
    L0 = anchors[1][1]
    x  = [L - L0 for (L, _t, _e) in anchors]
    y  = [tau   for (_L, tau, _e2) in anchors]
    mx, my = mean(x), mean(y)
    sxx = sum((xi - mx)^2 for xi in x)
    b   = sxx < 1e-12 ? 1.0 :
          sum((x[i] - mx) * (y[i] - my) for i in 1:n) / sxx
    a   = my - b * mx
    return L0, a, b
end

"""
    breakpoint_evidence(st::FuserState)

Compute evidence for every split on every device, oldest split first.
"""
function breakpoint_evidence(st::FuserState)
    out = BreakpointEvidence[]
    for id in sort(collect(keys(st.devices)))
        d = st.devices[id]
        length(d.segments) < 2 && continue
        for k in 2:length(d.segments)
            left, right = d.segments[k - 1], d.segments[k]
                (isempty(left.anchors) || isempty(right.anchors)) && continue
            L_left, tau_left, _e1   = last(left.anchors)
            L_right, tau_right, _e2 = first(right.anchors)
                # Breakpoint consensus time: first anchor of the new era.
                tau_boundary = tau_right
            # What the left piece would have predicted for the first
            # post-jump anchor: the discontinuity is the residual there.
            pred_left = left.a + left.b * (L_right - left.L0)
            jump = tau_right - pred_left
            slope_change = right.b - left.b

            combined = vcat(left.anchors, right.anchors)
            L0o, ao, bo = _ols(combined)
            L0l, al, bl = _ols(left.anchors)
            L0r, ar, br = _ols(right.anchors)
            rms_one = _residual_rms(combined, ao, bo, L0o)
            rms_two = sqrt(
                (sum((tau - (al + bl*(L - L0l)))^2 for (L, tau, _e) in left.anchors)
               + sum((tau - (ar + br*(L - L0r)))^2 for (L, tau, _e) in right.anchors))
                / length(combined))

            n = length(combined)
            eps2 = max(rms_two^2, 1e-12)
            bic_one = n * log(max(rms_one^2, 1e-12)) + 2 * log(n)
            bic_two = n * log(eps2) + 4 * log(n)
            bic_delta = bic_one - bic_two

            H = [1.0, L_right - left.L0]
            S = dot(H, left.P, H) + st.cfg.R
            maha = jump / sqrt(max(S, 1e-12))

            status = if abs(maha) > 5.0 && bic_delta > 10.0
                :strong
            elseif abs(maha) > 3.0 && bic_delta > 0.0
                :moderate
            else
                :weak
            end
            push!(out, BreakpointEvidence(
                id, right.id, L_left, L_right, tau_left, tau_right,
                jump, slope_change, rms_one, rms_two,
                bic_one, bic_two, bic_delta, maha, status))
        end
    end
    return out
end

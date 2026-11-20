# Post-split consolidation: merge adjacent pieces that describe the same
# clock era. Two pieces merge when their slopes agree within uncertainty and
# neither sees a significant discontinuity at the shared boundary. This
# removes fragmentation around a single reboot without erasing real jumps.

"""
    merge_pieces!(st; slope_tol=2.5e-4, jump_tol=0.75)

In-place merge of adjacent same-era pieces on every device. Returns the
number of merges performed.
"""
function merge_pieces!(st::FuserState;
                       slope_tol::Real=2.5e-4, jump_tol::Real=0.75)
    n_merge = 0
    for id in sort(collect(keys(st.devices)))
        d = st.devices[id]
        i = 1
        while i < length(d.segments)
            left, right = d.segments[i], d.segments[i + 1]
            if _compatible(left, right, st.cfg;
                           slope_tol=slope_tol, jump_tol=jump_tol)
                d.segments[i] = _merge_two(left, right)
                deleteat!(d.segments, i + 1)
                n_merge += 1
            else
                i += 1
            end
        end
    end
    return n_merge
end

function _compatible(left::Segment, right::Segment, cfg::FuserConfig;
                     slope_tol::Real, jump_tol::Real)
    isempty(left.anchors) || isempty(right.anchors) && return false
    # Short fragments (born from a couple of boundary gross errors) merge
    # whenever their anchors are explained by the neighbouring long piece.
    short, long = length(left.anchors) <= 3 ? (left, right) :
                  length(right.anchors) <= 3 ? (right, left) : (nothing, nothing)
    if short !== nothing
        L0l, al, bl = _ols(long.anchors)
        r = _residuals(short.anchors, al, bl, L0l)
        return median(abs.(r)) < jump_tol
    end
    # Pooled single-model fit. A real reboot leaves a systematic, large
    # residual on the shorter side; same-era fragmentation does not. Judge
    # with the robust median absolute residual of each side.
    anchors = vcat(left.anchors, right.anchors)
    L0, _a, b = _ols(anchors)
    rL = _residuals(left.anchors, _a, b, L0)
    rR = _residuals(right.anchors, _a, b, L0)
    medL = median(abs.(rL))
    medR = median(abs.(rR))
    # Slope agreement between the two independent fits.
    abs(right.b - left.b) > slope_tol && return false
    # Both halves must be explained by one model within detection-level error.
    return medL < jump_tol && medR < jump_tol
end

function _residuals(anchors, a, b, L0)
    return [tau - (a + b * (L - L0)) for (L, tau, _e) in anchors]
end

function _merge_two(left::Segment, right::Segment)
    anchors = vcat(left.anchors, right.anchors)
    sort!(anchors; by=a -> a[3])          # order by event index
    L0, tau0, e0 = first(anchors)
    # Refit a single affine model on the pooled anchors by OLS.
    x = [a[1] - L0 for a in anchors]
    y = [a[2]      for a in anchors]
    mx, my = mean(x), mean(y)
    sxx = sum((xi - mx)^2 for xi in x)
    b = sxx < 1e-12 ? left.b :
        sum((x[k] - mx) * (y[k] - my) for k in eachindex(x)) / sxx
    a = my - b * mx
    resid = y .- (a .+ b .* x)
    s2 = sum(abs2, resid) / max(length(anchors) - 2, 1)
    P = [s2          0.0;
         0.0   (3e-4)^2]
    seg = Segment(left.id, L0, tau0, a, b, P, anchors, Inf,
                  left.first_event, right.last_event, true)
    return seg
end

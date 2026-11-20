# End-to-end demonstration: simulate four drifting, rebooting recorders,
# fuse their beep timestamps online, and print the acceptance evidence.
#
#   julia run.jl [seed]

include(joinpath(@__DIR__, "src", "ForestClocks.jl"))
using .ForestClocks
using Printf

function main(seed::Int=20261121)
    cfg = SimConfig(rng_seed=seed)
    truths, observations, beeps = simulate_forest(cfg)

    println("=== Forest clock fusion (seed=$seed) ===")
    @printf("devices=%d  beeps=%d  observations=%d  detect=%.2f  gross=%.3f\n\n",
            cfg.n_devices, length(beeps), length(observations),
            cfg.detect_prob, cfg.gross_prob)

    for d in truths
        @printf("device %d boots at T=%.1fs, reboots at %s\n",
                d.id, d.boot_time,
                isempty(d.restarts) ? "—" :
                join(("$(round(rb; digits=1))s" for rb in d.restarts), ", "))
    end
    println()

    # Streaming fusion: beeps arrive one at a time, out of order devices
    # tolerated, missed beeps simply absent.
    st = FuserState()
    taus = ingest_obs!(st, observations)
    merge_pieces!(st; jump_tol=1.0)

    println("--- fitted piecewise clocks ---")
    for row in summarize(st)
        println("  device ", row.device,
                ": pieces=", row.pieces,
                " anchors=", row.anchors,
                " gross dropped=", row.rejected,
                " confirmed splits=", row.splits)
    end
    println()

    println("--- breakpoint evidence ---")
    for e in breakpoint_evidence(st)
        @printf("  dev%d jump=%7.2fs  slope Δ=%+.1e  Mahalanobis=%7.1f  ΔBIC=%7.1f  [%s]\n",
                e.device, e.jump_seconds, e.slope_change, e.mahalanobis,
                e.bic_delta, e.status)
    end
    println()

    report = run_acceptance(truths, observations, beeps, st)
    println(report)
    println()

    # Demonstrate honest uncertainty on an unknown / ambiguous timestamp.
    d1 = truths[1]
    if length(d1.eras) >= 2
        rb = d1.restarts[1]
        L_mid = local_clock(d1, rb + 1.0)
        r = correct(st, 1, L_mid)
        println("--- correction without era hint (reboot window) ---")
        println("  ", r)
        r2 = correct(st, 1, L_mid; event=findlast(<(rb + 5.0), beeps))
        println("  with event hint: ", r2)
    end

    ok = report.pass_beep && report.pass_breakpoint && report.pass_coverage
    println("\nOVERALL: ", ok ? "ALL CHECKS PASSED" : "CHECKS FAILED")
    return ok
end

seed = isempty(ARGS) ? 20261121 : parse(Int, ARGS[1])
main(seed)

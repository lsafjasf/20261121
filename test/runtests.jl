using Test
using Statistics
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "ForestClocks.jl"))
using .ForestClocks

@testset "ForestClocks" begin

    @testset "thermal clock is continuous inside an era" begin
        e = Era(1, 0.0, 2.0, 1.0 + 100e-6, 120e-6, 2π / 300.0, 0.7)
        t = 0.0:50.0:1000.0
        readings = local_clock.(Ref(e), t)
        # monotonic and close to unit-rate over a short window
        @test all(diff(readings) .> 40.0)
        @test abs((readings[end] - readings[1]) - (t[end] - t[1])) < 1.0
    end

    @testset "reboot starts a fresh, discontinuous era" begin
        cfg = SimConfig(rng_seed=42)
        truths, obs, _ = simulate_forest(cfg)
        n_eras = mapreduce(d -> length(d.eras), +, truths)
        n_restarts = mapreduce(d -> length(d.restarts), +, truths)
        @test n_eras == length(truths) + n_restarts
        # a rebooted clock jumps to a small reading (local time wraps)
        d = first(x for x in truths if !isempty(x.restarts))
        rb = d.restarts[1]
        before = local_clock(d, rb - 1.0)
        after  = local_clock(d.eras[2], d.eras[2].T0 + 0.5)
        @test after < before
    end

    @testset "ideal clocks fuse with near-zero error" begin
        st = FuserState()
        t = 0.0
        for _ in 1:20
            t += 10.0
            ingest_event!(st, Dict(1 => t + 1.0, 2 => t - 2.0,
                                   3 => t + 0.3, 4 => t - 0.8))
        end
        for id in 1:4
            seg = last(st.devices[id].segments)
            @test abs(seg.b - 1.0) < 1e-6
            @test length(seg.anchors) >= 18
        end
        # event 10 lies inside the fitted window of every device
        r = correct(st, 1, st.devices[1].segments[1].anchors[10][1];
                    event=st.devices[1].segments[1].anchors[10][3])
        @test r.status === :interpolated
        @test r.sigma < 1.0
    end

    @testset "gross outlier is rejected, not split on" begin
        st = FuserState()
        t = 0.0
        for k in 1:15
            t += 10.0
            stamps = Dict(1 => t, 2 => t, 3 => t, 4 => t)
            if k == 8
                stamps[1] = t + 7.0           # isolated gross error, 3 healthy
            end
            ingest_event!(st, stamps)
        end
        @test length(st.devices[1].segments) == 1
        # the single gross timestamp must not create a spurious piece nor
        # bias the fitted unit slope beyond a tight tolerance
        @test abs(last(st.devices[1].segments).b - 1.0) < 5e-3
    end

    @testset "hard reboot forces a new piece with strong evidence" begin
        st = FuserState()
        t = 0.0
        local1 = 0.0
        for k in 1:30
            t += 10.0
            local1 += 10.0
            if k == 16                      # device 1 reboots, clock wraps
                local1 = 20.0
            end
            ingest_event!(st, Dict(1 => local1, 2 => t, 3 => t))
        end
        merge_pieces!(st; jump_tol=1.0)
        @test length(st.devices[1].segments) == 2
        ev = breakpoint_evidence(st)
        @test length(ev) == 1
        @test ev[1].status === :strong
        @test abs(ev[1].jump_seconds) > 50.0
    end

    @testset "unknown and ambiguous timestamps keep uncertainty" begin
        st = FuserState()
        ingest_event!(st, Dict(1 => 10.0, 2 => 10.0))
        ingest_event!(st, Dict(1 => 20.0, 2 => 20.0))
        ingest_event!(st, Dict(1 => 30.0, 2 => 30.0))
        # device never seen
        r = correct(st, 9, 50.0)
        @test r.status === :unknown
        @test isinf(r.sigma)
        # extrapolation inflates uncertainty with distance
        near = correct(st, 1, 31.0)
        far  = correct(st, 1, 300.0)
        @test far.sigma > near.sigma
    end

    @testset "end-to-end acceptance across seeds and stress regimes" begin
        seeds = (1, 7, 42, 20261121, 31337)
        for seed in seeds
            cfg = SimConfig(rng_seed=seed)
            truths, obs, beeps = simulate_forest(cfg)
            st = FuserState()
            ingest_obs!(st, obs)
            merge_pieces!(st; jump_tol=1.0)
            rpt = run_acceptance(truths, obs, beeps, st)
            @test rpt.pass_beep
            @test rpt.pass_breakpoint
            @test rpt.pass_coverage
            @test rpt.beep_p95 < 0.15
            @test rpt.n_spurious == 0
            @test rpt.n_missing == 0
        end

        harsh = SimConfig(rng_seed=22, detect_prob=0.70, gross_prob=0.08)
        truths, obs, beeps = simulate_forest(harsh)
        st = FuserState()
        ingest_obs!(st, obs)
        merge_pieces!(st; jump_tol=1.0)
        rpt = run_acceptance(truths, obs, beeps, st)
        @test rpt.pass_beep && rpt.pass_breakpoint && rpt.pass_coverage
    end
end

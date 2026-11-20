module ForestClocks

using Statistics
using LinearAlgebra
using Random
using Printf

include("types.jl")
include("simulate.jl")
include("kalman.jl")
include("fuser.jl")
include("merging.jl")
include("evidence.jl")
include("acceptance.jl")

export
    Era, DeviceTruth, Obs, SimConfig, simulate_forest, local_clock,
    FuserConfig, FuserState, ingest_event!, ingest_obs!, correct,
    CorrectionResult, Segment, merge_pieces!,
    BreakpointEvidence, breakpoint_evidence, AcceptanceReport, run_acceptance,
    summarize

end # module ForestClocks

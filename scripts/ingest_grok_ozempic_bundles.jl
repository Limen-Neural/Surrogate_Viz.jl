#!/usr/bin/env julia
# ingest_grok_ozempic_bundles.jl
# Walk a directory tree of grok-ozempic validation report bundles, load each one,
# and write normalized CSV tables to the output directory.

using Pkg

# Only take over the active project when run as a script. Activating at load
# time would switch the caller's project out from under them if this file is
# ever `include`d (e.g. by a test).
if abspath(PROGRAM_FILE) == @__FILE__
    Pkg.activate(joinpath(@__DIR__, ".."))
end

import Surrogate_Viz as SV
using CSV
using DataFrames

function usage(io::IO)
    println(io, "Usage: julia --project=. scripts/ingest_grok_ozempic_bundles.jl <input_dir> <output_dir>")
    println(io, "")
    println(io, "  Walks <input_dir> recursively, finds all validation.report.json files,")
    println(io, "  loads each bundle, and writes normalized CSV tables to <output_dir>/.")
end

function main()
    if !isempty(ARGS) && ARGS[1] in ("--help", "-h")
        usage(stdout)
        exit(0)
    end

    if length(ARGS) < 2
        usage(stderr)
        exit(1)
    end

    input_dir = ARGS[1]
    output_dir = ARGS[2]

    if !isdir(input_dir)
        println(stderr, "Error: input directory not found: $(input_dir)")
        exit(1)
    end

    mkpath(output_dir)

    println("Ingesting grok-ozempic bundles from: $(input_dir)")
    println("Writing output to:                   $(output_dir)")

    runs_df, metrics_df, issues_df = SV.normalize_grok_ozempic_dir(input_dir)

    runs_path = joinpath(output_dir, "runs_table.csv")
    metrics_path = joinpath(output_dir, "metrics_table.csv")
    issues_path = joinpath(output_dir, "issues_table.csv")

    CSV.write(runs_path, runs_df)
    CSV.write(metrics_path, metrics_df)
    CSV.write(issues_path, issues_df)

    println("✓ Ingested $(nrow(runs_df)) bundles")
    println("  runs_table.csv:    $(runs_path)")
    println("  metrics_table.csv: $(metrics_path)")
    println("  issues_table.csv:  $(issues_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

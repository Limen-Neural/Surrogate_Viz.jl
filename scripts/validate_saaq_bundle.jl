#!/usr/bin/env julia
# validate_saaq_bundle.jl
# Validate a single corinth-canal run bundle.
# Exit code 0 = valid, non-zero = invalid.

using Pkg

# Only take over the active project when run as a script. Activating at load
# time would switch the caller's project out from under them if this file is
# ever `include`d (e.g. by a test).
if abspath(PROGRAM_FILE) == @__FILE__
    Pkg.activate(joinpath(@__DIR__, ".."))
end

import Surrogate_Viz as SV

function usage(io::IO)
    println(io, "Usage: julia --project=. scripts/validate_saaq_bundle.jl <run_dir>")
end

function main()
    if !isempty(ARGS) && ARGS[1] in ("--help", "-h")
        usage(stdout)
        exit(0)
    end

    if length(ARGS) < 1
        usage(stderr)
        exit(1)
    end

    bundle_dir = ARGS[1]

    if !isdir(bundle_dir)
        println(stderr, "Error: directory not found: $(bundle_dir)")
        exit(1)
    end

    is_valid, errors = SV.validate_saaq_bundle(bundle_dir)

    if is_valid
        println("✓ Bundle validation passed: $(bundle_dir)")
        try
            bundle = SV.load_saaq_bundle(bundle_dir)
            println("  run_id:       $(bundle.manifest.run_id)")
            println("  run_status:  $(bundle.manifest.run_status)")
            println("  model_family: $(bundle.manifest.model_family)")
            println("  saaq_rule:    $(bundle.manifest.saaq_rule)")
            println("  telemetry:    $(bundle.manifest.telemetry_source)")
            println("  ticks_effective: $(bundle.manifest.ticks_effective)")
        catch e
            println(stderr, "Error: bundle passed basic validation but failed to load: $(e)")
            exit(1)
        end
        exit(0)
    else
        println(stderr, "✗ Bundle validation FAILED: $(bundle_dir)")
        for err in errors
            println(stderr, "  - $(err)")
        end
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

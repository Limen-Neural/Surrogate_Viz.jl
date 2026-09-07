# Telemetry provenance — is this run's data measured, or fabricated?
#
# `run_status` (derived from validation_status) says whether a run COMPLETED.
# `telemetry_provenance` (derived from telemetry_source) says whether its
# numbers were MEASURED. These are independent, and conflating them is how a
# synthesised run ends up presented as a measurement: a run can be
# `completed` — so run_status == real, badged green — while corinth-canal
# actually fell back to synthesising its telemetry.

using Test
using DataFrames
using Surrogate_Viz: telemetry_provenance, is_measured_telemetry,
                     load_saaq_bundle, normalize_bundle_to_tables

const FIXTURES = joinpath(@__DIR__, "fixtures", "bundles")

@testset "telemetry_provenance classification" begin
    # The three values corinth-canal actually stamps.
    @test telemetry_provenance("csv_re4_path_tracing_telemetry") == "measured"
    @test telemetry_provenance("csv_anything_at_all") == "measured"
    @test telemetry_provenance("synthetic") == "synthetic"
    @test telemetry_provenance("synthetic_fallback") == "synthetic_fallback"

    # Anything unrecognised must NOT be assumed measured. An unfamiliar value
    # is not evidence of real data — this is the same defaulting mistake that
    # `_status_from_validation` makes by returning `real` for unknown input.
    @test telemetry_provenance("something_new_upstream") == "unknown"
    @test telemetry_provenance("") == "unknown"
    @test telemetry_provenance(nothing) == "unknown"

    # `csv` without the separator must not pass as measured.
    @test telemetry_provenance("csvish") == "unknown"

    @test is_measured_telemetry("csv_re4_path_tracing_telemetry")
    @test !is_measured_telemetry("synthetic")
    @test !is_measured_telemetry("synthetic_fallback")
    @test !is_measured_telemetry("something_new_upstream")
    @test !is_measured_telemetry(nothing)
end

@testset "provenance is independent of run_status" begin
    # The regression this whole change exists to prevent. This fixture is a
    # COMPLETED run (run_status == real) whose telemetry was synthesised.
    # Keying "is this real data?" off run_status reports it as measurement.
    path = joinpath(FIXTURES, "synthetic_fallback_run")
    bundle = load_saaq_bundle(path)

    @test string(bundle.manifest.run_status) == "real"          # completed
    @test bundle.manifest.telemetry_source == "synthetic_fallback"

    # ... and provenance must disagree with it.
    @test telemetry_provenance(bundle.manifest.telemetry_source) == "synthetic_fallback"
    @test !is_measured_telemetry(bundle.manifest.telemetry_source)
end

@testset "normalizer surfaces provenance columns" begin
    runs_df, _, _ = normalize_bundle_to_tables(
        load_saaq_bundle(joinpath(FIXTURES, "synthetic_fallback_run")))

    @test hasproperty(runs_df, :telemetry_provenance)
    @test hasproperty(runs_df, :telemetry_measured)
    @test runs_df.telemetry_provenance[1] == "synthetic_fallback"
    @test runs_df.telemetry_measured[1] === false

    # run_status still reports completion, unchanged — the two columns must
    # carry different answers for this row, which is the entire point.
    @test runs_df.run_status[1] == "real"
    @test runs_df.run_status[1] != runs_df.telemetry_provenance[1]

    # A genuinely measured bundle classifies the other way.
    measured_df, _, _ = normalize_bundle_to_tables(
        load_saaq_bundle(joinpath(FIXTURES, "missing_optional")))
    @test measured_df.telemetry_provenance[1] == "measured"
    @test measured_df.telemetry_measured[1] === true
end

@testset "every fixture bundle carries provenance" begin
    # Guards against a future fixture being added without the column, which
    # would make the dashboard silently fall back to "unknown".
    for dir in readdir(FIXTURES; join = true)
        isdir(dir) || continue
        isfile(joinpath(dir, "run_manifest.json")) || continue
        runs_df, _, _ = normalize_bundle_to_tables(load_saaq_bundle(dir))
        @test hasproperty(runs_df, :telemetry_provenance)
        @test runs_df.telemetry_provenance[1] in
              ("measured", "synthetic", "synthetic_fallback", "unknown")
    end
end

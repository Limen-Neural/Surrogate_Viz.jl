using Test
using CSV
using DataFrames
using Statistics
using TOML

using Surrogate_Viz
using Surrogate_Viz: real, synthetic, skipped, failed
using Surrogate_Viz: load_saaq_bundle, validate_saaq_bundle
using Surrogate_Viz: normalize_bundle_to_tables, normalize_bundles_dir
using Surrogate_Viz: RunManifest, RunMetrics, RunWarning, SaaqBundle, RunStatus
using Surrogate_Viz: CPUBackend, CUDABackend, has_cuda, compute_delta_per_tick
using Surrogate_Viz: pairwise_summary, summarise_run

@testset "Surrogate_Viz module exports" begin
    @test :imported_latent_path in names(Surrogate_Viz)
    @test :load_latent_df in names(Surrogate_Viz)
    @test :detect_delta_column in names(Surrogate_Viz)
    @test :maybe_entropy_column in names(Surrogate_Viz)
    @test :summarise_run in names(Surrogate_Viz)
    @test :pairwise_summary in names(Surrogate_Viz)
    @test :fmt in names(Surrogate_Viz)
    @test :to_float64_vec in names(Surrogate_Viz)
    @test :to_int_ms in names(Surrogate_Viz)
    @test :IMPORT_ROOT in names(Surrogate_Viz)
    @test :pretty_condition in names(Surrogate_Viz)
    @test :pretty_column in names(Surrogate_Viz)
    @test :pretty_model in names(Surrogate_Viz)
    @test :classify_walkers in names(Surrogate_Viz)
    @test :walker_label in names(Surrogate_Viz)
end

@testset "human-readable label registry" begin
    @test pretty_condition("sviz_math_logic") == "Math & Logic"
    @test pretty_condition("unknown_condition") == "unknown_condition"

    @test pretty_column("avg_pop_firing_rate_hz") == "Population Firing Rate (Hz)"
    @test pretty_column(:gpu_temp_c) == "GPU Temperature (°C)"
    @test pretty_column("cpu_tctl_c") == "CPU Temperature (°C)"
    @test pretty_column("elapsed_us") == "Elapsed Time (μs)"
    @test pretty_column("unknown_column") == "unknown_column"

    @test pretty_model("olmoe_1b_7b_f16") == "OLMoE 1B-7B"
    @test pretty_model("unknown_model") == "unknown_model"
end

@testset "walker behavioral labels" begin
    tick_data = DataFrame(best_walker = vcat(fill(10, 50), fill(20, 49), [30]))
    groups = classify_walkers(tick_data; attractor_threshold=0.5, rare_threshold=0.02)
    groups_by_id = Dict(group.id => group for group in groups)

    @test groups_by_id[10].role == :attractor
    @test groups_by_id[10].label == "Attractor #10"
    @test groups_by_id[20].role == :explorer
    @test groups_by_id[20].label == "Explorer #20"
    @test groups_by_id[30].role == :rare
    @test groups_by_id[30].label == "Rare #30"
    @test walker_label(groups, 20) == "Explorer #20"
    @test walker_label(groups, 99) == "Walker #99"

    @test isempty(classify_walkers(DataFrame(best_walker = Int[])))
    @test_throws ErrorException classify_walkers(DataFrame(tick = [1]))
end

@testset "fmt — -0.0 normalization" begin
    @test Surrogate_Viz.fmt(0.0) == "0.0"
    @test Surrogate_Viz.fmt(-0.0) == "0.0"
    @test Surrogate_Viz.fmt(1.234567) == "1.234567"
    @test Surrogate_Viz.fmt(-1.5e-7) == "0.0"
    @test Surrogate_Viz.fmt(missing) == "-"
    @test Surrogate_Viz.fmt("hello") == "hello"
end

@testset "to_float64_vec — skips missing" begin
    @test Surrogate_Viz.to_float64_vec([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    @test Surrogate_Viz.to_float64_vec([1.0, missing, 3.0]) == [1.0, 3.0]
    @test isempty(Surrogate_Viz.to_float64_vec([missing, missing]))
end

@testset "to_int_ms — rounding" begin
    @test Surrogate_Viz.to_int_ms(100.4) == 100
    @test Surrogate_Viz.to_int_ms(100.5) == 100
    @test Surrogate_Viz.to_int_ms(0.0) == 0
end

@testset "detect_delta_column" begin
    v15_df = DataFrame(saaq_delta_q_v15_target = [0.1], timestamp_ms = [0])
    legacy_df = DataFrame(saaq_delta_q_legacy_target = [0.2], timestamp_ms = [0])
    both_df = DataFrame(saaq_delta_q_v15_target = [0.1], saaq_delta_q_legacy_target = [0.2], timestamp_ms = [0])

    @test Surrogate_Viz.detect_delta_column([v15_df]) == :saaq_delta_q_v15_target
    @test Surrogate_Viz.detect_delta_column([both_df]) == :saaq_delta_q_v15_target
    @test Surrogate_Viz.detect_delta_column([legacy_df]) == :saaq_delta_q_legacy_target

    @test_throws ErrorException Surrogate_Viz.detect_delta_column(DataFrame[])

    no_delta_df = DataFrame(timestamp_ms = [0], avg_pop_firing_rate_hz = [1.0])
    @test_throws ErrorException Surrogate_Viz.detect_delta_column([no_delta_df])
end

@testset "maybe_entropy_column" begin
    with_entropy = DataFrame(routing_entropy = [0.5], timestamp_ms = [0])
    without_entropy = DataFrame(timestamp_ms = [0], avg_pop_firing_rate_hz = [1.0])

    @test Surrogate_Viz.maybe_entropy_column([with_entropy]) == :routing_entropy
    @test Surrogate_Viz.maybe_entropy_column([without_entropy]) === nothing
    @test Surrogate_Viz.maybe_entropy_column(DataFrame[]) === nothing
end

@testset "summarise_run — neutral paired-run fixtures" begin
    control_off_df = DataFrame(
        timestamp_ms = [0, 100, 200, 300],
        saaq_delta_q_v15_target = [0.5, 0.6, 0.7, 0.8],
    )

    control_on_df = DataFrame(
        timestamp_ms = [0, 100, 200, 300],
        saaq_delta_q_v15_target = [1.5, 1.6, 1.7, 1.8],
        routing_entropy = [0.1, 0.2, 0.3, 0.4],
    )
    off_run = Dict{String,Any}("id" => "run_001", "condition" => "baseline", "model" => "test_model", "telemetry_source" => "csv_re4_path_tracing_telemetry", "family" => "test")

    on_run = Dict{String,Any}("id" => "run_002", "condition" => "treatment", "model" => "test_model", "telemetry_source" => "csv_re4_path_tracing_telemetry", "family" => "test")

    delta_col = :saaq_delta_q_v15_target
    entropy_col = :routing_entropy

    off_summary = Surrogate_Viz.summarise_run(control_off_df, off_run, delta_col, nothing)
    @test off_summary["rows"] == 4
    @test off_summary["mean_delta"] ≈ 0.65
    @test off_summary["final_delta"] == 0.8
    @test off_summary["mean_entropy"] === missing

    on_summary = Surrogate_Viz.summarise_run(control_on_df, on_run, delta_col, entropy_col)
    @test on_summary["rows"] == 4
    @test on_summary["mean_delta"] ≈ 1.65
    @test on_summary["mean_entropy"] ≈ 0.25
    @test on_summary["final_entropy"] == 0.4
end

@testset "pairwise_summary — neutral paired-run mechanics" begin
    control_off_df = DataFrame(
        timestamp_ms = [0, 100, 200],
        saaq_delta_q_v15_target = [0.5, 0.6, 0.7],
        routing_entropy = [0.1, 0.2, 0.3],
    )

    control_on_df = DataFrame(
        timestamp_ms = [0, 100, 200],
        saaq_delta_q_v15_target = [1.5, 1.6, 1.7],
        routing_entropy = [0.4, 0.5, 0.6],
    )

    delta_col = :saaq_delta_q_v15_target
    entropy_col = :routing_entropy

    joined, summary = Surrogate_Viz.pairwise_summary(control_off_df, control_on_df, delta_col, entropy_col)

    @test nrow(joined) == 3
    @test summary["paired_rows"] == 3
    @test summary["mean_delta_on_minus_off"] ≈ 1.0
    @test summary["max_abs_delta_on_minus_off"] ≈ 1.0
    @test summary["final_delta_on_minus_off"] == 1.0
    @test summary["mean_entropy_on_minus_off"] ≈ 0.3
    @test summary["final_entropy_on_minus_off"] == 0.3
end

@testset "pairwise_summary — missing entropy values" begin
    control_off_df = DataFrame(
        timestamp_ms = [0, 100, 200],
        saaq_delta_q_v15_target = [0.5, 0.6, 0.7],
        routing_entropy = [0.1, missing, 0.3],
    )

    control_on_df = DataFrame(
        timestamp_ms = [0, 100, 200],
        saaq_delta_q_v15_target = [1.5, 1.6, 1.7],
        routing_entropy = [0.4, 0.5, 0.6],
    )

    joined, summary = Surrogate_Viz.pairwise_summary(control_off_df, control_on_df, :saaq_delta_q_v15_target, :routing_entropy)

    @test nrow(joined) == 3
    @test summary["paired_rows"] == 3
    @test summary["mean_delta_on_minus_off"] ≈ 1.0
    @test haskey(summary, "mean_entropy_on_minus_off")
    @test summary["mean_entropy_on_minus_off"] ≈ 0.3
    @test haskey(summary, "final_entropy_on_minus_off")
    @test summary["final_entropy_on_minus_off"] ≈ 0.3
end

@testset "pairwise_summary — no entropy column" begin
    control_off_df = DataFrame(
        timestamp_ms = [0, 100],
        saaq_delta_q_v15_target = [0.5, 0.6],
    )

    control_on_df = DataFrame(
        timestamp_ms = [0, 100],
        saaq_delta_q_v15_target = [1.5, 1.6],
    )

    joined, summary = Surrogate_Viz.pairwise_summary(control_off_df, control_on_df, :saaq_delta_q_v15_target, nothing)

    @test nrow(joined) == 2
    @test summary["paired_rows"] == 2
    @test !haskey(summary, "mean_entropy_on_minus_off")
    @test !haskey(summary, "final_entropy_on_minus_off")
end

@testset "import contract — selected_runs.toml parsing" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "selected_runs.toml")
    @test isfile(fixture_path)

    runs = TOML.parsefile(fixture_path)
    @test haskey(runs, "runs")
    run_list = runs["runs"]
    @test length(run_list) == 2

    off_run = run_list[1]
    on_run = run_list[2]

    @test off_run["condition"] == "baseline"
    @test on_run["condition"] == "treatment"
    @test off_run["model"] == "test_model"
    @test off_run["campaign"] == "test_campaign"
    @test off_run["blessed"] == true
    @test on_run["id"] == "test_on_001"
end

@testset "import contract — local_run_dir path structure" begin
    off_run = Dict{String,Any}(
        "id" => "run_abc",
        "model" => "olmoe-1b-7b",
        "telemetry_source" => "csv_re4_path_tracing_telemetry",
        "condition" => "baseline",
        "family" => "olmoe",
    )

    expected = joinpath(Surrogate_Viz.IMPORT_ROOT, "olmoe-1b-7b", "csv_re4_path_tracing_telemetry", "baseline", "run_abc", "latent_telemetry.csv")
    @test Surrogate_Viz.imported_latent_path(off_run) == expected
end

@testset "import contract — IMPORT_ROOT is normalized" begin
    @test !occursin("..", Surrogate_Viz.IMPORT_ROOT)
    @test !occursin("//", Surrogate_Viz.IMPORT_ROOT)
end

@testset "import contract — path traversal rejection" begin
    traversal_run = Dict{String,Any}(
        "id" => "../etc/passwd",
        "model" => "test_model",
        "telemetry_source" => "csv_re4_path_tracing_telemetry",
        "condition" => "baseline",
    )
    @test_throws ErrorException Surrogate_Viz.imported_latent_path(traversal_run)

    abs_run = Dict{String,Any}(
        "id" => "run_abc",
        "model" => "/etc",
        "telemetry_source" => "csv_re4_path_tracing_telemetry",
        "condition" => "baseline",
    )
    @test_throws ErrorException Surrogate_Viz.imported_latent_path(abs_run)

    backslash_run = Dict{String,Any}(
        "id" => "run_abc",
        "model" => "test_model",
        "telemetry_source" => "csv_re4\\path_tracing_telemetry",
        "condition" => "baseline",
    )
    @test_throws ErrorException Surrogate_Viz.imported_latent_path(backslash_run)

    good_run = Dict{String,Any}(
        "id" => "run_abc",
        "model" => "olmoe-1b-7b",
        "telemetry_source" => "csv_re4_path_tracing_telemetry",
        "condition" => "baseline",
    )
    @test Surrogate_Viz.imported_latent_path(good_run) isa String
end

@testset "SAAQ_latent_discovery — column validation with fixture" begin
    fixture_csv = joinpath(@__DIR__, "fixtures", "latent_telemetry.csv")
    @test isfile(fixture_csv)

    df = CSV.read(fixture_csv, DataFrame)
    required_cols = [:avg_pop_firing_rate_hz, :membrane_dv_dt, :routing_entropy, :saaq_delta_q_prev, :saaq_delta_q_target]
    missing_cols = setdiff(required_cols, propertynames(df))
    @test isempty(missing_cols)
end

@testset "SAAQ_latent_discovery — feature matrix shape" begin
    fixture_csv = joinpath(@__DIR__, "fixtures", "latent_telemetry.csv")
    df = CSV.read(fixture_csv, DataFrame)

    X = Matrix{Float64}(hcat(
        df.avg_pop_firing_rate_hz,
        df.membrane_dv_dt,
        df.routing_entropy,
        df.saaq_delta_q_prev,
    )')

    y = Float64.(df.saaq_delta_q_target)

    @test size(X, 1) == 4
    @test size(X, 2) == 3
    @test length(y) == 3
end

@testset "SAAQ_latent_discovery — run selection by RUN_ID" begin
    fixture_path = joinpath(@__DIR__, "fixtures", "selected_runs.toml")
    runs = TOML.parsefile(fixture_path)
    run_list = runs["runs"]

    off_run = filter(r -> r["condition"] == "baseline", run_list)
    on_run = filter(r -> r["condition"] == "treatment", run_list)

    @test length(off_run) == 1
    @test length(on_run) == 1
    @test off_run[1]["id"] == "test_off_001"
end

@testset "SaaqBundleLoader — successful synthetic bundle" begin
    fixture = joinpath(@__DIR__, "fixtures", "bundles", "successful_synthetic")
    @test isdir(fixture)

    bundle = load_saaq_bundle(fixture)
    @test bundle.manifest.run_id == "test_synthetic_20260528T120000"
    @test bundle.manifest.run_status == synthetic
    @test bundle.manifest.model_family == "Llama3.2"
    @test bundle.manifest.saaq_rule == "SaaqV1_5SqrtRate"
    @test bundle.manifest.telemetry_source == "csv_re4_path_tracing_telemetry"
    @test bundle.metrics.ticks_completed == 512
    @test bundle.metrics.latent_rows == 512
    @test !ismissing(bundle.metrics.mean_tick_elapsed_us)
    @test bundle.metrics.repeat_determinism == "matched"
end

@testset "SaaqBundleLoader — skipped run bundle" begin
    fixture = joinpath(@__DIR__, "fixtures", "bundles", "skipped_run")
    bundle = load_saaq_bundle(fixture)
    @test bundle.manifest.run_id == "test_skipped_20260528T120001"
    @test bundle.manifest.run_status == skipped
    @test bundle.manifest.validation_status == "skipped"
    @test bundle.metrics.ticks_completed == 0
    @test bundle.metrics.latent_rows == 0
end

@testset "SaaqBundleLoader — failed run bundle" begin
    fixture = joinpath(@__DIR__, "fixtures", "bundles", "failed_run")
    bundle = load_saaq_bundle(fixture)
    @test bundle.manifest.run_id == "test_failed_20260528T120002"
    @test bundle.manifest.run_status == failed
    @test bundle.manifest.validation_status == "failed"
    @test bundle.manifest.error !== nothing
    @test !isempty(bundle.manifest.error)
    @test bundle.metrics.ticks_completed == 0
end

@testset "SaaqBundleLoader — missing optional fields" begin
    fixture = joinpath(@__DIR__, "fixtures", "bundles", "missing_optional")
    bundle = load_saaq_bundle(fixture)
    @test bundle.manifest.run_id == "test_missing_optional_20260528T120003"
    @test bundle.manifest.model_family == "Llama3.2"
    @test bundle.manifest.saaq_rule == "SaaqV1_5SqrtRate"
    @test bundle.metrics.ticks_completed == 512
    @test ismissing(bundle.metrics.latent_rows)
    @test ismissing(bundle.metrics.mean_tick_elapsed_us)
end

@testset "SaaqBundleLoader — unknown extra fields preserved" begin
    fixture = joinpath(@__DIR__, "fixtures", "bundles", "unknown_extras")
    bundle = load_saaq_bundle(fixture)
    @test bundle.manifest.run_id == "test_unknown_extras_20260528T120004"
    @test haskey(bundle.manifest.extra, "experimental_custom_field")
    @test bundle.manifest.extra["experimental_custom_field"] == 42
    @test haskey(bundle.manifest.extra, "research_alpha")
    @test bundle.manifest.extra["research_alpha"] == 3.14159
    @test haskey(bundle.manifest.extra, "future_extension")
    @test bundle.metrics.extra isa Dict
    @test haskey(bundle.metrics.extra, "custom_metric_qos_score")
    @test bundle.metrics.extra["custom_metric_qos_score"] == 0.987
end

@testset "SaaqBundleLoader — synthetic smokescreen bundle" begin
    fixture = joinpath(@__DIR__, "fixtures", "bundles", "synthetic_smokescreen")
    @test isdir(fixture)

    bundle = load_saaq_bundle(fixture)
    @test bundle.manifest.run_id == "test_smokescreen_20260528T130000"
    @test bundle.manifest.run_status == synthetic
    @test bundle.manifest.model_slug == "olmoe-1b-7b"
    @test bundle.manifest.model_family == "olmoe"
    @test isnothing(bundle.manifest.model_descriptor) || !startswith(bundle.manifest.model_descriptor, "[synthetic]")
    @test bundle.manifest.repeat_idx == 1
    @test bundle.manifest.repeat_count == 3
    @test bundle.manifest.saaq_rule == "SaaqV1_5SqrtRate"
    @test bundle.metrics.ticks_completed == 1000
    @test bundle.metrics.latent_rows == 1000
    @test bundle.metrics.mean_tick_elapsed_us ≈ 207.3
    @test bundle.metrics.repeat_determinism == "matched"
end

@testset "SaaqBundleLoader — validate_saaq_bundle" begin
    valid_fixture = joinpath(@__DIR__, "fixtures", "bundles", "successful_synthetic")
    is_valid, errors = validate_saaq_bundle(valid_fixture)
    @test is_valid
    @test isempty(errors)

    invalid_fixture = joinpath(@__DIR__, "fixtures", "bundles", "missing_optional")
    is_valid2, errors2 = validate_saaq_bundle(invalid_fixture)
    @test is_valid2
    @test isempty(errors2)

    nonexistent = joinpath(@__DIR__, "fixtures", "bundles", "nonexistent_bundle")
    is_valid3, errors3 = validate_saaq_bundle(nonexistent)
    @test !is_valid3
    @test length(errors3) > 0
end

@testset "SaaqNormalizer — normalize_bundle_to_tables" begin
    fixture = joinpath(@__DIR__, "fixtures", "bundles", "successful_synthetic")
    bundle = load_saaq_bundle(fixture)
    runs_df, metrics_df, warnings_df = normalize_bundle_to_tables(bundle)

    @test nrow(runs_df) == 1
    @test runs_df[1, :run_id] == "test_synthetic_20260528T120000"
    @test runs_df[1, :run_status] == "synthetic"
    @test runs_df[1, :model_family] == "Llama3.2"
    @test runs_df[1, :saaq_formula_version] == "SaaqV1_5SqrtRate"

    @test nrow(metrics_df) >= 4
    metric_names = metrics_df.metric_name
    @test "ticks_completed" in metric_names
    @test "latent_rows" in metric_names
    @test "mean_tick_elapsed_us" in metric_names

    @test nrow(warnings_df) == 0
end

@testset "SaaqNormalizer — normalize_bundles_dir smoke" begin
    fixture_root = joinpath(@__DIR__, "fixtures", "bundles")
    runs_df, metrics_df, warnings_df = normalize_bundles_dir(fixture_root)

    @test nrow(runs_df) == 6
    status_vals = Set(runs_df.run_status)
    @test "real" in status_vals
    @test "synthetic" in status_vals
    @test "skipped" in status_vals
    @test "failed" in status_vals

    run_ids = Set(runs_df.run_id)
    @test "test_synthetic_20260528T120000" in run_ids
    @test "test_skipped_20260528T120001" in run_ids
    @test "test_failed_20260528T120002" in run_ids
    @test "test_missing_optional_20260528T120003" in run_ids
    @test "test_unknown_extras_20260528T120004" in run_ids
    @test "test_smokescreen_20260528T130000" in run_ids

    @test nrow(metrics_df) > 0

    @test nrow(warnings_df) == 0
end

@testset "ComputeBackend types" begin
    @test Surrogate_Viz.CPUBackend() isa Surrogate_Viz.ComputeBackend
    @test Surrogate_Viz.CUDABackend() isa Surrogate_Viz.ComputeBackend
    @test Surrogate_Viz.CPUBackend() !== Surrogate_Viz.CUDABackend()
    @test Surrogate_Viz.has_cuda() isa Bool
end

@testset "compute_delta_per_tick — CPUBackend" begin
    features = Float32[1 2 3 4;
                        5 6 7 8;
                        9 10 11 12]
    result = compute_delta_per_tick(features, CPUBackend())
    @test size(result) == (2, 4)
    @test result[1, 1] == 4.0f0
    @test result[2, 4] == 4.0f0
    @test result == diff(features, dims=1)
end

@testset "compute_delta_per_tick — with timestamps" begin
    ts = [0, 100, 200, 300]
    features = Float32[1 2 3 4;
                        5 6 7 8;
                        9 10 11 12;
                        13 14 15 16]
    t_out, deltas = compute_delta_per_tick(ts, features, CPUBackend())
    @test t_out == [100, 200, 300]
    @test size(deltas) == (3, 4)
    @test deltas[1, 1] == 4.0f0
end

@testset "pairwise_summary — CPUBackend no-regression" begin
    off_df = DataFrame(
        timestamp_ms = [0, 100, 200],
        saaq_delta_q_v15_target = [0.5, 0.6, 0.7],
        routing_entropy = [0.1, 0.2, 0.3],
    )
    on_df = DataFrame(
        timestamp_ms = [0, 100, 200],
        saaq_delta_q_v15_target = [1.5, 1.6, 1.7],
        routing_entropy = [0.4, 0.5, 0.6],
    )
    delta_col = :saaq_delta_q_v15_target
    entropy_col = :routing_entropy

    joined, summary = pairwise_summary(off_df, on_df, delta_col, entropy_col; backend=CPUBackend())

    @test nrow(joined) == 3
    @test summary["paired_rows"] == 3
    @test summary["mean_delta_on_minus_off"] ≈ 1.0
    @test summary["max_abs_delta_on_minus_off"] ≈ 1.0
    @test summary["final_delta_on_minus_off"] == 1.0
    @test summary["mean_entropy_on_minus_off"] ≈ 0.3
    @test summary["final_entropy_on_minus_off"] == 0.3
end

@testset "summarise_run — CPUBackend no-regression" begin
    df = DataFrame(
        timestamp_ms = [0, 100, 200, 300],
        saaq_delta_q_v15_target = [0.5, 0.6, 0.7, 0.8],
        routing_entropy = [0.1, 0.2, 0.3, 0.4],
    )
    run = Dict{String,Any}("id" => "run_001", "condition" => "baseline")

    row = summarise_run(df, run, :saaq_delta_q_v15_target, :routing_entropy; backend=CPUBackend())

    @test row["run_id"] == "run_001"
    @test row["rows"] == 4
    @test row["mean_delta"] ≈ 0.65
    @test row["max_delta"] == 0.8
    @test row["final_delta"] == 0.8
    @test row["mean_entropy"] ≈ 0.25
    @test row["final_entropy"] == 0.4
end

@testset "CUDA path (if available)" begin
    if has_cuda()
        off_df = DataFrame(
            timestamp_ms = [0, 100, 200],
            saaq_delta_q_v15_target = Float32[0.5, 0.6, 0.7],
        )
        on_df = DataFrame(
            timestamp_ms = [0, 100, 200],
            saaq_delta_q_v15_target = Float32[1.5, 1.6, 1.7],
        )

        joined_cpu, summary_cpu = pairwise_summary(off_df, on_df, :saaq_delta_q_v15_target, nothing; backend=CPUBackend())
        joined_gpu, summary_gpu = pairwise_summary(off_df, on_df, :saaq_delta_q_v15_target, nothing; backend=CUDABackend())

        @test summary_cpu["paired_rows"] == summary_gpu["paired_rows"]
        @test abs(summary_cpu["mean_delta_on_minus_off"] - summary_gpu["mean_delta_on_minus_off"]) < 1e-5
        @test abs(summary_cpu["max_abs_delta_on_minus_off"] - summary_gpu["max_abs_delta_on_minus_off"]) < 1e-5
        @test abs(summary_cpu["final_delta_on_minus_off"] - summary_gpu["final_delta_on_minus_off"]) < 1e-5

        features = Float32[1 2 3 4;
                          5 6 7 8;
                          9 10 11 12]
        cpu_result = compute_delta_per_tick(features, CPUBackend())
        gpu_result = compute_delta_per_tick(features, CUDABackend())
        @test cpu_result ≈ gpu_result
    else
        @test has_cuda() == false
        @info "CUDA not functional on this runner — skipping CUDA-specific tests"
    end
end

@testset "CUDABackend falls back to CPU when CUDA is absent" begin
    if !has_cuda()
        @test_logs (
            :warn,
            "CUDABackend requested but CUDA unavailable; using CPUBackend",
        ) (
            :warn,
            "CUDABackend requested but CUDA unavailable; using CPUBackend",
        ) (
            :warn,
            "CUDABackend requested but CUDA unavailable; using CPUBackend",
        ) begin
            off_df = DataFrame(
                timestamp_ms = [0, 100, 200],
                saaq_delta_q_v15_target = [0.5, 0.6, 0.7],
                routing_entropy = [0.1, 0.2, 0.3],
            )
            on_df = DataFrame(
                timestamp_ms = [0, 100, 200],
                saaq_delta_q_v15_target = [1.5, 1.6, 1.7],
                routing_entropy = [0.4, 0.5, 0.6],
            )

            joined_cpu, summary_cpu = pairwise_summary(
                off_df, on_df, :saaq_delta_q_v15_target, :routing_entropy; backend=CPUBackend()
            )
            joined_gpu, summary_gpu = pairwise_summary(
                off_df, on_df, :saaq_delta_q_v15_target, :routing_entropy; backend=CUDABackend()
            )
            @test summary_cpu == summary_gpu

            df = DataFrame(
                timestamp_ms = [0, 100, 200, 300],
                saaq_delta_q_v15_target = [0.5, 0.6, 0.7, 0.8],
                routing_entropy = [0.1, 0.2, 0.3, 0.4],
            )
            run = Dict{String,Any}("id" => "run_001", "condition" => "baseline")
            row_cpu = summarise_run(
                df, run, :saaq_delta_q_v15_target, :routing_entropy; backend=CPUBackend()
            )
            row_gpu = summarise_run(
                df, run, :saaq_delta_q_v15_target, :routing_entropy; backend=CUDABackend()
            )
            @test row_cpu == row_gpu

            features = Float32[1 2 3 4;
                              5 6 7 8;
                              9 10 11 12]
            cpu_result = compute_delta_per_tick(features, CPUBackend())
            gpu_result = compute_delta_per_tick(features, CUDABackend())
            @test cpu_result == gpu_result
        end
    else
        @test has_cuda() == true
    end
end

using Surrogate_Viz: GrokOzempicFailure, GrokOzempicWarning, GrokOzempicReport, GrokOzempicBundle
using Surrogate_Viz: load_grok_ozempic_bundle, validate_grok_ozempic_bundle
using Surrogate_Viz: normalize_grok_ozempic_to_tables, normalize_grok_ozempic_dir, normalize_grok_ozempic_bundle_to_tables

@testset "GrokOzempic — passing bundle" begin
    fixture = joinpath(@__DIR__, "fixtures", "grok_ozempic", "pass")
    bundle = load_grok_ozempic_bundle(fixture)
    @test bundle.report.status == "PASS"
    @test bundle.report.source_tensor_count == 770
    @test bundle.report.artifact_tensor_count == 770
    @test bundle.report.router_count == 64
    @test bundle.report.protected_router_violations == 0
    @test bundle.report.byte_accounting_result == "match"
    @test isempty(bundle.report.failures)
    @test isempty(bundle.report.warnings)
end

@testset "GrokOzempic — failing bundle with failures and warnings" begin
    fixture = joinpath(@__DIR__, "fixtures", "grok_ozempic", "fail")
    bundle = load_grok_ozempic_bundle(fixture)
    @test bundle.report.status == "FAIL"
    @test bundle.report.source_tensor_count == 770
    @test bundle.report.artifact_tensor_count == 768
    @test bundle.report.protected_router_violations == 2
    @test bundle.report.byte_accounting_result == "mismatch"
    @test length(bundle.report.failures) == 3
    @test bundle.report.failures[1].category == "missing_tensor"
    @test bundle.report.failures[1].tensor == "blk.0.ffn_gate_inp.weight"
    @test bundle.report.failures[2].category == "router_policy_violation"
    @test bundle.report.failures[3].category == "shape_mismatch"
    @test length(bundle.report.warnings) == 2
    @test bundle.report.warnings[1].category == "unresolved_expert_projection"
end

@testset "GrokOzempic — warnings-only bundle" begin
    fixture = joinpath(@__DIR__, "fixtures", "grok_ozempic", "warnings_only")
    bundle = load_grok_ozempic_bundle(fixture)
    @test bundle.report.status == "PASS"
    @test isempty(bundle.report.failures)
    @test length(bundle.report.warnings) == 2
    @test bundle.report.unknown_unresolved_warning_count == 2
end

@testset "GrokOzempic — missing optional fields" begin
    fixture = joinpath(@__DIR__, "fixtures", "grok_ozempic", "missing_optional")
    bundle = load_grok_ozempic_bundle(fixture)
    @test bundle.report.status == "PASS"
    @test bundle.report.source_tensor_count == 0
    @test bundle.report.source_total_bytes == 0
    @test isempty(bundle.report.failures)
    @test isempty(bundle.report.warnings)
    @test haskey(bundle.report.extra, "artifact_tensor_count") == false
end

@testset "GrokOzempic — validate_grok_ozempic_bundle" begin
    valid_fixture = joinpath(@__DIR__, "fixtures", "grok_ozempic", "pass")
    is_valid, errors = validate_grok_ozempic_bundle(valid_fixture)
    @test is_valid
    @test isempty(errors)

    nonexistent = joinpath(@__DIR__, "fixtures", "grok_ozempic", "nonexistent")
    is_valid2, errors2 = validate_grok_ozempic_bundle(nonexistent)
    @test !is_valid2
    @test length(errors2) > 0
end

@testset "GrokOzempicNormalizer — normalize_grok_ozempic_bundle_to_tables" begin
    fixture = joinpath(@__DIR__, "fixtures", "grok_ozempic", "fail")
    runs_df, metrics_df, issues_df = normalize_grok_ozempic_bundle_to_tables(fixture)

    @test nrow(runs_df) == 1
    @test runs_df[1, :status] == "FAIL"
    @test runs_df[1, :source_tensor_count] == 770
    @test runs_df[1, :protected_router_violations] == 2

    @test nrow(metrics_df) == 6
    metric_names = metrics_df.metric_name
    @test "source_tensor_count" in metric_names
    @test "router_count" in metric_names
    @test "artifact_total_bytes" in metric_names

    @test nrow(issues_df) == 5
    @test sum(issues_df.severity .== "failure") == 3
    @test sum(issues_df.severity .== "warning") == 2
    @test "issue_category" in names(issues_df)
end

@testset "GrokOzempicNormalizer — normalize_grok_ozempic_dir smoke" begin
    fixture_root = joinpath(@__DIR__, "fixtures", "grok_ozempic")
    runs_df, metrics_df, issues_df = normalize_grok_ozempic_dir(fixture_root)

    @test nrow(runs_df) == 4
    status_vals = Set(runs_df.status)
    @test "PASS" in status_vals
    @test "FAIL" in status_vals
    @test nrow(metrics_df) > 0
    @test nrow(issues_df) > 0
end

include("smoke_scripts_test.jl")

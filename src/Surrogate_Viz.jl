# SurrogateViz — shared utilities for SAAQ comparison scripts

module Surrogate_Viz

using CSV
using DataFrames
using JSON: JSON, parse
using Statistics

const IMPORT_ROOT = normpath(joinpath(@__DIR__, "..", "data", "corinth_runs"))

include("backend.jl")
include("kernels.jl")

function validate_path_component(name::AbstractString, value::AbstractString)
    occursin("..", value) && error("Invalid $(name) path component (contains '..'): $(value)")
    (startswith(value, "/") || occursin("\\", value)) && error("Invalid $(name) path component (absolute or contains backslash): $(value)")
    return value
end

function imported_latent_path(run::Dict{String,<:Any})
    model = validate_path_component("model", run["model"])
    src = validate_path_component("telemetry_source", run["telemetry_source"])
    cond = validate_path_component("condition", run["condition"])
    id = validate_path_component("id", run["id"])
    joinpath(IMPORT_ROOT, model, src, cond, id, "latent_telemetry.csv")
end

function load_latent_df(run::Dict{String,<:Any})
    path = imported_latent_path(run)
    isfile(path) || error("Imported latent telemetry not found at $(path). Run `julia import_corinth_runs.jl` first.")
    df = CSV.read(path, DataFrame)
    "timestamp_ms" in names(df) || error("Missing timestamp_ms column in $(path)")
    sort!(df, :timestamp_ms)
    return df
end

function detect_delta_column(frames::Vector{DataFrame})
    isempty(frames) && error("detect_delta_column: no frames provided")
    common_names = intersect((Set(String.(names(df))) for df in frames)...)
    "saaq_delta_q_v15_target" in common_names && return :saaq_delta_q_v15_target

    best_name = nothing
    best_score = -typemax(Int)
    for name in common_names
        lower_name = lowercase(name)
        occursin("saaq", lower_name) || continue
        occursin("delta", lower_name) || continue
        occursin("target", lower_name) || continue

        score = 0
        occursin("v15", lower_name) && (score += 100)
        occursin("legacy", lower_name) && (score -= 25)
        occursin("delta_q", lower_name) && (score += 10)
        startswith(lower_name, "saaq_delta_q") && (score += 5)

        if score > best_score
            best_score = score
            best_name = Symbol(name)
        end
    end

    best_name === nothing && error("Could not find a SAAQ target delta column in imported latent telemetry")
    return best_name
end

function maybe_entropy_column(frames::Vector{DataFrame})
    isempty(frames) && return nothing
    common_names = intersect((Set(String.(names(df))) for df in frames)...)
    return "routing_entropy" in common_names ? :routing_entropy : nothing
end

to_float64_vec(col) = Float64[Float64(v) for v in skipmissing(col)]
to_int_ms(v) = Int(round(Float64(v)))

# Grok Build 0.1 model: re-export the new pure-Julia CUDA visual kernels
# (walker density histogram etc.) for use by plot_latent_space.jl and related
# when CUDA is available on the target (RTX 5080 / sm_120). These are the
# kernels added for the combined #43 visuals + #44 runner PR. They are 100%
# Julia (CUDA.jl) — corinth-canal inspiration is read-only.
export walker_density_bins_and_counts  # safe public API; uses the CUDA-backed path when CUDA is installed and functional

function summarise_run(df::DataFrame, run::Dict{String,<:Any}, delta_col::Symbol, entropy_col::Union{Nothing,Symbol};
    backend::ComputeBackend=CPUBackend())
    nrow(df) > 0 || error("summarise_run: empty DataFrame for run id=$(run["id"]) condition=$(run["condition"])")

    delta_values = to_float64_vec(df[!, delta_col])
    isempty(delta_values) && error("summarise_run: column $(delta_col) is empty after dropping missing values for run $(run["id"])")

    entropy_values = entropy_col !== nothing ? to_float64_vec(df[!, entropy_col]) : nothing
    stats = compute_run_stats(delta_values, entropy_values, backend)

    row = Dict{String,Any}(
        "run_id" => run["id"],
        "condition" => run["condition"],
        "rows" => nrow(df),
        "first_ms" => to_int_ms(first(df[!, :timestamp_ms])),
        "last_ms" => to_int_ms(last(df[!, :timestamp_ms])),
        "mean_delta" => stats.mean_delta,
        "max_delta" => stats.max_delta,
        "final_delta" => stats.final_delta,
    )

    row["mean_entropy"] = stats.mean_entropy
    row["final_entropy"] = stats.final_entropy

    return row
end

function pairwise_summary(off_df::DataFrame, on_df::DataFrame, delta_col::Symbol, entropy_col::Union{Nothing,Symbol};
    backend::ComputeBackend=CPUBackend())
    joined = innerjoin(
        select(off_df, :timestamp_ms, delta_col => :delta_off, (entropy_col === nothing ? [] : [entropy_col => :entropy_off])...),
        select(on_df, :timestamp_ms, delta_col => :delta_on, (entropy_col === nothing ? [] : [entropy_col => :entropy_on])...),
        on=:timestamp_ms,
    )

    nrow(joined) > 0 || error("No overlapping timestamps between the provided runs")
    joined.delta_on_minus_off = joined.delta_on .- joined.delta_off

    delta_stats = compute_pairwise_deltas(joined.delta_off, joined.delta_on, backend)
    summary = Dict{String,Any}(
        "paired_rows" => nrow(joined),
        "mean_delta_on_minus_off" => delta_stats.mean_delta,
        "max_abs_delta_on_minus_off" => delta_stats.max_abs_delta,
        "final_delta_on_minus_off" => delta_stats.final_delta,
    )

    if entropy_col !== nothing
        joined.entropy_on_minus_off = joined.entropy_on .- joined.entropy_off
        emv = to_float64_vec(joined.entropy_on_minus_off)
        if isempty(emv)
            summary["mean_entropy_on_minus_off"] = missing
            summary["final_entropy_on_minus_off"] = missing
        else
            summary["mean_entropy_on_minus_off"] = Statistics.mean(emv)
            summary["final_entropy_on_minus_off"] = last(emv)
        end
    end

    return joined, summary
end

fmt(x::Missing) = "-"
function fmt(x::Real)
    rounded = round(Float64(x); digits=6)
    rounded == 0.0 && (rounded = 0.0)
    return string(rounded)
end
fmt(x) = string(x)

@enum RunStatus real synthetic skipped failed

Base.@kwdef struct RunManifest
    run_id::String
    run_status::RunStatus = real
    model_family::String
    model_slug::String = ""
    model_descriptor::String = ""
    architecture::String = ""
    checkpoint_format::String = ""
    prompt_profile::String = ""
    saaq_rule::String = ""
    saaq_dual_emit::Bool = false
    telemetry_source::String = ""
    routing_mode::String = ""
    run_tag::String = ""
    repeat_idx::Int = 0
    repeat_count::Int = 1
    validation_status::String = "unknown"
    error::Union{Nothing,String} = nothing
    ticks::Int = 0
    ticks_effective::Int = 0
    extra::Dict{String,Any} = Dict{String,Any}()
end

Base.@kwdef struct RunMetrics
    ticks_completed::Union{Int,Missing} = missing
    latent_rows::Union{Int,Missing} = missing
    mean_tick_elapsed_us::Union{Float64,Missing} = missing
    first_timestamp_ms::Union{Int,Missing} = missing
    last_timestamp_ms::Union{Int,Missing} = missing
    repeat_determinism::Union{String,Missing} = missing
    extra::Dict{String,Any} = Dict{String,Any}()
end

Base.@kwdef struct RunWarning
    category::String = ""
    message::String = ""
    tensor_name::Union{String,Nothing} = nothing
end

Base.@kwdef struct SaaqBundle
    manifest::RunManifest
    metrics::RunMetrics
    warnings::Vector{RunWarning} = RunWarning[]
end

KNOWN_MANIFEST_FIELDS = [
    :run_id, :model_slug, :model_family, :architecture, :checkpoint_path,
    :routing_tensor_name, :synapse_source, :checkpoint_format, :prompt_embedding_source,
    :prompt_profile, :prompt_text, :ticks, :saaq_rule, :saaq_primary_rule,
    :saaq_dual_emit, :validation_status, :error, :telemetry_source, :telemetry_csv_path,
    :telemetry_row_count, :wraparound_enabled, :wraparound_loops,
    :ticks_effective, :run_dir, :output_root, :repeat_idx,
    :repeat_count, :cwd_routing_csv_contaminated, :run_tag, :routing_mode,
    :generated_files, :created_at, :repo, :commit_sha,
]

KNOWN_METRICS_FIELDS = [
    :ticks_completed, :latent_rows, :mean_tick_elapsed_us,
    :first_timestamp_ms, :last_timestamp_ms, :repeat_determinism,
]

function _status_from_validation(v::Union{Nothing,String}, error::Union{Nothing,String})
    if v === nothing
        return error !== nothing && !isempty(error) ? failed : real
    elseif error !== nothing && !isempty(error) && v ∈ ("completed", "success")
        return failed
    elseif v ∈ ("completed", "success")
        return real
    elseif v ∈ ("synthetic",)
        return synthetic
    elseif v ∈ ("skipped", "skip")
        return skipped
    elseif v ∈ ("failed", "error", "crash")
        return failed
    elseif error !== nothing && !isempty(error)
        return failed
    else
        return real
    end
end

function load_saaq_bundle(path::AbstractString)::SaaqBundle
    manifest_path = joinpath(path, "run_manifest.json")
    summary_path = joinpath(path, "summary.json")

    if !isfile(manifest_path)
        error("Bundle validation failed: run_manifest.json not found at $(manifest_path)")
    end

    raw_manifest = parse(read(manifest_path, String))

    extra_manifest = Dict{String,Any}()
    for (k, v) in raw_manifest
        Symbol(k) in KNOWN_MANIFEST_FIELDS || (extra_manifest[k] = v)
    end

    validation_status = get(raw_manifest, "validation_status", "unknown")
    run_error = get(raw_manifest, "error", nothing)
    run_status = _status_from_validation(validation_status, run_error)

    manifest = RunManifest(
        run_id=get(raw_manifest, "run_id", ""),
        run_status=run_status,
        model_family=get(raw_manifest, "model_family", ""),
        model_slug=get(raw_manifest, "model_slug", ""),
        model_descriptor=get(raw_manifest, "checkpoint_path", ""),
        architecture=get(raw_manifest, "architecture", ""),
        checkpoint_format=get(raw_manifest, "checkpoint_format", ""),
        prompt_profile=get(raw_manifest, "prompt_profile", ""),
        saaq_rule=get(raw_manifest, "saaq_rule", ""),
        saaq_dual_emit=get(raw_manifest, "saaq_dual_emit", false),
        telemetry_source=get(raw_manifest, "telemetry_source", ""),
        routing_mode=get(raw_manifest, "routing_mode", ""),
        run_tag=get(raw_manifest, "run_tag", ""),
        repeat_idx=get(raw_manifest, "repeat_idx", 0),
        repeat_count=get(raw_manifest, "repeat_count", 1),
        validation_status=validation_status,
        error=run_error,
        ticks=get(raw_manifest, "ticks", 0),
        ticks_effective=get(raw_manifest, "ticks_effective", 0),
        extra=extra_manifest,
    )

    metrics = RunMetrics()
    warnings = RunWarning[]

    if isfile(summary_path)
        raw_summary = parse(read(summary_path, String))

        extra_metrics = Dict{String,Any}()
        metrics_inner = get(raw_summary, "metrics", raw_summary)
        for (k, v) in metrics_inner
            Symbol(k) in KNOWN_METRICS_FIELDS || (extra_metrics[k] = v)
        end

        raw_metrics = get(raw_summary, "metrics", Dict{String,Any}())
        metrics = RunMetrics(
            ticks_completed=get(raw_metrics, "ticks_completed", missing),
            latent_rows=get(raw_metrics, "latent_rows", missing),
            mean_tick_elapsed_us=get(raw_metrics, "mean_tick_elapsed_us", missing),
            first_timestamp_ms=get(raw_metrics, "first_timestamp_ms", missing),
            last_timestamp_ms=get(raw_metrics, "last_timestamp_ms", missing),
            repeat_determinism=get(raw_metrics, "repeat_determinism", missing),
            extra=extra_metrics,
        )
    end

    return SaaqBundle(manifest, metrics, warnings)
end

function validate_saaq_bundle(path::AbstractString)::Tuple{Bool,Vector{String}}
    errors = String[]
    manifest_path = joinpath(path, "run_manifest.json")
    if !isfile(manifest_path)
        push!(errors, "run_manifest.json not found at $(manifest_path)")
        return false, errors
    end
    try
        raw = parse(read(manifest_path, String))
        if !haskey(raw, "run_id")
            push!(errors, "run_manifest.json missing required field: run_id")
        end
        if !haskey(raw, "run_dir")
            push!(errors, "run_manifest.json missing required field: run_dir")
        end
    catch e
        push!(errors, "run_manifest.json is not valid JSON: $(e)")
    end
    return isempty(errors), errors
end

function compute_delta_per_tick end

function _nothing_to_missing(v::Union{Nothing,T}) where T
    v === nothing ? missing : v
end

include("grok_ozempic.jl")

include(joinpath(@__DIR__, "normalizers", "SaaqNormalizer.jl"))
const _saaq_normalizer = getfield(@__MODULE__, :SaaqNormalizer)
const normalize_bundle_to_tables = getfield(_saaq_normalizer, :normalize_bundle_to_tables)
const normalize_bundles_dir = getfield(_saaq_normalizer, :normalize_bundles_dir)
const normalize_saaq_bundle_to_tables = getfield(_saaq_normalizer, :normalize_saaq_bundle_to_tables)

include(joinpath(@__DIR__, "normalizers", "GrokOzempicNormalizer.jl"))
const _grok_normalizer = getfield(@__MODULE__, :GrokOzempicNormalizer)
const normalize_grok_ozempic_to_tables = getfield(_grok_normalizer, :normalize_grok_ozempic_to_tables)
const normalize_grok_ozempic_dir = getfield(_grok_normalizer, :normalize_grok_ozempic_dir)
const normalize_grok_ozempic_bundle_to_tables = getfield(_grok_normalizer, :normalize_grok_ozempic_bundle_to_tables)

include("labels.jl")

export imported_latent_path, load_latent_df, detect_delta_column, maybe_entropy_column
export to_float64_vec, to_int_ms, summarise_run, pairwise_summary, fmt
export validate_path_component, IMPORT_ROOT
export RunStatus, RunManifest, RunMetrics, RunWarning, SaaqBundle
export real, synthetic, skipped, failed
export load_saaq_bundle, validate_saaq_bundle
export normalize_bundle_to_tables, normalize_bundles_dir, normalize_saaq_bundle_to_tables
export ComputeBackend, CPUBackend, CUDABackend, has_cuda, compute_delta_per_tick
export GrokOzempicFailure, GrokOzempicWarning, GrokOzempicReport, GrokOzempicBundle
export load_grok_ozempic_bundle, validate_grok_ozempic_bundle
export normalize_grok_ozempic_to_tables, normalize_grok_ozempic_dir, normalize_grok_ozempic_bundle_to_tables
export pretty_condition, pretty_column, pretty_model
export KNOWN_MODEL_SLUGS, known_model_slugs_consistent
export classify_walkers, walker_label, WalkerGroup

end # module Surrogate_Viz

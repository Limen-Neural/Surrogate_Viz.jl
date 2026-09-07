# SaaqNormalizer — corinth-canal bundle → normalized DataFrame tables

module SaaqNormalizer

using DataFrames

const _parent_mod = parentmodule(@__MODULE__)
const _load_saaq_bundle = getfield(_parent_mod, :load_saaq_bundle)
const _nothing_to_missing = getfield(_parent_mod, :_nothing_to_missing)
const _saaq_bundle_type = getfield(_parent_mod, :SaaqBundle)
const _telemetry_provenance = getfield(_parent_mod, :telemetry_provenance)
const _is_measured_telemetry = getfield(_parent_mod, :is_measured_telemetry)

"""
    normalize_saaq_bundle_to_tables(bundle_path::AbstractString) -> (DataFrame, DataFrame, DataFrame)

Load a bundle from `bundle_path` and return three normalized DataFrames:
`(runs_df, metrics_df, warnings_df)`.

This is a convenience wrapper around `load_saaq_bundle` + `normalize_bundle_to_tables`.
"""
function normalize_saaq_bundle_to_tables(bundle_path::AbstractString)::Tuple{DataFrame, DataFrame, DataFrame}
    bundle = _load_saaq_bundle(bundle_path)
    return normalize_bundle_to_tables(bundle)
end

"""
    normalize_bundle_to_tables(bundle::SaaqBundle) -> (DataFrame, DataFrame, DataFrame)

Convert a single `SaaqBundle` into three normalized DataFrames:
- `runs_df`: one row per run, with all manifest fields
- `metrics_df`: one row per metric, with run_id foreign key
- `warnings_df`: one row per warning, with run_id foreign key

Unknown manifest fields are preserved as `extra_<field>` columns.
Missing optional metrics are represented as `missing`.
"""
function normalize_bundle_to_tables(bundle::_saaq_bundle_type)::Tuple{DataFrame, DataFrame, DataFrame}
    m = bundle.manifest
    metrics_row = bundle.metrics

    run_status_str = string(m.run_status)

    runs_row = Dict{String,Any}(
        "run_id" => m.run_id,
        "run_status" => run_status_str,
        "model_family" => m.model_family,
        "model_slug" => m.model_slug,
        "model_descriptor" => _nothing_to_missing(m.model_descriptor),
        "architecture" => m.architecture,
        "checkpoint_format" => m.checkpoint_format,
        "prompt_profile" => m.prompt_profile,
        "saaq_formula_version" => m.saaq_rule,
        "saaq_dual_emit" => m.saaq_dual_emit,
        "telemetry_source" => m.telemetry_source,
        # Derived, and deliberately separate from run_status: a run can be
        # `completed` (run_status == real) while its telemetry was synthesised.
        # Downstream consumers must key "is this real data?" off provenance.
        "telemetry_provenance" => _telemetry_provenance(m.telemetry_source),
        "telemetry_measured" => _is_measured_telemetry(m.telemetry_source),
        "routing_mode" => m.routing_mode,
        "run_tag" => m.run_tag,
        "repeat_idx" => m.repeat_idx,
        "repeat_count" => m.repeat_count,
        "validation_status" => m.validation_status,
        "error" => _nothing_to_missing(m.error),
        "ticks" => m.ticks,
        "ticks_effective" => m.ticks_effective,
    )

    if !isempty(m.extra)
        for (k, v) in m.extra
            clean_v = v === nothing ? missing :
                      v isa AbstractArray ? Any[vi === nothing ? missing : vi for vi in v] :
                      v
            runs_row["extra_$(k)"] = clean_v
        end
    end

    runs_df = DataFrame([runs_row])

    metrics_rows = Dict{String,Any}[]
    for (k, v) in metrics_row.extra
        push!(metrics_rows, Dict{String,Any}(
            "run_id" => m.run_id,
            "metric_name" => k,
            "metric_value" => v === nothing ? missing : v,
            "metric_category" => "extra",
        ))
    end

    if !ismissing(metrics_row.ticks_completed)
        push!(metrics_rows, Dict{String,Any}(
            "run_id" => m.run_id, "metric_name" => "ticks_completed",
            "metric_value" => metrics_row.ticks_completed, "metric_category" => "runtime"))
    end
    if !ismissing(metrics_row.latent_rows)
        push!(metrics_rows, Dict{String,Any}(
            "run_id" => m.run_id, "metric_name" => "latent_rows",
            "metric_value" => metrics_row.latent_rows, "metric_category" => "runtime"))
    end
    if !ismissing(metrics_row.mean_tick_elapsed_us)
        push!(metrics_rows, Dict{String,Any}(
            "run_id" => m.run_id, "metric_name" => "mean_tick_elapsed_us",
            "metric_value" => metrics_row.mean_tick_elapsed_us, "metric_category" => "runtime"))
    end
    if !ismissing(metrics_row.first_timestamp_ms)
        push!(metrics_rows, Dict{String,Any}(
            "run_id" => m.run_id, "metric_name" => "first_timestamp_ms",
            "metric_value" => metrics_row.first_timestamp_ms, "metric_category" => "runtime"))
    end
    if !ismissing(metrics_row.last_timestamp_ms)
        push!(metrics_rows, Dict{String,Any}(
            "run_id" => m.run_id, "metric_name" => "last_timestamp_ms",
            "metric_value" => metrics_row.last_timestamp_ms, "metric_category" => "runtime"))
    end
    if !ismissing(metrics_row.repeat_determinism)
        push!(metrics_rows, Dict{String,Any}(
            "run_id" => m.run_id, "metric_name" => "repeat_determinism",
            "metric_value" => metrics_row.repeat_determinism, "metric_category" => "quality"))
    end

    metrics_df = isempty(metrics_rows) ? DataFrame(run_id=String[], metric_name=String[], metric_value=Any[], metric_category=String[]) : DataFrame(metrics_rows)

    warnings_rows = Dict{String,Any}[
        Dict{String,Any}("run_id" => m.run_id, "warning_category" => w.category,
         "warning_message" => w.message, "tensor_name" => _nothing_to_missing(w.tensor_name), "severity" => "info")
        for w in bundle.warnings
    ]
    warnings_df = isempty(warnings_rows) ? DataFrame(run_id=String[], warning_category=String[], warning_message=String[], tensor_name=Union{String,Missing}[], severity=String[]) : DataFrame(warnings_rows)

    return runs_df, metrics_df, warnings_df
end

"""
    normalize_bundles_dir(input_dir::AbstractString) -> (DataFrame, DataFrame, DataFrame)

Batch-normalize all bundles under `input_dir` into unified DataFrames.

Deduplicates by `run_id`: if the same run_id appears in multiple bundles,
only the last-loaded bundle's data is retained.
"""
function normalize_bundles_dir(input_dir::AbstractString)::Tuple{DataFrame, DataFrame, DataFrame}
    runs_dfs = DataFrame[]
    metrics_dfs = DataFrame[]
    warnings_dfs = DataFrame[]
    bundle_seq = 0

    if !isdir(input_dir)
        error("Input directory not found: $(input_dir)")
    end

    for (root, dirs, files) in walkdir(input_dir)
        if "run_manifest.json" in files
            bundle_path = root
            try
                bundle = _load_saaq_bundle(bundle_path)
                runs_df, metrics_df, warnings_df = normalize_bundle_to_tables(bundle)
                bundle_seq += 1
                runs_df._bundle_seq = fill(bundle_seq, nrow(runs_df))
                metrics_df._bundle_seq = fill(bundle_seq, nrow(metrics_df))
                warnings_df._bundle_seq = fill(bundle_seq, nrow(warnings_df))
                push!(runs_dfs, runs_df)
                push!(metrics_dfs, metrics_df)
                push!(warnings_dfs, warnings_df)
            catch e
                @warn "Failed to load bundle at $(bundle_path): $(e)"
            end
        end
    end

    if isempty(runs_dfs)
        all_runs = DataFrame(run_id=String[], run_status=String[])
        all_metrics = DataFrame(run_id=String[], metric_name=String[], metric_value=Any[], metric_category=String[])
        all_warnings = DataFrame(run_id=String[], warning_category=String[], warning_message=String[], tensor_name=Union{String,Missing}[], severity=String[])
        return all_runs, all_metrics, all_warnings
    end

    all_runs = vcat(runs_dfs..., cols=:union)
    all_metrics = vcat(metrics_dfs..., cols=:union)
    all_warnings = vcat(warnings_dfs..., cols=:union)

    # Deduplicate: keep last occurrence per run_id
    all_runs = unique(all_runs, :run_id, keep=:last)
    
    all_metrics = semijoin(all_metrics, all_runs, on=[:run_id, :_bundle_seq])
    all_warnings = semijoin(all_warnings, all_runs, on=[:run_id, :_bundle_seq])
    
    # Drop the temporary _bundle_seq column
    select!(all_runs, Not(:_bundle_seq))
    select!(all_metrics, Not(:_bundle_seq))
    select!(all_warnings, Not(:_bundle_seq))

    return all_runs, all_metrics, all_warnings
end

export normalize_bundle_to_tables, normalize_bundles_dir, normalize_saaq_bundle_to_tables

end # module SaaqNormalizer

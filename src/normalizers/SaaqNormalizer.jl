# SaaqNormalizer — corinth-canal bundle → normalized DataFrame tables

module SaaqNormalizer

using DataFrames

const _parent_mod = parentmodule(@__MODULE__)
const _load_saaq_bundle = getfield(_parent_mod, :load_saaq_bundle)
const _nothing_to_missing = getfield(_parent_mod, :_nothing_to_missing)
const _saaq_bundle_type = getfield(_parent_mod, :SaaqBundle)

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
    LOAD_FAILURE_CATEGORY

`warning_category` for a bundle that could not be read at all, as distinct from
a bundle that loaded and reported warnings about itself.
"""
const LOAD_FAILURE_CATEGORY = "bundle_load_failure"

"""
    normalize_bundles_dir(input_dir::AbstractString) -> (DataFrame, DataFrame, DataFrame)

Batch-normalize all bundles under `input_dir` into unified DataFrames.

Deduplicates by `run_id`: if the same run_id appears in multiple bundles,
only the last-loaded bundle's data is retained.

A bundle that fails to load is recorded in the warnings table with
`warning_category = "bundle_load_failure"` and `severity = "load_error"`,
carrying its `bundle_path`. It previously vanished from all three tables after
a `@warn`, so callers reported success over a silently smaller corpus.
"""
function normalize_bundles_dir(input_dir::AbstractString)::Tuple{DataFrame, DataFrame, DataFrame}
    runs_dfs = DataFrame[]
    metrics_dfs = DataFrame[]
    warnings_dfs = DataFrame[]
    load_failures = Dict{String,Any}[]
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
                # A bundle that fails to load used to be dropped from all three
                # tables, leaving only a line on stderr. Callers then reported
                # success over a silently smaller corpus — the ingest CLI prints
                # "Ingested N runs" from nrow(runs_df), which counts only what
                # loaded, and exits 0.
                push!(load_failures, Dict{String,Any}(
                    # A bundle that failed to load has no run_id — reading it is
                    # what failed. bundle_path is the only identifier available.
                    "run_id" => "",
                    "bundle_path" => bundle_path,
                    "warning_category" => LOAD_FAILURE_CATEGORY,
                    "warning_message" => sprint(showerror, e),
                    "tensor_name" => missing,
                    "severity" => "load_error",
                ))
            end
        end
    end

    failures_df = isempty(load_failures) ? nothing : DataFrame(load_failures)

    if isempty(runs_dfs)
        all_runs = DataFrame(run_id=String[], run_status=String[])
        all_metrics = DataFrame(run_id=String[], metric_name=String[], metric_value=Any[], metric_category=String[])
        all_warnings = DataFrame(run_id=String[], warning_category=String[], warning_message=String[], tensor_name=Union{String,Missing}[], severity=String[])
        # runs can be empty while failures are not — a directory where every
        # bundle failed to load. Returning bare empty frames here would make
        # that indistinguishable from an empty directory.
        failures_df === nothing || (all_warnings = vcat(all_warnings, failures_df; cols=:union))
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

    # Appended only here, after the semijoin above and after _bundle_seq is
    # dropped. A load-failure row has no corresponding run — that is the whole
    # point — so the semijoin on run_id would filter it straight back out,
    # silently reintroducing the bug this is fixing.
    failures_df === nothing || (all_warnings = vcat(all_warnings, failures_df; cols=:union))

    return all_runs, all_metrics, all_warnings
end

export normalize_bundle_to_tables, normalize_bundles_dir, normalize_saaq_bundle_to_tables

end # module SaaqNormalizer

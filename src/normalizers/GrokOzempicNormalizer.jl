module GrokOzempicNormalizer

using DataFrames

const _parent_mod = parentmodule(@__MODULE__)
const _load_grok_ozempic_bundle = getfield(_parent_mod, :load_grok_ozempic_bundle)
const _nothing_to_missing = getfield(_parent_mod, :_nothing_to_missing)
const _grok_bundle_type = getfield(_parent_mod, :GrokOzempicBundle)

function normalize_grok_ozempic_bundle_to_tables(bundle_path::AbstractString)::Tuple{DataFrame, DataFrame, DataFrame}
    bundle = _load_grok_ozempic_bundle(bundle_path)
    return normalize_grok_ozempic_to_tables(bundle)
end

function normalize_grok_ozempic_to_tables(bundle::_grok_bundle_type)::Tuple{DataFrame, DataFrame, DataFrame}
    r = bundle.report

    runs_row = Dict{String,Any}(
        "bundle_path" => bundle.bundle_path,
        "status" => r.status,
        "source_tensor_count" => r.source_tensor_count,
        "artifact_tensor_count" => r.artifact_tensor_count,
        "router_count" => r.router_count,
        "protected_router_violations" => r.protected_router_violations,
        "protected_norm_violations" => r.protected_norm_violations,
        "expert_association_count" => r.expert_association_count,
        "unknown_unresolved_warning_count" => r.unknown_unresolved_warning_count,
        "checksum_coverage" => r.checksum_coverage,
        "source_total_bytes" => r.source_total_bytes,
        "artifact_total_bytes" => r.artifact_total_bytes,
        "byte_accounting_result" => r.byte_accounting_result,
    )

    if !isempty(r.extra)
        for (k, v) in r.extra
            clean_v = v === nothing ? missing :
                      v isa AbstractArray ? Any[vi === nothing ? missing : vi for vi in v] :
                      v
            runs_row["extra_$(k)"] = clean_v
        end
    end

    runs_df = DataFrame([runs_row])

    metrics_rows = Dict{String,Any}[
        Dict{String,Any}("bundle_path" => bundle.bundle_path, "metric_name" => "source_tensor_count", "metric_value" => r.source_tensor_count, "metric_category" => "artifact"),
        Dict{String,Any}("bundle_path" => bundle.bundle_path, "metric_name" => "artifact_tensor_count", "metric_value" => r.artifact_tensor_count, "metric_category" => "artifact"),
        Dict{String,Any}("bundle_path" => bundle.bundle_path, "metric_name" => "router_count", "metric_value" => r.router_count, "metric_category" => "routing"),
        Dict{String,Any}("bundle_path" => bundle.bundle_path, "metric_name" => "expert_association_count", "metric_value" => r.expert_association_count, "metric_category" => "expert"),
        Dict{String,Any}("bundle_path" => bundle.bundle_path, "metric_name" => "source_total_bytes", "metric_value" => r.source_total_bytes, "metric_category" => "byte_accounting"),
        Dict{String,Any}("bundle_path" => bundle.bundle_path, "metric_name" => "artifact_total_bytes", "metric_value" => r.artifact_total_bytes, "metric_category" => "byte_accounting"),
    ]
    metrics_df = DataFrame(metrics_rows)

    failures_rows = Dict{String,Any}[
        Dict{String,Any}(
            "bundle_path" => bundle.bundle_path,
            "issue_category" => f.category,
            "tensor" => _nothing_to_missing(f.tensor),
            "message" => f.message,
            "severity" => "failure",
        ) for f in r.failures
    ]
    warnings_rows = Dict{String,Any}[
        Dict{String,Any}(
            "bundle_path" => bundle.bundle_path,
            "issue_category" => w.category,
            "tensor" => _nothing_to_missing(w.tensor),
            "message" => w.message,
            "severity" => "warning",
        ) for w in r.warnings
    ]

    all_issues = vcat(failures_rows, warnings_rows)
    issues_df = isempty(all_issues) ?
        DataFrame(bundle_path=String[], issue_category=String[], tensor=Union{String,Missing}[], message=String[], severity=String[]) :
        DataFrame(all_issues)

    return runs_df, metrics_df, issues_df
end

"""
    normalize_grok_ozempic_dir(input_dir; strict=false) -> (runs_df, metrics_df, issues_df)

Batch-normalize every grok-ozempic bundle under `input_dir`.

A bundle that fails to load is skipped. Previously that skip left no trace
beyond a single `@warn` per bundle, so a caller comparing "bundles on disk" to
"rows returned" had no way to tell a partial ingest from a complete one — the
returned tables look the same either way. A summary is now always emitted when
anything failed, naming the count and the paths.

Pass `strict=true` to turn any load failure into an error instead. Use it
wherever a silently short table would be worse than a loud stop, such as an
ingest whose output is published.
"""
function normalize_grok_ozempic_dir(input_dir::AbstractString; strict::Bool=false)::Tuple{DataFrame, DataFrame, DataFrame}
    runs_dfs = DataFrame[]
    metrics_dfs = DataFrame[]
    issues_dfs = DataFrame[]
    failed_paths = String[]
    found = 0

    if !isdir(input_dir)
        error("Input directory not found: $(input_dir)")
    end

    for (root, dirs, files) in walkdir(input_dir)
        if "validation.report.json" in files
            bundle_path = root
            found += 1
            try
                runs_df, metrics_df, issues_df = normalize_grok_ozempic_bundle_to_tables(bundle_path)
                push!(runs_dfs, runs_df)
                push!(metrics_dfs, metrics_df)
                push!(issues_dfs, issues_df)
            catch e
                push!(failed_paths, bundle_path)
                @warn "Failed to load grok-ozempic bundle at $(bundle_path): $(e)"
            end
        end
    end

    if !isempty(failed_paths)
        summary = "normalize_grok_ozempic_dir: $(length(failed_paths)) of $(found) bundle(s) under " *
                  "$(input_dir) failed to load and were skipped:\n  " *
                  join(failed_paths, "\n  ")
        strict && error(summary * "\n(strict=true)")
        @warn summary
    end

    if isempty(runs_dfs)
        all_runs = DataFrame(bundle_path=String[], status=String[])
        all_metrics = DataFrame(bundle_path=String[], metric_name=String[], metric_value=Any[], metric_category=String[])
        all_issues = DataFrame(bundle_path=String[], issue_category=String[], tensor=Union{String,Missing}[], message=String[], severity=String[])
        return all_runs, all_metrics, all_issues
    end

    return reduce(vcat, runs_dfs; cols=:union), reduce(vcat, metrics_dfs; cols=:union), reduce(vcat, issues_dfs; cols=:union)
end

export normalize_grok_ozempic_to_tables, normalize_grok_ozempic_dir, normalize_grok_ozempic_bundle_to_tables

end # module GrokOzempicNormalizer

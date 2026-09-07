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

function normalize_grok_ozempic_dir(input_dir::AbstractString)::Tuple{DataFrame, DataFrame, DataFrame}
    runs_dfs = DataFrame[]
    metrics_dfs = DataFrame[]
    issues_dfs = DataFrame[]

    if !isdir(input_dir)
        error("Input directory not found: $(input_dir)")
    end

    n_found = 0
    n_failed = 0

    for (root, dirs, files) in walkdir(input_dir)
        if "validation.report.json" in files
            bundle_path = root
            n_found += 1
            try
                runs_df, metrics_df, issues_df = normalize_grok_ozempic_bundle_to_tables(bundle_path)
                push!(runs_dfs, runs_df)
                push!(metrics_dfs, metrics_df)
                push!(issues_dfs, issues_df)
            catch e
                n_failed += 1
                @warn "Failed to load grok-ozempic bundle at $(bundle_path): $(e)"
                # A bundle that fails to load used to be dropped from all three
                # tables, leaving no trace in the output. Callers then reported
                # success over a silently smaller corpus: the ingest CLI prints
                # "Ingested N bundles" from nrow(runs_df), which counts only
                # what loaded, and exits 0.
                #
                # Record it as an issue row instead. The issues table already
                # carries a severity column, so this reaches issues_table.csv
                # through the existing path with no caller change.
                push!(issues_dfs, DataFrame([Dict{String,Any}(
                    "bundle_path" => bundle_path,
                    "issue_category" => "bundle_load_failure",
                    "tensor" => missing,
                    "message" => sprint(showerror, e),
                    "severity" => "load_error",
                )]))
            end
        end
    end

    if n_failed > 0
        @warn "grok-ozempic ingest incomplete: $(n_failed) of $(n_found) bundles failed to load; " *
              "they are recorded in the issues table with severity=\"load_error\""
    end

    all_issues = isempty(issues_dfs) ?
        DataFrame(bundle_path=String[], issue_category=String[], tensor=Union{String,Missing}[], message=String[], severity=String[]) :
        reduce(vcat, issues_dfs; cols=:union)

    if isempty(runs_dfs)
        # Match the populated schema. This previously returned a 2-column frame
        # while the populated path returns 13+, so any consumer touching e.g.
        # runs_df.source_tensor_count threw only on the empty-corpus case — the
        # one most likely to happen silently (a mistyped input_dir, or a
        # directory where every bundle failed to load).
        all_runs = DataFrame(
            bundle_path = String[],
            status = String[],
            source_tensor_count = Union{Int,Missing}[],
            artifact_tensor_count = Union{Int,Missing}[],
            router_count = Union{Int,Missing}[],
            protected_router_violations = Union{Int,Missing}[],
            protected_norm_violations = Union{Int,Missing}[],
            expert_association_count = Union{Int,Missing}[],
            unknown_unresolved_warning_count = Union{Int,Missing}[],
            checksum_coverage = Union{Float64,Missing}[],
            source_total_bytes = Union{Int,Missing}[],
            artifact_total_bytes = Union{Int,Missing}[],
            byte_accounting_result = Union{String,Missing}[],
        )
        all_metrics = DataFrame(bundle_path=String[], metric_name=String[], metric_value=Any[], metric_category=String[])
        # Note: issues may be non-empty even when runs is — a directory where
        # every bundle failed to load produces load_error rows and no runs.
        return all_runs, all_metrics, all_issues
    end

    return reduce(vcat, runs_dfs; cols=:union), reduce(vcat, metrics_dfs; cols=:union), all_issues
end

export normalize_grok_ozempic_to_tables, normalize_grok_ozempic_dir, normalize_grok_ozempic_bundle_to_tables

end # module GrokOzempicNormalizer

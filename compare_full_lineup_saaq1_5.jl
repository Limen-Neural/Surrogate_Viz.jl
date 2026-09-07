using Pkg
Pkg.activate(@__DIR__)

using CSV
using DataFrames
using Dates
using TOML
include(joinpath(@__DIR__, "src", "Surrogate_Viz.jl"))
const SV = getfield(Main, :Surrogate_Viz)

# NOTE (Grok Build): This script was originally for historical "full_lineup" heartbeat on/off
# paired runs (control-off / control-on). Current sviz_* experiments use prompt conditions
# instead. If no matching runs are found it will error early to avoid producing empty PNGs.

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
ENV["QT_QPA_PLATFORM"] = get(ENV, "QT_QPA_PLATFORM", "offscreen")
using Plots

const REPO_ROOT = @__DIR__
const SELECTED_RUNS_PATH = joinpath(REPO_ROOT, "data", "selected_runs.toml")
const OUTPUT_DIR = joinpath(REPO_ROOT, "outputs", "full_lineup")
const REPORT_PATH = joinpath(OUTPUT_DIR, "full_lineup_saaq1_5_comparison.md")

const CAMPAIGN = "full_lineup"
const TELEMETRY_SOURCE = "csv_re4_path_tracing_telemetry"
const RULE = "SaaqV1_5SqrtRate"
const EQUATION_SOURCE = "outputs/20260414_194227_v2pNMk/pareto_front.csv"
const REPEAT_IDX = 0

# Sourced from the label registry rather than retyped. The roster was renamed
# once already (GH#40/MET-115, dropping placeholder GGUF slugs such as
# "olmoe_baseline" and "qwen3_moe_i1_iq3_m"); keeping one declaration means the
# next rename is a single edit. This script remains legacy/historical for old
# full_lineup data only (see #38); it will error on current sviz_* runs.
const MODEL_ORDER = SV.KNOWN_MODEL_SLUGS

function load_selected_runs(path::AbstractString)
    manifest = TOML.parsefile(path)
    runs = get(manifest, "runs", nothing)
    runs isa Vector || error("Expected [[runs]] entries in $(path)")
    return runs
end

function full_lineup_runs(all_runs::AbstractVector)
    return filter(all_runs) do run
        get(run, "campaign", nothing) == CAMPAIGN &&
        run["telemetry_source"] == TELEMETRY_SOURCE &&
        run["rule"] == RULE &&
        Int(run["repeat_idx"]) == REPEAT_IDX &&
        get(run, "blessed", false) == true
    end
end

function model_pair(runs::AbstractVector, model_slug::AbstractString)
    model_runs = filter(run -> run["model"] == model_slug, runs)
    off_run = only(filter(run -> run["condition"] == "baseline", model_runs))
    on_run = only(filter(run -> run["condition"] == "treatment", model_runs))
    return off_run, on_run
end

function validate_pairs(runs::AbstractVector, models::AbstractVector{<:AbstractString})
    available = String[]
    problems = String[]
    for slug in models
        off_matches = filter(run -> run["model"] == slug && run["condition"] == "baseline", runs)
        on_matches = filter(run -> run["model"] == slug && run["condition"] == "treatment", runs)

        if length(off_matches) == 0 && length(on_matches) == 0
            continue
        end
        if length(off_matches) == 0
            push!(problems, "missing run: model=$(slug) (off=0, on=$(length(on_matches)))")
            continue
        end
        if length(on_matches) == 0
            push!(problems, "missing run: model=$(slug) (off=$(length(off_matches)), on=0)")
            continue
        end

        if length(off_matches) > 1
            push!(problems, "duplicate baseline runs: model=$(slug) ids=[$(join((m["id"] for m in off_matches), ", "))]")
        end
        if length(on_matches) > 1
            push!(problems, "duplicate treatment runs: model=$(slug) ids=[$(join((m["id"] for m in on_matches), ", "))]")
        end

        if length(off_matches) == 1 && length(on_matches) == 1
            push!(available, slug)
        end
    end

    for p in problems
        @warn "Model validation: $p"
    end

    isempty(available) && error(
        "compare_full_lineup_saaq1_5.jl: no models with complete (off+on) runs found. " *
        "Filter: campaign=$(CAMPAIGN), repeat_idx=$(REPEAT_IDX), " *
        "telemetry_source=$(TELEMETRY_SOURCE), rule=$(RULE). " *
        "Run `import_corinth_runs.jl` first.",
    )

    return available
end

function build_plot(model_slug::AbstractString, off_df::DataFrame, on_df::DataFrame, joined_df::DataFrame, delta_col::Symbol, entropy_col::Union{Nothing,Symbol})
    panel_attrs = (fontfamily = "Helvetica", legend = :topright, lw = 2.0)
    fig_size = (1400, entropy_col === nothing ? 900 : 1200)

    p1 = plot(
        off_df.timestamp_ms,
        off_df[!, delta_col];
        label = SV.pretty_condition("baseline"),
        color = :navy,
        xlabel = SV.pretty_column("timestamp_ms"),
        ylabel = SV.pretty_column(delta_col),
        title = "$(SV.pretty_model(model_slug)) — SAAQ 1.5 $(SV.pretty_column(delta_col))",
        panel_attrs...,
    )
    plot!(p1, on_df.timestamp_ms, on_df[!, delta_col]; label = SV.pretty_condition("treatment"), color = :crimson)

    p2 = plot(
        joined_df.timestamp_ms,
        joined_df.delta_on_minus_off;
        label = "$(SV.pretty_condition("treatment")) - $(SV.pretty_condition("baseline"))",
        color = :darkgreen,
        xlabel = SV.pretty_column("timestamp_ms"),
        ylabel = "Delta Difference",
        title = "Pairwise Delta Difference",
        panel_attrs...,
    )
    hline!(p2, [0.0]; label = "zero", color = :black, linestyle = :dash)

    if entropy_col === nothing
        return plot(p1, p2; layout = (2, 1), size = fig_size, dpi = 180)
    end

    p3 = plot(
        off_df.timestamp_ms,
        off_df[!, entropy_col];
        label = "$(SV.pretty_column("routing_entropy")) (off)",
        color = :purple4,
        xlabel = SV.pretty_column("timestamp_ms"),
        ylabel = SV.pretty_column(entropy_col),
        title = "Routing Entropy",
        panel_attrs...,
    )
    plot!(p3, on_df.timestamp_ms, on_df[!, entropy_col]; label = "$(SV.pretty_column("routing_entropy")) (on)", color = :orange3)

    return plot(p1, p2, p3; layout = (3, 1), size = fig_size, dpi = 180)
end

struct ModelResult
    slug::String
    family::String
    off_run::Dict{String,Any}
    on_run::Dict{String,Any}
    delta_col::Symbol
    entropy_col::Union{Nothing,Symbol}
    run_summaries::Vector{Dict{String,Any}}
    pair_summary::Dict{String,Any}
    plot_path::String
end

function process_model(runs::AbstractVector, model_slug::AbstractString)
    off_run, on_run = model_pair(runs, model_slug)
    family = String(get(off_run, "family", ""))

    off_df = SV.load_latent_df(off_run)
    on_df = SV.load_latent_df(on_run)

    delta_col = SV.detect_delta_column([off_df, on_df])
    entropy_col = SV.maybe_entropy_column([off_df, on_df])

    run_summaries = [
        SV.summarise_run(off_df, off_run, delta_col, entropy_col),
        SV.summarise_run(on_df, on_run, delta_col, entropy_col),
    ]
    joined_df, pair_summary = SV.pairwise_summary(off_df, on_df, delta_col, entropy_col)

    plot_path = joinpath(OUTPUT_DIR, "$(model_slug).png")
    fig = build_plot(model_slug, off_df, on_df, joined_df, delta_col, entropy_col)
    savefig(fig, plot_path)

    return ModelResult(
        String(model_slug),
        family,
        Dict{String,Any}(off_run),
        Dict{String,Any}(on_run),
        delta_col,
        entropy_col,
        run_summaries,
        pair_summary,
        plot_path,
    )
end

function append_model_section!(lines::Vector{String}, result::ModelResult)
    push!(lines, "## $(SV.pretty_model(result.slug)) ($(result.family))")
    push!(lines, "")
    push!(lines, "- $(SV.pretty_condition("baseline")) run: `$(result.off_run["id"])`")
    push!(lines, "- $(SV.pretty_condition("treatment")) run: `$(result.on_run["id"])`")
    push!(lines, "- imported off:    `$(relpath(SV.imported_latent_path(result.off_run), REPO_ROOT))`")
    push!(lines, "- imported on:     `$(relpath(SV.imported_latent_path(result.on_run), REPO_ROOT))`")
    push!(lines, "- delta column:    `$(result.delta_col)`")
    push!(lines, "- routing entropy: `$(result.entropy_col === nothing ? "not present" : string(result.entropy_col))`")
    push!(lines, "")
    push!(lines, "| Condition | Rows | First ms | Last ms | Mean delta | Max delta | Final delta | Mean entropy | Final entropy |")
    push!(lines, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in result.run_summaries
        push!(
            lines,
            "| `$(row["condition"])` | $(row["rows"]) | $(row["first_ms"]) | $(row["last_ms"]) | $(SV.fmt(row["mean_delta"])) | $(SV.fmt(row["max_delta"])) | $(SV.fmt(row["final_delta"])) | $(SV.fmt(row["mean_entropy"])) | $(SV.fmt(row["final_entropy"])) |",
        )
    end
    push!(lines, "")
    push!(lines, "**Pairwise summary**")
    push!(lines, "")
    push!(lines, "| Metric | Value |")
    push!(lines, "| --- | ---: |")
    push!(lines, "| Paired rows | $(result.pair_summary["paired_rows"]) |")
    push!(lines, "| Mean delta (on - off) | $(SV.fmt(result.pair_summary["mean_delta_on_minus_off"])) |")
    push!(lines, "| Max abs delta (on - off) | $(SV.fmt(result.pair_summary["max_abs_delta_on_minus_off"])) |")
    push!(lines, "| Final delta (on - off) | $(SV.fmt(result.pair_summary["final_delta_on_minus_off"])) |")
    if haskey(result.pair_summary, "mean_entropy_on_minus_off")
        push!(lines, "| Mean entropy (on - off) | $(SV.fmt(result.pair_summary["mean_entropy_on_minus_off"])) |")
        push!(lines, "| Final entropy (on - off) | $(SV.fmt(result.pair_summary["final_entropy_on_minus_off"])) |")
    end
    push!(lines, "")
    push!(lines, "![$(result.slug)]($(basename(result.plot_path)))")
    push!(lines, "")
end

function append_cheat_sheet!(lines::Vector{String}, results::Vector{ModelResult})
    push!(lines, "## Cross-Model Delta Cheat Sheet")
    push!(lines, "")
    push!(lines, "| Model | Family | Mean delta (on - off) | Max abs delta (on - off) | Final delta (on - off) |")
    push!(lines, "| --- | --- | ---: | ---: | ---: |")
    for r in results
        ps = r.pair_summary
        push!(
            lines,
            "| $(SV.pretty_model(r.slug)) | `$(r.family)` | $(SV.fmt(ps["mean_delta_on_minus_off"])) | $(SV.fmt(ps["max_abs_delta_on_minus_off"])) | $(SV.fmt(ps["final_delta_on_minus_off"])) |",
        )
    end
    push!(lines, "")
end

function write_report(results::Vector{ModelResult})
    lines = String[]
    push!(lines, "# Full Lineup SAAQ 1.5 Paired-Run Comparison")
    push!(lines, "")
    push!(lines, "- Campaign: `$(CAMPAIGN)`")
    push!(lines, "- Repeat index: `$(REPEAT_IDX)`")
    push!(lines, "- Telemetry source: `$(TELEMETRY_SOURCE)`")
    push!(lines, "- Rule: `$(RULE)`")
    push!(lines, "- Equation source: `$(EQUATION_SOURCE)` (SymbolicRegression discovery)")
    push!(lines, "- Models: $(length(results)) ($(join((r.slug for r in results), ", ")))")
    push!(lines, "- Generated: `$(Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"))`")
    push!(lines, "")

    for r in results
        append_model_section!(lines, r)
    end
    append_cheat_sheet!(lines, results)

    open(REPORT_PATH, "w") do io
        write(io, join(lines, "\n"))
        write(io, "\n")
    end
end

function main()
    runs = load_selected_runs(SELECTED_RUNS_PATH)
    campaign_runs = full_lineup_runs(runs)
    isempty(campaign_runs) && error("No runs found for campaign=$(CAMPAIGN), repeat_idx=$(REPEAT_IDX) in $(SELECTED_RUNS_PATH)")
    available = validate_pairs(campaign_runs, MODEL_ORDER)

    mkpath(OUTPUT_DIR)
    results = ModelResult[]
    for slug in available
        push!(results, process_model(campaign_runs, slug))
    end

    write_report(results)

    println("Compared $(length(results)) full_lineup models for repeat $(REPEAT_IDX)")
    for r in results
        println("  $(r.slug): plot=$(relpath(r.plot_path, REPO_ROOT))")
    end
    println("Saved markdown report to $(REPORT_PATH)")
end

main()

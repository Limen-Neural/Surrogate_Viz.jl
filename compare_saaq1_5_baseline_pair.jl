using Pkg

# Only take over the active project when run as a script. Activating at load
# time would switch the caller's project out from under them when this file is
# `include`d — which is required to unit-test no_match_report below.
if abspath(PROGRAM_FILE) == @__FILE__
    Pkg.activate(@__DIR__)
end

using CSV
using DataFrames
using TOML
include(joinpath(@__DIR__, "src", "Surrogate_Viz.jl"))
# Resolve from the including module, not Main. The include above defines
# Surrogate_Viz *here*, so reading it from Main only works when Main happens to
# have it too — which makes this file unloadable in an isolated module. Same
# fix as #52 applied to SAAQ_latent_discovery.jl.
const SV = getfield(@__MODULE__, :Surrogate_Viz)

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
ENV["QT_QPA_PLATFORM"] = get(ENV, "QT_QPA_PLATFORM", "offscreen")
using Plots

# NOTE (Grok Build): Legacy script for old heartbeat baseline pair comparison.
# Updated to avoid references to removed control signal. Use for historical data only.

const REPO_ROOT = @__DIR__
const SELECTED_RUNS_PATH = joinpath(REPO_ROOT, "data", "selected_runs.toml")
const OUTPUT_DIR = joinpath(REPO_ROOT, "outputs", "olmoe_1b_7b_f16", "dashboards")
const REPORT_PATH = joinpath(OUTPUT_DIR, "saaq1_5_re4_condition_comparison.md")
const PLOT_PATH = joinpath(OUTPUT_DIR, "saaq1_5_re4_condition_comparison.png")
# NOTE (Grok Build 0.1 model, GH#40/MET-115): Updated OUTPUT_DIR to current slug to be consistent
# with hygiene cleanup (prevents recreating deleted "olmoe-1b-7b" placeholder-style dir).
# This remains a legacy script for historical baseline pair data only (guarded from #38 work).

function selected_repeat_idx()
    if !isempty(ARGS)
        return parse(Int, ARGS[1])
    end
    return parse(Int, get(ENV, "REPEAT_IDX", "0"))
end

function load_selected_runs(path::AbstractString)
    manifest = TOML.parsefile(path)
    runs = get(manifest, "runs", nothing)
    runs isa Vector || error("Expected [[runs]] entries in $(path)")
    return runs
end

"""
    no_match_report(runs, repeat_idx) -> String

Explain why `blessed_pair`'s filter selected nothing.

`only()` would report this as a bare "Collection is empty, must contain exactly
1 element", which says nothing about the cause. The sibling
compare_full_lineup_saaq1_5.jl already names its filter on failure.

Every predicate is counted *independently* against the manifest rather than
being explained by a fixed narrative. Whichever one is actually responsible
shows `0 match` beside the values that are present, so this stays correct for a
historical manifest that merely lacks the requested `repeat_idx` — a
hard-coded "your data uses prompt-profile conditions" explanation would be
wrong in exactly that case.
"""
function no_match_report(runs::AbstractVector, repeat_idx::Int)
    n = length(runs)
    observed(key) = sort(unique(string(get(r, key, "<absent>")) for r in runs))
    matching(pred) = count(pred, runs)

    # (label, required value, predicate) for each conjunct of the filter.
    checks = [
        ("blessed",          "true",                          r -> get(r, "blessed", false) == true),
        ("campaign",         "baseline_csv",                  r -> get(r, "campaign", nothing) == "baseline_csv"),
        ("model",            "olmoe_baseline",                r -> get(r, "model", nothing) == "olmoe_baseline"),
        ("family",           "Olmoe",                         r -> get(r, "family", nothing) == "Olmoe"),
        ("telemetry_source", "csv_re4_path_tracing_telemetry", r -> get(r, "telemetry_source", nothing) == "csv_re4_path_tracing_telemetry"),
        ("rule",             "SaaqV1_5SqrtRate",              r -> get(r, "rule", nothing) == "SaaqV1_5SqrtRate"),
        ("repeat_idx",       string(repeat_idx),              r -> haskey(r, "repeat_idx") && Int(r["repeat_idx"]) == repeat_idx),
    ]

    lines = String[
        "compare_saaq1_5_baseline_pair.jl: no blessed runs matched.",
        "  Manifest: $(SELECTED_RUNS_PATH) ($(n) runs)",
        "  Per-criterion match counts (each counted independently):",
    ]
    blockers = String[]
    for (label, required, pred) in checks
        hits = matching(pred)
        hits == 0 && push!(blockers, label)
        marker = hits == 0 ? "  <- blocks everything" : ""
        push!(lines, "    $(rpad(label, 17)) == $(rpad(required, 31)) $(lpad(hits, 3)) match$(marker)")
        if hits == 0
            push!(lines, "      observed: $(observed(label))")
        end
    end

    push!(lines, "  Blocking criteria: $(isempty(blockers) ? "none individually — no single run satisfies all at once" : join(blockers, ", "))")

    # The counts above are per-criterion and independent, and independence says
    # nothing about the conjunction: `model` can match one run while `rule`
    # matches a different one, with no single run satisfying both. So before
    # claiming the manifest is "otherwise compatible", evaluate the conjunction
    # of every criterion except repeat_idx against actual runs.
    others = [pred for (label, _, pred) in checks if label != "repeat_idx"]
    satisfying_others = filter(r -> all(p -> p(r), others), runs)

    if "campaign" in blockers && "model" in blockers
        push!(lines,
            "  This looks like the heartbeat-era schema mismatch: the manifest has no " *
            "matching campaign or model. That combination is what current sviz_* data " *
            "produces, and this script is for historical paired runs only.")
    elseif blockers == ["repeat_idx"] && !isempty(satisfying_others)
        # Safe to blame repeat_idx only now that runs are known to satisfy
        # everything else. Report the repeat values on *those* runs rather than
        # globally — a repeat that appears only on non-matching runs is not a
        # usable suggestion.
        available = sort(unique(Int(r["repeat_idx"]) for r in satisfying_others if haskey(r, "repeat_idx")))
        push!(lines,
            "  Only repeat_idx blocked, and $(length(satisfying_others)) run(s) satisfy every " *
            "other criterion. Available repeat_idx among those runs: " *
            "$(isempty(available) ? "none (they carry no repeat_idx key)" : string(available)).")
    elseif isempty(satisfying_others)
        # Covers the case where repeat_idx looks like the lone blocker but the
        # remaining criteria are only individually satisfiable.
        push!(lines,
            "  No single run satisfies the other criteria together, so the counts above " *
            "are individually satisfiable but jointly unsatisfiable — changing repeat_idx " *
            "alone will not help.")
    end

    return join(lines, "\n")
end

function blessed_pair(runs::AbstractVector, repeat_idx::Int)
    blessed_runs = filter(runs) do run
        get(run, "blessed", false) == true &&
        get(run, "campaign", nothing) == "baseline_csv" &&
        run["model"] == "olmoe_baseline" &&
        run["family"] == "Olmoe" &&
        run["telemetry_source"] == "csv_re4_path_tracing_telemetry" &&
        run["rule"] == "SaaqV1_5SqrtRate" &&
        Int(run["repeat_idx"]) == repeat_idx
    end

    if isempty(blessed_runs)
        error(no_match_report(runs, repeat_idx))
    end

    off_matches = filter(run -> run["condition"] == "baseline", blessed_runs)
    on_matches = filter(run -> run["condition"] == "treatment", blessed_runs)
    length(off_matches) == 1 || error(
        "compare_saaq1_5_baseline_pair.jl: expected exactly 1 baseline run, found " *
        "$(length(off_matches)) among $(length(blessed_runs)) blessed runs " *
        "(repeat_idx=$(repeat_idx)).",
    )
    length(on_matches) == 1 || error(
        "compare_saaq1_5_baseline_pair.jl: expected exactly 1 treatment run, found " *
        "$(length(on_matches)) among $(length(blessed_runs)) blessed runs " *
        "(repeat_idx=$(repeat_idx)).",
    )
    return only(off_matches), only(on_matches)
end

function build_plot(off_df::DataFrame, on_df::DataFrame, joined_df::DataFrame, delta_col::Symbol, entropy_col::Union{Nothing,Symbol})
    default(fontfamily = "Helvetica", legend = :topright, lw = 2.5, size = (1400, entropy_col === nothing ? 900 : 1200), dpi = 180)

    p1 = plot(
        off_df.timestamp_ms,
        off_df[!, delta_col];
        label = SV.pretty_condition("baseline"),
        color = :navy,
        xlabel = SV.pretty_column("timestamp_ms"),
        ylabel = SV.pretty_column(delta_col),
        title = "SAAQ 1.5 Baseline Pair: $(SV.pretty_column(delta_col))",
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
    )
    hline!(p2, [0.0]; label = "zero", color = :black, linestyle = :dash)

    if entropy_col === nothing
        return plot(p1, p2; layout = (2, 1))
    end

    p3 = plot(
        off_df.timestamp_ms,
        off_df[!, entropy_col];
        label = "$(SV.pretty_column("routing_entropy")) (off)",
        color = :purple4,
        xlabel = SV.pretty_column("timestamp_ms"),
        ylabel = SV.pretty_column(entropy_col),
        title = "Routing Entropy",
    )
    plot!(p3, on_df.timestamp_ms, on_df[!, entropy_col]; label = "$(SV.pretty_column("routing_entropy")) (on)", color = :orange3)

    return plot(p1, p2, p3; layout = (3, 1))
end

function write_report(
    off_run::Dict{String,<:Any},
    on_run::Dict{String,<:Any},
    repeat_idx::Int,
    delta_col::Symbol,
    entropy_col::Union{Nothing,Symbol},
    run_summaries::Vector{Dict{String,Any}},
    pair_summary::Dict{String,Any},
)
    lines = String[]
    push!(lines, "# SAAQ 1.5 RE4 Paired-Run Comparison")
    push!(lines, "")
    push!(lines, "- Repeat index: `$(repeat_idx)`")
    push!(lines, "- Model: `$(off_run["model"])` (`$(off_run["family"])`)")
    push!(lines, "- Telemetry source: `$(off_run["telemetry_source"])`")
    push!(lines, "- Rule: `$(off_run["rule"])`")
    push!(lines, "- Equation source: `outputs/20260414_194227_v2pNMk/pareto_front.csv` (SymbolicRegression discovery)")
    push!(lines, "- Delta column: `$(delta_col)`")
    push!(lines, "- Routing entropy column: `$(entropy_col === nothing ? "not present" : string(entropy_col))`")
    push!(lines, "- Plot: `$(relpath(PLOT_PATH, REPO_ROOT))`")
    push!(lines, "")
    push!(lines, "| Run | Condition | Rows | First ms | Last ms | Mean delta | Max delta | Final delta | Mean entropy | Final entropy |")
    push!(lines, "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for row in run_summaries
        push!(
            lines,
            "| `$(row["run_id"])` | `$(row["condition"])` | $(row["rows"]) | $(row["first_ms"]) | $(row["last_ms"]) | $(SV.fmt(row["mean_delta"])) | $(SV.fmt(row["max_delta"])) | $(SV.fmt(row["final_delta"])) | $(SV.fmt(row["mean_entropy"])) | $(SV.fmt(row["final_entropy"])) |",
        )
    end

    push!(lines, "")
    push!(lines, "## Pairwise Summary")
    push!(lines, "")
    push!(lines, "| Metric | Value |")
    push!(lines, "| --- | ---: |")
    push!(lines, "| Paired rows | $(pair_summary["paired_rows"]) |")
    push!(lines, "| Mean delta (on - off) | $(SV.fmt(pair_summary["mean_delta_on_minus_off"])) |")
    push!(lines, "| Max abs delta (on - off) | $(SV.fmt(pair_summary["max_abs_delta_on_minus_off"])) |")
    push!(lines, "| Final delta (on - off) | $(SV.fmt(pair_summary["final_delta_on_minus_off"])) |")
    if haskey(pair_summary, "mean_entropy_on_minus_off")
        push!(lines, "| Mean entropy (on - off) | $(SV.fmt(pair_summary["mean_entropy_on_minus_off"])) |")
        push!(lines, "| Final entropy (on - off) | $(SV.fmt(pair_summary["final_entropy_on_minus_off"])) |")
    end

    push!(lines, "")
    push!(lines, "## Imported Inputs")
    push!(lines, "")
    push!(lines, "- `$(relpath(SV.imported_latent_path(off_run), REPO_ROOT))`")
    push!(lines, "- `$(relpath(SV.imported_latent_path(on_run), REPO_ROOT))`")

    open(REPORT_PATH, "w") do io
        write(io, join(lines, "\n"))
        write(io, "\n")
    end
end

function main()
    repeat_idx = selected_repeat_idx()
    runs = load_selected_runs(SELECTED_RUNS_PATH)
    off_run, on_run = blessed_pair(runs, repeat_idx)

    off_df = SV.load_latent_df(off_run)
    on_df = SV.load_latent_df(on_run)

    delta_col = SV.detect_delta_column([off_df, on_df])
    entropy_col = SV.maybe_entropy_column([off_df, on_df])

    run_summaries = [
        SV.summarise_run(off_df, off_run, delta_col, entropy_col),
        SV.summarise_run(on_df, on_run, delta_col, entropy_col),
    ]
    joined_df, pair_summary = SV.pairwise_summary(off_df, on_df, delta_col, entropy_col)

    mkpath(OUTPUT_DIR)
    fig = build_plot(off_df, on_df, joined_df, delta_col, entropy_col)
    savefig(fig, PLOT_PATH)
    write_report(off_run, on_run, repeat_idx, delta_col, entropy_col, run_summaries, pair_summary)

    println("Compared blessed OLMoE baseline runs for repeat $(repeat_idx)")
    println("Loaded control-off from $(SV.imported_latent_path(off_run))")
    println("Loaded control-on from $(SV.imported_latent_path(on_run))")
    println("Saved plot to $(PLOT_PATH)")
    println("Saved markdown report to $(REPORT_PATH)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

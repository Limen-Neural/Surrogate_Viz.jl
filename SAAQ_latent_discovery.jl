# SAAQ latent equation discovery over imported corinth-canal runs
#
# Selects runs by RUN_ID or MODEL/CAMPAIGN/CONDITION/REPEAT_IDX from
# data/selected_runs.toml, loads their latent telemetry, validates columns,
# and runs SymbolicRegression.jl.
#
# Outputs go to outputs/<model>/sr_results/<run_id>/.

using Pkg

# Only take over the active project when run as a script. Activating (and
# especially instantiating) at load time would switch the caller's project out
# from under them when this file is `include`d — which the smoke tests do, and
# instantiate() would resolve the whole dependency tree on every test run.
if abspath(PROGRAM_FILE) == @__FILE__
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

using CSV
using DataFrames
using TOML
import SymbolicRegression
include(joinpath(@__DIR__, "src", "Surrogate_Viz.jl"))
# Resolve from the including module, not Main: the include above defines
# Surrogate_Viz *here*, so reading it from Main only works when Main happens to
# have it too. That made this script unloadable in an isolated module.
const SV = getfield(@__MODULE__, :Surrogate_Viz)

const REPO_ROOT = @__DIR__
const SELECTED_RUNS_PATH = joinpath(REPO_ROOT, "data", "selected_runs.toml")

const REQUIRED_COLS = [
    :avg_pop_firing_rate_hz,
    :membrane_dv_dt,
    :routing_entropy,
    :saaq_delta_q_prev,
    :saaq_delta_q_target,
]

const FEATURE_COLS = [
    :avg_pop_firing_rate_hz,
    :membrane_dv_dt,
    :routing_entropy,
    :saaq_delta_q_prev,
]

const TARGET_COL = :saaq_delta_q_target

function load_runs_manifest(path::AbstractString = SELECTED_RUNS_PATH)
    isfile(path) || error("Missing selected runs manifest: $(path)")
    manifest = TOML.parsefile(path)
    runs = get(manifest, "runs", nothing)
    runs isa Vector || error("Expected [[runs]] entries in $(path)")
    isempty(runs) && error("No selected runs found in $(path)")
    return runs
end

function select_run_by_id(runs, run_id::AbstractString)
    idx = findfirst(r -> r["id"] == run_id, runs)
    idx === nothing && error("No run with id=$(run_id) in manifest")
    return runs[idx]
end

function select_runs_by_metadata(runs;
    model::Union{Nothing,AbstractString} = nothing,
    campaign::Union{Nothing,AbstractString} = nothing,
    condition::Union{Nothing,AbstractString} = nothing,
    repeat_idx::Union{Nothing,Integer} = nothing,
    telemetry_source::Union{Nothing,AbstractString} = nothing,
    rule::Union{Nothing,AbstractString} = nothing,
)
    matched = filter(runs) do r
        (model === nothing || r["model"] == model) &&
        (campaign === nothing || r["campaign"] == campaign) &&
        (condition === nothing || r["condition"] == condition) &&
        (repeat_idx === nothing || Int(r["repeat_idx"]) == repeat_idx) &&
        (telemetry_source === nothing || r["telemetry_source"] == telemetry_source) &&
        (rule === nothing || r["rule"] == rule)
    end
    isempty(matched) && error("No runs match the given metadata filters")
    return matched
end

function select_runs(runs; run_id::Union{Nothing,AbstractString} = nothing, kw...)
    if run_id !== nothing
        return [select_run_by_id(runs, run_id)]
    end
    return select_runs_by_metadata(runs; kw...)
end

function validate_columns(df::DataFrame, required::Vector{Symbol})
    missing_cols = setdiff(required, propertynames(df))
    isempty(missing_cols) || error("Telemetry missing required column(s): $(join(string.(missing_cols), ", ")): $(propertynames(df))")
end

function build_feature_matrix(df::DataFrame, features::Vector{Symbol}, target::Symbol)
    X = Matrix{Float64}(hcat((Float64.(df[!, col]) for col in features)...))'
    y = Float64.(df[!, target])
    return X, y
end

function sr_output_dir(run::Dict{String,<:Any})
    model = SV.validate_path_component("model", run["model"])
    id = SV.validate_path_component("id", run["id"])
    joinpath(REPO_ROOT, "outputs", model, "sr_results", id)
end

function write_metadata(run::Dict{String,<:Any}, out_dir::AbstractString;
    options_dict::Dict = Dict{String,Any}(),
    niterations::Int = 0,
)
    mkpath(out_dir)
    meta = Dict{String,Any}(
        "run_id" => run["id"],
        "model" => run["model"],
        "telemetry_source" => run["telemetry_source"],
        "condition" => run["condition"],
        "repeat_idx" => Int(run["repeat_idx"]),
        "rule" => run["rule"],
        "feature_columns" => string.(FEATURE_COLS),
        "target_column" => string(TARGET_COL),
        "niterations" => niterations,
    )
    merge!(meta, options_dict)
    open(joinpath(out_dir, "sr_manifest.json"), "w") do io
        println(io, "{")
        items = collect(pairs(meta))
        for (i, (k, v)) in enumerate(items)
            val = v isa Vector ? (isempty(v) ? "[]" : "[\"" * join(string.(v), "\",\"") * "\"]") :
                  v isa Bool ? (v ? "true" : "false") :
                  v isa Number ? string(v) :
                  "\"" * replace(replace(string(v), "\\" => "\\\\"), "\"" => "\\\"") * "\""
            println(io, "  \"$(k)\": $(val)$(i < length(items) ? "," : "")")
        end
        println(io, "}")
    end
    return meta
end

function run_sr(X, y; niterations::Int = 30, options_kwargs...)
    opts = SymbolicRegression.Options(; options_kwargs...)
    hof = SymbolicRegression.equation_search(X, y; niterations = niterations, options = opts,
        variable_names = string.(FEATURE_COLS))
    return hof
end

function main()
    runs = load_runs_manifest()

    run_id = get(ENV, "RUN_ID", nothing)
    model = get(ENV, "MODEL", nothing)
    campaign = get(ENV, "CAMPAIGN", nothing)
    condition = get(ENV, "CONDITION", nothing)
    repeat_idx = let v = get(ENV, "REPEAT_IDX", nothing)
        v === nothing ? nothing : parse(Int, v)
    end
    telemetry_source = get(ENV, "TELEMETRY_SOURCE", nothing)
    rule = get(ENV, "RULE", nothing)
    niterations = let v = get(ENV, "SR_ITERATIONS", "30")
        parse(Int, v)
    end

    selected = select_runs(runs;
        run_id = run_id,
        model = model,
        campaign = campaign,
        condition = condition,
        repeat_idx = repeat_idx,
        telemetry_source = telemetry_source,
        rule = rule,
    )

    println("Selected $(length(selected)) run(s)")

    for run in selected
        println("\n=== Processing run $(run["id"]) ===")
        csv_path = SV.imported_latent_path(run)

        if !isfile(csv_path)
            @warn "Skipping run $(run["id"]): latent telemetry not found at $(csv_path)"
            continue
        end

        df = CSV.read(csv_path, DataFrame)
        validate_columns(df, REQUIRED_COLS)

        X, y = build_feature_matrix(df, FEATURE_COLS, TARGET_COL)
        println("Feature matrix: $(size(X)) rows=$(size(X, 2)), features=$(size(X, 1))")
        println("Target vector: $(length(y)) samples")

        out_dir = sr_output_dir(run)
        write_metadata(run, out_dir;
            niterations = niterations,
            options_dict = Dict{String,Any}(
                "binary_operators" => ["+", "-", "*", "/"],
                "unary_operators" => ["exp", "sqrt", "square"],
                "maxsize" => 15,
                "parsimony" => 0.01,
            ),
        )
        println("Metadata written to $(out_dir)/sr_manifest.json")

        if niterations > 0
            println("Launching SR search ($(niterations) iterations)...")
            sr_opts = SymbolicRegression.Options(
                binary_operators = [+, -, *, /],
                unary_operators = [exp, sqrt, SymbolicRegression.square],
                maxsize = 15,
                parsimony = 0.01,
                npopulations = 20,
                output_directory = out_dir,
            )
            hof = SymbolicRegression.equation_search(X, y; niterations = niterations, options = sr_opts,
                variable_names = string.(FEATURE_COLS))
            println("\n=== Pareto front for $(run["id"]) ===")
            dominating = SymbolicRegression.calculate_pareto_frontier(hof)
            for member in dominating
                comp = SymbolicRegression.compute_complexity(member, sr_opts)
                println("Loss: $(member.loss)  Complexity: $(comp)  Eq: $(member.tree)")
            end
            open(joinpath(out_dir, "pareto_front.csv"), "w") do io
                println(io, "complexity,loss,equation")
                for member in dominating
                    comp = SymbolicRegression.compute_complexity(member, sr_opts)
                    println(io, "$(comp),$(member.loss),\"$(replace(string(member.tree), "\"" => "\"\""))\"")
                end
            end
        else
            println("SR_ITERATIONS=0 — skipping SR search (dry run)")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

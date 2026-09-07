# SAAQ discovery over raw hardware telemetry — exploratory prototype.
#
# Searches for an expression predicting a normalised "compression level" target
# from GPU/CPU power and temperature, using SymbolicRegression.jl.
#
# ── Read this before trusting any output ──────────────────────────────────────
# The SNN firing-rate feature here is NOT measured. It is a stand-in derived
# from GPU temperature (see `synthetic_snn_firing_rate`), and the target is a
# normalised power sum rather than an observed compression level. Any expression
# this finds therefore describes a relationship between hardware channels and a
# construction of those same channels — it is a pipeline exercise, not evidence
# about the SNN.
#
# For discovery over real latent telemetry, use SAAQ_latent_discovery.jl, which
# reads actual per-run corinth-canal data selected from data/selected_runs.toml.
# ──────────────────────────────────────────────────────────────────────────────

using Pkg

# Only take over the active project when run as a script, so this file can be
# loaded (e.g. by the test suite) without switching the caller's project.
if abspath(PROGRAM_FILE) == @__FILE__
    Pkg.activate(@__DIR__)
end

using SymbolicRegression
using CSV
using DataFrames

const REPO_ROOT = @__DIR__
const TELEMETRY_PATH = joinpath(REPO_ROOT, "data", "hardware_telemetry.csv")

"""
Combined GPU + CPU package watts used as the divisor for the regression target.

**This does not bound the target to [0, 1].** It is a thermal-headroom
reference, not a hardware maximum, and the data routinely exceeds it. Measured
against the committed `data/hardware_telemetry.csv` (2000 rows):

| | watts | target |
|---|---|---|
| minimum | 290.3 | 0.726 |
| first row | 413.4 | 1.034 |
| maximum | 467.8 | 1.169 |

1589 of 2000 rows (79.4%) land above 1.0.

That is the intended direction — an earlier divisor of 500.0 W left a sustained
432 W draw at 0.864, so the target never reached the saturated end of the range
during a thermal transient and the search never saw it. But note the current
value overshoots: exceeding 1.0 is the common case rather than the exceptional
one, so "above 1.0" carries little signal as a saturation marker.

Deliberately not retuned here. Picking a divisor that makes the distribution
straddle 1.0 sensibly requires deciding what the target is meant to represent,
which is a modelling decision, not a cleanup. `main` prints the observed
fraction above 1.0 on every run so the choice stays visible.
"""
const FULL_LOAD_WATTS = 400.0

"""
Divisor mapping GPU temperature in °C onto roughly [0, 1] for use as a
stand-in firing rate. 100 °C is a nominal thermal ceiling, not a measured
maximum.
"""
const TEMP_NORMALISER_C = 100.0

"Rows read from the telemetry CSV. Kept small so the prototype stays quick."
const DEFAULT_ROW_LIMIT = 1000

"""
    synthetic_snn_firing_rate(gpu_temp_c) -> Vector{Float64}

A **stand-in** for SNN firing rate, derived from GPU temperature.

There is no SNN telemetry in `hardware_telemetry.csv`. This assumes firing rate
tracks GPU thermal load, which is an assumption this script cannot test — the
resulting feature is correlated with `gpu_temp_c` by construction, so any
expression using both is partly fitting the same signal twice.
"""
synthetic_snn_firing_rate(gpu_temp_c) = Float64.(gpu_temp_c) ./ TEMP_NORMALISER_C

"""
    normalised_load_target(gpu_power_w, cpu_package_power_w) -> Vector{Float64}

The regression target: combined draw as a fraction of [`FULL_LOAD_WATTS`].
"""
normalised_load_target(gpu_power_w, cpu_package_power_w) =
    (Float64.(gpu_power_w) .+ Float64.(cpu_package_power_w)) ./ FULL_LOAD_WATTS

"""
    build_features(df) -> (X, y, variable_names)

Assemble the SymbolicRegression input matrix. `X` is features-by-rows, as
`equation_search` expects.
"""
function build_features(df::DataFrame)
    required = [:gpu_temp_c, :gpu_power_w, :cpu_package_power_w]
    missing_cols = setdiff(required, propertynames(df))
    isempty(missing_cols) || error(
        "SAAQ_discovery.jl: $(TELEMETRY_PATH) is missing required column(s): " *
        "$(join(string.(missing_cols), ", ")). Present: $(join(string.(propertynames(df)), ", "))",
    )

    snn = synthetic_snn_firing_rate(df.gpu_temp_c)
    X = hcat(
        Float64.(df.gpu_temp_c),
        Float64.(df.gpu_power_w),
        Float64.(df.cpu_package_power_w),
        snn,
    )'
    y = normalised_load_target(df.gpu_power_w, df.cpu_package_power_w)
    names = ["gpu_temp_c", "gpu_power_w", "cpu_package_power_w", "synthetic_snn_firing_rate"]
    return X, y, names
end

function main()
    niterations = parse(Int, get(ENV, "SR_ITERATIONS", "30"))
    row_limit = parse(Int, get(ENV, "ROW_LIMIT", string(DEFAULT_ROW_LIMIT)))

    # ROW_LIMIT must be positive: 0 would hand equation_search an empty dataset,
    # and a negative value makes `first` throw with a message that says nothing
    # about where the value came from.
    row_limit > 0 || error(
        "SAAQ_discovery.jl: ROW_LIMIT must be a positive integer, got $(row_limit).",
    )
    niterations >= 0 || error(
        "SAAQ_discovery.jl: SR_ITERATIONS must be >= 0 (0 means dry run), got $(niterations).",
    )

    isfile(TELEMETRY_PATH) || error(
        "SAAQ_discovery.jl: telemetry not found at $(TELEMETRY_PATH).",
    )

    println("1. Loading raw hardware telemetry from $(TELEMETRY_PATH)")
    raw_data = CSV.read(TELEMETRY_PATH, DataFrame)
    nrow(raw_data) > 0 || error("SAAQ_discovery.jl: no rows in $(TELEMETRY_PATH)")
    df = first(raw_data, min(row_limit, nrow(raw_data)))
    println("   $(nrow(df)) of $(nrow(raw_data)) rows")

    println("2. Building features (NOTE: SNN firing rate is synthetic — see header)")
    X, y, variable_names = build_features(df)
    println("   features $(size(X, 1)) x rows $(size(X, 2))")

    # Report the target's actual range. FULL_LOAD_WATTS is a reference point,
    # not a bound, so this is where an unhelpful divisor becomes visible rather
    # than staying an assumption in the docstring.
    over = count(>(1.0), y)
    println("   target range    $(round(minimum(y), digits=3)) .. $(round(maximum(y), digits=3)) " *
            "(divisor $(FULL_LOAD_WATTS) W)")
    println("   above 1.0       $(over)/$(length(y)) rows " *
            "($(round(100 * over / length(y), digits=1))%)" *
            (over > length(y) ÷ 2 ? "  <- majority; divisor is low for a saturation marker" : ""))

    if niterations <= 0
        println("SR_ITERATIONS=0 — skipping the search (dry run)")
        return nothing
    end

    println("3. Launching symbolic regression ($(niterations) iterations)")
    options = SymbolicRegression.Options(
        binary_operators = [+, -, *, /],
        npopulations = 20,
        parsimony = 0.01,
    )
    hof = SymbolicRegression.equation_search(
        X, y;
        niterations = niterations,
        options = options,
        variable_names = variable_names,
    )

    println("\n=== DISCOVERY COMPLETE ===")
    println("Dominant equations found (against a synthetic target — see header):")
    print(hof)
    return hof
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

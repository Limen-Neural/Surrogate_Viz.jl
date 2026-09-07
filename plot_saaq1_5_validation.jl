const PkgMod = Base.require(Base.PkgId(Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"), "Pkg"))
PkgMod.activate(@__DIR__)

using CSV
using DataFrames
const DatesMod = Base.require(Base.PkgId(Base.UUID("ade2ca70-3891-5945-98fb-dc099432e06a"), "Dates"))
const RandomMod = Base.require(Base.PkgId(Base.UUID("9a3f8284-a2c9-5f02-9a11-845980a1fd5c"), "Random"))

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
ENV["QT_QPA_PLATFORM"] = get(ENV, "QT_QPA_PLATFORM", "offscreen")
using Plots

include(joinpath(@__DIR__, "src", "Surrogate_Viz.jl"))
const SV = getfield(Main, :Surrogate_Viz)

const TICK_PATTERN = r"^tick=(\d+) best_walker=(\d+) elapsed_us=(\d+) gpu_temp_c=([-+0-9.eE]+) gpu_power_w=([-+0-9.eE]+) cpu_tctl_c=([-+0-9.eE]+) cpu_package_power_w=([-+0-9.eE]+)$"

# Sourced from the label registry rather than retyped. Previously this list and
# compare_full_lineup_saaq1_5.jl's MODEL_ORDER were byte-identical copies of
# MODEL_DISPLAY's key set, so the GH#40/MET-115 slug rename (dropping GGUF
# placeholder names) had to be applied by hand in three places.
const MODELS = SV.KNOWN_MODEL_SLUGS

function model_name()
    m = get(ENV, "MODEL", MODELS[1])  # default olmoe-1b-7b
    m in MODELS || @warn "Unknown MODEL=$(m); expected one of $(MODELS)"
    return m
end

function default_dashboard_path()
    stamp = DatesMod.format(DatesMod.now(), "yyyymmdd_HHMMSS")
    suffix = RandomMod.randstring(6)
    dir = joinpath(@__DIR__, "outputs", model_name(), "dashboards", string(stamp, "_", suffix))
    mkpath(dir)
    return joinpath(dir, "saaq1_5_validation_dashboard.png")
end

# NOTE (Grok Build 0.1 model, GH#40 / MET-115 hygiene cleanup):
# Old flat data/<model>/ placeholder dirs were deleted. Active telemetry lives
# under data/corinth_runs/<slug>/csv_re4_path_tracing_telemetry/sviz_*/<run_id>/...
# Always prefer explicit paths for current sviz runs. The defaults below now
# error with guidance if no args provided (addresses bot feedback on broken defaults).
function latent_input_path()
    if length(ARGS) >= 1
        return ARGS[1]
    end
    error("No latent telemetry path provided as first argument.\n" *
          "Post GGUF placeholder cleanup (see plan and GH#40/MET-115), use explicit path e.g.\n" *
          "data/corinth_runs/<slug>/csv_re4_path_tracing_telemetry/sviz_math_logic/<run_id>/latent_telemetry.csv\n" *
          "Old data/<model>/snn_latent_telemetry.csv paths no longer exist.")
end

function tick_input_path()
    if length(ARGS) >= 2
        return ARGS[2]
    end
    error("No tick telemetry path provided as second argument.\n" *
          "Post GGUF placeholder cleanup (see plan and GH#40/MET-115), use explicit path e.g.\n" *
          "data/corinth_runs/<slug>/csv_re4_path_tracing_telemetry/sviz_math_logic/<run_id>/tick_telemetry.txt\n" *
          "Old data/<model>/tick_telemetry.txt paths no longer exist.")
end

output_path() = length(ARGS) >= 3 ? ARGS[3] : default_dashboard_path()

function load_latent_data(path::AbstractString)
    df = CSV.read(path, DataFrame)
    required = [
        :timestamp_ms,
        :avg_pop_firing_rate_hz,
        :routing_entropy,
        :gpu_temp_c,
        :gpu_power_w,
        :cpu_tctl_c,
        :cpu_package_power_w,
    ]
    missing = setdiff(required, Symbol.(names(df)))
    isempty(missing) || error("Missing required latent columns in $(path): $(join(string.(missing), ", "))")
    nrow(df) > 0 || error("No latent rows found in $(path)")

    df.tick = Int.(round.(df.timestamp_ms))
    sort!(df, :tick)
    return df
end

function load_tick_data(path::AbstractString)
    ticks = Int[]
    best_walkers = Int[]
    elapsed_us = Int[]
    gpu_temp_c = Float64[]
    gpu_power_w = Float64[]
    cpu_tctl_c = Float64[]
    cpu_package_power_w = Float64[]

    for line in eachline(path)
        match_result = match(TICK_PATTERN, line)
        match_result === nothing && continue

        push!(ticks, parse(Int, match_result.captures[1]))
        push!(best_walkers, parse(Int, match_result.captures[2]))
        push!(elapsed_us, parse(Int, match_result.captures[3]))
        push!(gpu_temp_c, parse(Float64, match_result.captures[4]))
        push!(gpu_power_w, parse(Float64, match_result.captures[5]))
        push!(cpu_tctl_c, parse(Float64, match_result.captures[6]))
        push!(cpu_package_power_w, parse(Float64, match_result.captures[7]))
    end

    isempty(ticks) && error("No tick data found in $(path).")

    return DataFrame(
        tick = ticks,
        best_walker = best_walkers,
        elapsed_us = elapsed_us,
        gpu_temp_c = gpu_temp_c,
        gpu_power_w = gpu_power_w,
        cpu_tctl_c = cpu_tctl_c,
        cpu_package_power_w = cpu_package_power_w,
    )
end

function scaled_overlay(signal, target)
    length(signal) == length(target) || error("Overlay length mismatch")
    isempty(signal) && return Float64[]

    signal_values = Float64.(signal)
    target_values = Float64.(target)

    signal_lo = minimum(signal_values)
    signal_hi = maximum(signal_values)
    target_lo = minimum(target_values)
    target_hi = maximum(target_values)

    if signal_hi - signal_lo < 1e-9
        fill((target_lo + target_hi) / 2, length(signal_values))
    elseif target_hi - target_lo < 1e-9
        fill(target_lo, length(signal_values))
    else
        target_lo .+ ((signal_values .- signal_lo) ./ (signal_hi - signal_lo)) .* (target_hi - target_lo)
    end
end

function build_dashboard(latent_df::DataFrame, tick_df::DataFrame)
    default(fontfamily = "Helvetica", legend = :topright, lw = 2, size = (1500, 1200), dpi = 180)

    p1 = plot(
        latent_df.tick,
        latent_df.avg_pop_firing_rate_hz;
        label = SV.pretty_column("avg_pop_firing_rate_hz"),
        color = :navy,
        xlabel = SV.pretty_column("tick"),
        ylabel = SV.pretty_column("avg_pop_firing_rate_hz"),
        title = "SAAQ 1.5 Validation: Firing Rate",
    )

    p2 = plot(
        latent_df.tick,
        latent_df.routing_entropy;
        label = SV.pretty_column("routing_entropy"),
        color = :darkgreen,
        xlabel = SV.pretty_column("tick"),
        ylabel = SV.pretty_column("routing_entropy"),
        title = "Routing Entropy",
    )

    p3 = scatter(
        tick_df.tick,
        tick_df.best_walker;
        label = SV.pretty_column("best_walker"),
        color = :purple4,
        markersize = 4,
        xlabel = SV.pretty_column("tick"),
        ylabel = SV.pretty_column("best_walker"),
        title = "Walker Activity",
    )

    p4 = plot(
        latent_df.tick,
        latent_df.gpu_temp_c;
        label = SV.pretty_column("gpu_temp_c"),
        color = :orange3,
        xlabel = SV.pretty_column("tick"),
        ylabel = "Telemetry",
        title = "Raw Telemetry Channels",
    )
    plot!(p4, latent_df.tick, latent_df.gpu_power_w; label = SV.pretty_column("gpu_power_w"), color = :royalblue3)
    plot!(p4, latent_df.tick, latent_df.cpu_tctl_c; label = SV.pretty_column("cpu_tctl_c"), color = :forestgreen)
    plot!(p4, latent_df.tick, latent_df.cpu_package_power_w; label = SV.pretty_column("cpu_package_power_w"), color = :brown3)

    plot(p1, p2, p3, p4; layout = (4, 1))
end

function main()
    latent_path = latent_input_path()
    tick_path = tick_input_path()
    output_file = output_path()

    latent_df = load_latent_data(latent_path)
    tick_df = load_tick_data(tick_path)
    fig = build_dashboard(latent_df, tick_df)
    savefig(fig, output_file)

    println("Loaded $(nrow(latent_df)) latent rows from $(latent_path)")
    println("Loaded $(nrow(tick_df)) tick rows from $(tick_path)")
    println("Saved dashboard to $(output_file)")
end

main()

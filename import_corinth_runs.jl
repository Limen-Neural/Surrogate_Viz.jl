using Pkg

# Only take over the active project when run as a script. Activating at load
# time would switch the caller's project out from under them when this file is
# `include`d — which the smoke tests do.
if abspath(PROGRAM_FILE) == @__FILE__
    Pkg.activate(@__DIR__)
end

using CSV
using DataFrames
using TOML

const REPO_ROOT = @__DIR__
const SELECTED_RUNS_PATH = joinpath(REPO_ROOT, "data", "selected_runs.toml")
const IMPORT_ROOT = joinpath(REPO_ROOT, "data", "corinth_runs")
const INDEX_PATH = joinpath(IMPORT_ROOT, "imported_runs_index.csv")
const IMPORT_FILES = [
    ("latent_telemetry_csv", "latent_telemetry.csv"),
    ("tick_telemetry_txt", "tick_telemetry.txt"),
    ("summary_json", "summary.json"),
]

force_import_enabled() = lowercase(strip(get(ENV, "FORCE_IMPORT", "false"))) in ("1", "true", "yes", "on")

function load_selected_runs(path::AbstractString)
    isfile(path) || error("Missing selected runs manifest: $(path)")
    manifest = TOML.parsefile(path)
    runs = get(manifest, "runs", nothing)
    runs isa Vector || error("Expected [[runs]] entries in $(path)")
    isempty(runs) && error("No selected runs found in $(path)")
    return runs
end

function local_run_dir(run::Dict{String,<:Any})
    joinpath(IMPORT_ROOT, run["model"], run["telemetry_source"], run["condition"], run["id"])
end

path_exists(path::AbstractString) = ispath(path) || islink(path)

function import_file!(source::AbstractString, dest::AbstractString; force::Bool)
    isfile(source) || error("Missing source file: $(source)")

    if path_exists(dest)
        if force
            rm(dest; force = true)
        else
            return "kept_existing"
        end
    end

    try
        symlink(source, dest)
        return "symlinked"
    catch err
        @warn "Symlink failed for $(dest); copying instead" exception = (err, catch_backtrace())
        cp(source, dest; force = true)
        return "copied"
    end
end

function import_run!(run::Dict{String,<:Any}; force::Bool)
    dest_dir = local_run_dir(run)
    mkpath(dest_dir)

    modes = String[]
    for (manifest_key, local_name) in IMPORT_FILES
        dest_path = joinpath(dest_dir, local_name)
        push!(modes, import_file!(run[manifest_key], dest_path; force = force))
    end

    return (
        id = run["id"],
        model = run["model"],
        family = run["family"],
        telemetry_source = run["telemetry_source"],
        condition = run["condition"],
        repeat_idx = Int(run["repeat_idx"]),
        rule = run["rule"],
        blessed = Bool(run["blessed"]),
        source_run_dir = run["source_run_dir"],
        imported_run_dir = relpath(dest_dir, REPO_ROOT),
        latent_telemetry_csv = relpath(joinpath(dest_dir, "latent_telemetry.csv"), REPO_ROOT),
        tick_telemetry_txt = relpath(joinpath(dest_dir, "tick_telemetry.txt"), REPO_ROOT),
        summary_json = relpath(joinpath(dest_dir, "summary.json"), REPO_ROOT),
        import_mode = join(modes, ","),
    )
end

function main()
    runs = sort(load_selected_runs(SELECTED_RUNS_PATH); by = run -> (run["model"], run["telemetry_source"], run["condition"], Int(run["repeat_idx"]), run["id"]))
    force = force_import_enabled()

    mkpath(IMPORT_ROOT)
    seen = Set{String}()
    imported_rows = []
    for run in runs
        rid = run["id"]
        if rid in seen
            @warn "Skipping duplicate run_id in selected_runs.toml: $(rid) (condition=$(run["condition"]))"
            continue
        end
        push!(seen, rid)
        push!(imported_rows, import_run!(run; force = force))
    end
    index_df = DataFrame(imported_rows)
    CSV.write(INDEX_PATH, index_df)

    println("Imported $(nrow(index_df)) corinth-canal runs into $(IMPORT_ROOT) (deduped by run_id)")
    println("Wrote import index to $(INDEX_PATH)")
    println("Import mode: $(force ? "force overwrite enabled" : "keep existing files")")
    println("Note: imports prefer symlinks and fall back to copies if symlinks are unavailable")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

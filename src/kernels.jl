using Statistics

function compute_pairwise_deltas(
    off_col::AbstractVector,
    on_col::AbstractVector,
    ::CPUBackend
)
    deltas = on_col .- off_col
    return (
        mean_delta = Statistics.mean(deltas),
        max_abs_delta = maximum(abs.(deltas)),
        final_delta = last(deltas),
    )
end

function compute_run_stats(
    delta_values::AbstractVector,
    entropy_values::Union{Nothing,AbstractVector},
    ::CPUBackend
)
    mean_delta = Statistics.mean(delta_values)
    max_delta = maximum(delta_values)
    final_delta = last(delta_values)

    if entropy_values !== nothing && !isempty(entropy_values)
        mean_entropy = Statistics.mean(entropy_values)
        final_entropy = last(entropy_values)
    else
        mean_entropy = missing
        final_entropy = missing
    end

    return (;
        mean_delta,
        max_delta,
        final_delta,
        mean_entropy,
        final_entropy,
    )
end

function compute_delta_per_tick(
    features::AbstractMatrix,
    ::CPUBackend
)
    return diff(features, dims=1)
end

function compute_delta_per_tick(
    timestamps::AbstractVector,
    features::AbstractMatrix,
    ::CPUBackend
)
    deltas = diff(features, dims=1)
    t_out = timestamps[2:end]
    return t_out, deltas
end

# Core shims for CUDABackend. These methods are always available in the package so
# callers can pass `backend=CUDABackend()` without manually loading `CUDA` first.
# When CUDA is installed and functional, they load the extension and dispatch to the
# GPU implementations; otherwise they fall back to CPUBackend with a warning.
function compute_pairwise_deltas(
    off_col::AbstractVector,
    on_col::AbstractVector,
    ::CUDABackend
)
    if has_cuda()
        ext = Base.get_extension(@__MODULE__, :CUDABackendExt)
        if ext !== nothing
            return Base.invokelatest(ext._compute_pairwise_deltas_cuda, off_col, on_col)
        end
    end
    @warn "CUDABackend requested but CUDA unavailable; using CPUBackend"
    return compute_pairwise_deltas(off_col, on_col, CPUBackend())
end

function compute_run_stats(
    delta_values::AbstractVector,
    entropy_values::Union{Nothing,AbstractVector},
    ::CUDABackend
)
    if has_cuda()
        ext = Base.get_extension(@__MODULE__, :CUDABackendExt)
        if ext !== nothing
            return Base.invokelatest(ext._compute_run_stats_cuda, delta_values, entropy_values)
        end
    end
    @warn "CUDABackend requested but CUDA unavailable; using CPUBackend"
    return compute_run_stats(delta_values, entropy_values, CPUBackend())
end

function compute_delta_per_tick(
    features::AbstractMatrix,
    ::CUDABackend
)
    if has_cuda()
        ext = Base.get_extension(@__MODULE__, :CUDABackendExt)
        if ext !== nothing
            return Base.invokelatest(ext._compute_delta_per_tick_cuda, features)
        end
    end
    @warn "CUDABackend requested but CUDA unavailable; using CPUBackend"
    return compute_delta_per_tick(features, CPUBackend())
end

function compute_delta_per_tick(
    timestamps::AbstractVector,
    features::AbstractMatrix,
    ::CUDABackend
)
    if has_cuda()
        ext = Base.get_extension(@__MODULE__, :CUDABackendExt)
        if ext !== nothing
            return Base.invokelatest(ext._compute_delta_per_tick_cuda, timestamps, features)
        end
    end
    @warn "CUDABackend requested but CUDA unavailable; using CPUBackend"
    return compute_delta_per_tick(timestamps, features, CPUBackend())
end

# -----------------------------------------------------------------------------
# Pure Julia CUDA visual kernels (Grok Build 0.1 model)
#
# The CUDA-backed implementations live in ext/CUDABackendExt.jl, a package
# extension that is only loaded when CUDA is loaded by the user. This keeps the
# package loadable and editor-friendly even when CUDA is not installed in the
# active environment.
#
# For standalone scripts and other callers, we always provide a CPU fallback.
# walker_density_bins_and_counts selects the GPU-backed implementation only when
# CUDA is installed and functional.
# -----------------------------------------------------------------------------

# Plain-Julia fallback (always available, no CUDA symbols here).
function _plain_walker_histogram(best_walkers::AbstractVector{Int}, n_bins::Int, max_walker::Int)
    hist = zeros(Int, n_bins)
    bin_size = (max_walker + 1) / n_bins
    for w in best_walkers
        b = clamp(floor(Int, w / bin_size), 0, n_bins - 1) + 1
        hist[b] += 1
    end
    return hist
end

"""
    _walker_histogram_dispatch(best_walkers, n_bins, max_walker) -> Vector{Int}

Choose the GPU or CPU walker-density histogram and say so when the GPU path is
skipped.

The two public entry points below previously had their own copies of this
selection logic and disagreed about it: `walker_density_bins_and_counts` fell
back to the CPU silently, while every other backend dispatcher in this file
warns, and `cuda_best_walker_density_histogram` warned with text that was wrong
in one of its two fallback cases. Both now route through here so the behaviour
cannot drift apart again.

The two reasons for falling back are reported distinctly, because they call for
different action: "CUDA is not functional" is a host or driver problem, while
"the extension is not loaded" just means nothing has run `using CUDA` in this
session and the GPU is probably fine.
"""
function _walker_histogram_dispatch(best_walkers::AbstractVector{Int}, n_bins::Int, max_walker::Int)
    if !has_cuda()
        @warn "CUDA is not functional; using CPU fallback for the walker density histogram" maxlog = 1
        return _plain_walker_histogram(best_walkers, n_bins, max_walker)
    end

    # Prefer the CUDA version only when the package extension is actually attached.
    # Use Base.invokelatest to avoid world-age issues when the extension is loaded
    # dynamically (e.g., first call in a fresh process).
    ext = Base.get_extension(@__MODULE__, :CUDABackendExt)
    if ext === nothing
        @warn "CUDA is functional but the CUDABackendExt extension is not loaded; using CPU fallback for the walker density histogram. Run `using CUDA` to enable the GPU path." maxlog = 1
        return _plain_walker_histogram(best_walkers, n_bins, max_walker)
    end

    return Base.invokelatest(ext._cuda_best_walker_density_histogram, best_walkers; n_bins=n_bins, max_walker=max_walker)
end

function walker_density_bins_and_counts(
    best_walkers::AbstractVector{Int};
    n_bins::Int = 32,
    max_walker::Int = 2047
)
    counts = _walker_histogram_dispatch(best_walkers, n_bins, max_walker)
    edges = range(0, max_walker, length = n_bins + 1)
    return collect(edges), counts
end

# The CUDA-backed implementation lives in ext/CUDABackendExt.jl.
# We document the public API here for discoverability.
"""
    cuda_best_walker_density_histogram(best_walkers; n_bins=32, max_walker=2047)

Pure-Julia CUDA implementation (with CPU fallback) of the histogram used for the
"Best Walker Firing Density" panel.

When CUDA is not installed or not functional, callers should use
`walker_density_bins_and_counts`, which safely falls back to the CPU path.
See the CUDA section in this file and ext/CUDABackendExt.jl for details.
(Grok Build 0.1 model — part of the #43/#44 combined visuals + runner work.)
"""
function cuda_best_walker_density_histogram(
    best_walkers::AbstractVector{Int};
    n_bins::Int = 32,
    max_walker::Int = 2047
)
    return _walker_histogram_dispatch(best_walkers, n_bins, max_walker)
end

module CUDABackendExt

using Statistics
using CUDA
import Surrogate_Viz: CPUBackend, CUDABackend, _plain_walker_histogram, compute_delta_per_tick,
    compute_pairwise_deltas, compute_run_stats, has_cuda

function _compute_pairwise_deltas_cuda(off_col::AbstractVector, on_col::AbstractVector)
    off_host = Float32.(collect(off_col))
    on_host = Float32.(collect(on_col))
    off_gpu = CUDA.cu(off_host)
    on_gpu = CUDA.cu(on_host)
    deltas_gpu = on_gpu .- off_gpu

    return (
        mean_delta = Float64(mean(deltas_gpu)),
        max_abs_delta = Float64(maximum(abs.(deltas_gpu))),
        final_delta = Float64(last(on_host) - last(off_host)),
    )
end

function _compute_run_stats_cuda(delta_values::AbstractVector, entropy_values::Union{Nothing,AbstractVector})
    delta_host = Float32.(collect(delta_values))
    deltas_gpu = CUDA.cu(delta_host)
    mean_delta = Float64(mean(deltas_gpu))
    max_delta = Float64(maximum(deltas_gpu))
    final_delta = Float64(last(delta_host))

    if entropy_values !== nothing && !isempty(entropy_values)
        entropy_host = Float32.(collect(entropy_values))
        ent_gpu = CUDA.cu(entropy_host)
        mean_entropy = Float64(mean(ent_gpu))
        final_entropy = Float64(last(entropy_host))
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

function _compute_delta_per_tick_cuda(features::AbstractMatrix)
    feat_gpu = CUDA.cu(Float32.(collect(features)))
    deltas_gpu = diff(feat_gpu, dims=1)
    return Array(deltas_gpu)
end

function _compute_delta_per_tick_cuda(timestamps::AbstractVector, features::AbstractMatrix)
    feat_gpu = CUDA.cu(Float32.(collect(features)))
    deltas_gpu = diff(feat_gpu, dims=1)
    return timestamps[2:end], Array(deltas_gpu)
end

# Grok Build 0.1 model: top-level kernel to avoid closure/Box issues on GPU.
function hist_kernel!(walkers, hist, bin_size_f, n_items, n_bins)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while i <= n_items
        w = walkers[i]
        b = clamp(floor(Int, w / bin_size_f), 0, n_bins - 1) + 1
        CUDA.@atomic hist[b] += Int32(1)
        i += stride
    end
    return
end

function _cuda_best_walker_density_histogram(
    best_walkers::AbstractVector{Int};
    n_bins::Int=32,
    max_walker::Int=2047,
)
    n = length(best_walkers)
    n == 0 && return zeros(Int, n_bins)

    if n > typemax(Int32) || any(w -> w < typemin(Int32) || w > typemax(Int32), best_walkers)
        @warn "CUDA unavailable for walker density histogram inputs outside Int32 range; using CPU fallback"
        return _plain_walker_histogram(best_walkers, n_bins, max_walker)
    end

    d_walkers = CUDA.cu(Int32.(collect(best_walkers)))
    d_hist = CUDA.zeros(Int32, n_bins)

    bin_size = (max_walker + 1) / n_bins

    threads = 256
    blocks = min(1024, cld(n, threads))
    CUDA.@cuda threads=threads blocks=blocks hist_kernel!(d_walkers, d_hist, Float32(bin_size), n, n_bins)

    return Int.(Array(d_hist))
end

# NOTE: this extension deliberately does NOT define a public
# `cuda_best_walker_density_histogram`. It used to, with the identical
# signature to the one in src/kernels.jl — and because the name is imported
# from the parent above, that was a method *replacement*, not an addition.
# Loading the extension silently swapped out the parent's implementation
# (which is also what triggered "Method overwriting is not permitted during
# Module precompilation"), so the named entry point kept its own fallback
# branch, its own out-of-date warning text, and no maxlog — the exact drift
# between the two entry points that consolidating the dispatch was meant to
# end.
#
# The parent's dispatcher reaches the GPU path through
# `_cuda_best_walker_density_histogram` above, so nothing here needs a public
# wrapper.

end # module CUDABackendExt

# Labels — centralized human-readable label registry for Surrogate_Viz
# All plot axes, legends, condition names, and model names reference this module.

using DataFrames

# ──────────────────────────────────────────────────────────────────────────────
# Condition labels
# ──────────────────────────────────────────────────────────────────────────────

const CONDITION_LABELS = Dict(
    "sviz_math_logic"   => "Math & Logic",
    "sviz_rust_syntax"  => "Rust Syntax",
    "sviz_english_snn"  => "English SNN",
    "baseline"          => "Baseline (Control Off)",
    "treatment"         => "Treatment (Control On)",
)

function pretty_condition(code::AbstractString)::String
    get(CONDITION_LABELS, code, code)
end

# ──────────────────────────────────────────────────────────────────────────────
# Column / axis labels
# ──────────────────────────────────────────────────────────────────────────────

const COLUMN_LABELS = Dict(
    "avg_pop_firing_rate_hz"   => "Population Firing Rate (Hz)",
    "routing_entropy"          => "Routing Entropy",
    "best_walker"              => "Best Walker Index",
    "saaq_delta_q_v15_target"  => "SAAQ Delta Q (v1.5 Target)",
    "saaq_delta_q_target"      => "SAAQ Delta Q (Target)",
    "saaq_delta_q_prev"       => "SAAQ Delta Q (Previous)",
    "gpu_temp_c"               => "GPU Temperature (\u00b0C)",
    "gpu_power_w"              => "GPU Power (W)",
    "cpu_tctl_c"               => "CPU Temperature (\u00b0C)",
    "cpu_package_power_w"      => "CPU Package Power (W)",
    "timestamp_ms"             => "Timestamp (ms)",
    "tick"                     => "Tick",
    "elapsed_us"               => "Elapsed Time (\u03bcs)",
    "membrane_dv_dt"           => "Membrane dV/dt",
)

function pretty_column(name::AbstractString)::String
    get(COLUMN_LABELS, name, name)
end
pretty_column(sym::Symbol) = pretty_column(string(sym))

# ──────────────────────────────────────────────────────────────────────────────
# Model display names
# ──────────────────────────────────────────────────────────────────────────────

const MODEL_DISPLAY = Dict(
    "olmoe_1b_7b_f16"                => "OLMoE 1B-7B",
    "qwen3_moe_iq3_m"                => "Qwen3 MoE",
    "gemma4_26b_a4b_iq4_nl"          => "Gemma4 26B-A4B",
    "deepseek_coder_v2_lite_q6_k_l"  => "DeepSeek Coder V2 Lite",
    "llama_3_2_dark_champion_q5_k_m" => "Llama 3.2 Dark Champion",
    "zaya1_8b_q8_0"                  => "Zaya 1.8B",
    "kimi_vl_a3b_q6_k"               => "Kimi VL A3B",
    "marco_nano_base_q8_0"           => "Marco Nano Base",
)

function pretty_model(slug::AbstractString)::String
    get(MODEL_DISPLAY, slug, slug)
end

"""
    KNOWN_MODEL_SLUGS

The active model roster, in canonical presentation order.

`MODEL_DISPLAY` above is a `Dict` and therefore unordered, so it cannot serve
as the roster on its own — scripts that iterate models need a stable sequence.
This is that sequence, and it is the single place the roster is declared.

Before this existed the same eight slugs were retyped in three files
(`compare_full_lineup_saaq1_5.jl`'s `MODEL_ORDER`,
`plot_saaq1_5_validation.jl`'s `MODELS`, and `MODEL_DISPLAY`'s key set), so a
rename needed three synchronised hand-edits with nothing tying them together —
and the repo has already been through exactly that rename once (GH#40).

`known_model_slugs_consistent()` asserts this list and `MODEL_DISPLAY` describe
the same set, so adding to one without the other is a test failure rather than
a silent drift.
"""
const KNOWN_MODEL_SLUGS = [
    "olmoe_1b_7b_f16",
    "qwen3_moe_iq3_m",
    "gemma4_26b_a4b_iq4_nl",
    "deepseek_coder_v2_lite_q6_k_l",
    "llama_3_2_dark_champion_q5_k_m",
    "zaya1_8b_q8_0",
    "kimi_vl_a3b_q6_k",
    "marco_nano_base_q8_0",
]

"""
    known_model_slugs_consistent() -> Bool

True when `KNOWN_MODEL_SLUGS` and the keys of `MODEL_DISPLAY` are the same set.

They are two views of one roster: the list gives order, the dict gives display
names. Nothing in the language keeps them aligned, so this is asserted in the
test suite.
"""
known_model_slugs_consistent() = Set(KNOWN_MODEL_SLUGS) == Set(keys(MODEL_DISPLAY))

# ──────────────────────────────────────────────────────────────────────────────
# Walker behavioral groups
# ──────────────────────────────────────────────────────────────────────────────

struct WalkerGroup
    id::Int
    role::Symbol       # :attractor, :explorer, :rare
    label::String      # "Attractor #129", "Explorer #1527", "Rare #42"
end

function classify_walkers(tick_data::DataFrame; attractor_threshold::Float64=0.05, rare_threshold::Float64=0.001)
    hasproperty(tick_data, :best_walker) || error("classify_walkers: best_walker column not found")

    counts = combine(groupby(tick_data, :best_walker), nrow => :count)
    total = sum(counts.count)
    total == 0 && return WalkerGroup[]
    counts[!, :freq] = counts.count ./ total

    groups = WalkerGroup[]
    for row in eachrow(counts)
        role = if row.freq >= attractor_threshold
            :attractor
        elseif row.freq >= rare_threshold
            :explorer
        else
            :rare
        end
        label = if role == :attractor
            "Attractor #$(row.best_walker)"
        elseif role == :explorer
            "Explorer #$(row.best_walker)"
        else
            "Rare #$(row.best_walker)"
        end
        push!(groups, WalkerGroup(row.best_walker, role, label))
    end
    return groups
end

function walker_label(groups::Vector{WalkerGroup}, walker_id::Int)::String
    for g in groups
        g.id == walker_id && return g.label
    end
    return "Walker #$walker_id"
end

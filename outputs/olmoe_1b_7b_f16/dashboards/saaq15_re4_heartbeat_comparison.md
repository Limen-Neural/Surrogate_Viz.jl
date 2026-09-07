# SAAQ 1.5 RE4 Heartbeat Comparison

- Repeat index: `0`
- Model: `olmoe_baseline` (`Olmoe`)
- Telemetry source: `csv_re4_path_tracing_telemetry`
- Rule: `SaaqV1_5SqrtRate`
- Delta column: `saaq_delta_q_v15_target`
- Routing entropy column: `routing_entropy`
- Plot: `outputs/olmoe-1b-7b/dashboards/saaq15_re4_heartbeat_comparison.png`

| Run | Heartbeat | Rows | First ms | Last ms | Mean delta | Max delta | Final delta | Mean entropy | Final entropy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260423T195615_math_logic_r0_baseline_csv_off` | `heartbeat_off` | 2000 | 1 | 2000 | 0.0 | 0.0 | 0.0 | 0.999985 | 0.999984 |
| `20260423T195816_math_logic_r0_baseline_csv_on` | `heartbeat_on` | 2000 | 1 | 2000 | 0.0 | 0.0 | 0.0 | 0.999984 | 0.999983 |

## Pairwise Summary

| Metric | Value |
| --- | ---: |
| Paired rows | 2000 |
| Mean delta (on - off) | 0.0 |
| Max abs delta (on - off) | 0.0 |
| Final delta (on - off) | 0.0 |
| Mean entropy (on - off) | -0.0 |
| Final entropy (on - off) | -1.0e-6 |

## Imported Inputs

- `data/corinth_runs/olmoe_baseline/csv_re4_path_tracing_telemetry/heartbeat_off/20260423T195615_math_logic_r0_baseline_csv_off/latent_telemetry.csv`
- `data/corinth_runs/olmoe_baseline/csv_re4_path_tracing_telemetry/heartbeat_on/20260423T195816_math_logic_r0_baseline_csv_on/latent_telemetry.csv`

# Validation Commands

This file collects exact smoke-test commands for CI, contributors, and PR verification.

All commands assume the working directory is the repo root (`Surrogate_Viz.jl/`).

## 1. Install Dependencies

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## 2. Run Full Test Suite

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all tests pass including SAAQ bundle, compute backend, and grok-ozempic tests.

## SAAQ Bundle Smoke Tests

### Validate a single SAAQ fixture bundle

```bash
julia --project=. scripts/validate_saaq_bundle.jl test/fixtures/bundles/successful_synthetic/
# Expected: exit code 0, prints run details

julia --project=. scripts/validate_saaq_bundle.jl test/fixtures/bundles/skipped_run/
# Expected: exit code 0

julia --project=. scripts/validate_saaq_bundle.jl test/fixtures/bundles/failed_run/
# Expected: exit code 0 (failed bundles are valid bundles, just with failed status)

julia --project=. scripts/validate_saaq_bundle.jl test/fixtures/bundles/nonexistent/
# Expected: exit code 1, error message
```

### Ingest all SAAQ fixture bundles → normalized CSVs

```bash
julia --project=. scripts/ingest_saaq_bundles.jl test/fixtures/bundles/ /tmp/saaq_normalized/
# Expected: writes runs_table.csv, metrics_table.csv, warnings_table.csv under /tmp/saaq_normalized/
# 6 runs ingested (successful_synthetic, skipped_run, failed_run, missing_optional, unknown_extras, synthetic_smokescreen)
```

### Build dashboard from normalized SAAQ fixtures

```bash
julia --project=. scripts/build_saaq_dashboard.jl /tmp/saaq_normalized/ /tmp/saaq_reports/
# Expected: writes dashboard.html, summary.md, runs_table.csv, metrics_table.csv,
#           warnings_table.csv under /tmp/saaq_reports/<yyyy-mm-dd>/
```

## Grok-Ozempic Bundle Smoke Tests

### Validate a single grok-ozempic fixture bundle

```bash
julia --project=. scripts/validate_grok_ozempic_bundle.jl test/fixtures/grok_ozempic/pass/
# Expected: exit code 0, prints report details (status=PASS, 770 tensors, 0 failures)

julia --project=. scripts/validate_grok_ozempic_bundle.jl test/fixtures/grok_ozempic/fail/
# Expected: exit code 0, prints report details (status=FAIL, 3 failures, 2 warnings)

julia --project=. scripts/validate_grok_ozempic_bundle.jl test/fixtures/grok_ozempic/warnings_only/
# Expected: exit code 0, prints report details (status=PASS, 2 warnings)

julia --project=. scripts/validate_grok_ozempic_bundle.jl test/fixtures/grok_ozempic/missing_optional/
# Expected: exit code 0, prints report details (status=PASS, defaults for omitted fields)

julia --project=. scripts/validate_grok_ozempic_bundle.jl test/fixtures/grok_ozempic/nonexistent/
# Expected: exit code 1, error message
```

### Ingest all grok-ozempic fixture bundles → normalized CSVs

```bash
julia --project=. scripts/ingest_grok_ozempic_bundles.jl test/fixtures/grok_ozempic/ /tmp/grok_normalized/
# Expected: writes runs_table.csv, metrics_table.csv, issues_table.csv under /tmp/grok_normalized/
# 4 bundles ingested (pass, fail, warnings_only, missing_optional)
```

## Fixture Coverage

### SAAQ Fixtures

| Fixture | Run Status | Coverage |
|---|---|---|
| `successful_synthetic/` | `synthetic` | Complete manifest + summary |
| `skipped_run/` | `skipped` | Missing metrics (ticks_completed=0) |
| `failed_run/` | `failed` | error field set, validation_status=failed |
| `missing_optional/` | `real` | Many optional manifest fields absent |
| `unknown_extras/` | `real` | Unknown extra fields in manifest + summary.metrics |
| `synthetic_smokescreen/` | `synthetic` | Full fixture with model_slug, repeat_idx/count |

### Grok-Ozempic Fixtures

| Fixture | Status | Coverage |
|---|---|---|
| `pass/` | `PASS` | Complete report, 770 tensors, byte match, no failures/warnings |
| `fail/` | `FAIL` | 3 failures (missing_tensor, router_policy_violation, shape_mismatch), 2 warnings |
| `warnings_only/` | `PASS` | 2 unresolved expert projection warnings, no failures |
| `missing_optional/` | `PASS` | Omitted optional fields (source_tensor_count, source_total_bytes, etc.) |

## What Is Real vs Synthetic

- **Real runs**: runs where `validation_status = "completed"` and `error = null`
- **Synthetic runs**: fixture/test runs marked `run_status = synthetic` in the manifest
  (currently `successful_synthetic` is marked synthetic since it has validation_status=synthetic)
- **Skipped runs**: `validation_status = "skipped"` — incomplete runs, no metrics
- **Failed runs**: `validation_status = "failed"` or `error !== null` — runs that errored

## Notes

- **corinth-canal is not yet ready.** All SAAQ bundles come from synthetic fixtures
  under `test/fixtures/bundles/`. Real corinth-canal integration will be documented
  once the pipeline is live.
- `warnings.jsonl` is defined in the corinth-canal schema but not currently emitted,
  so no bundle contributes per-run warning rows yet.
- The warning table is **not** always empty, however: a bundle that fails to load is
  recorded there with `warning_category = "bundle_load_failure"`,
  `severity = "load_error"`, and the offending `bundle_path`. Such a bundle has no
  `run_id` — reading it is what failed — so that column is blank on those rows.
  Check for `severity == "load_error"` before treating an ingest as complete: the
  ingest CLIs report the number of bundles that *loaded*, not the number found.
- `artifacts.json` is defined in the schema but not emitted. Artifact references
  are not yet supported in this ingestion layer.
- Unknown manifest and metrics fields are preserved in the `extra` dict on each struct
  and exposed as `extra_<field>` columns in the runs/metrics tables.
- Grok-ozempic bundles only read embedded `failures`/`warnings` arrays in
  `validation.report.json`. Sidecar files (`validation.failures.json`,
  `validation.warnings.json`) are not yet supported.

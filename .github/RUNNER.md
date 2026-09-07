# Self-hosted GPU runner — configuration and threat model

This repository dispatches three CI jobs to a self-hosted runner with an NVIDIA
RTX 5080 (sm_120). This document records how that runner is configured, what the
fork-protection controls actually do, and — importantly — what they do **not**
cover.

Closes the documentation criterion of #44.

## What runs where

| Job | Workflow | Runner | Needs a GPU? |
|---|---|---|---|
| `test-cpu` | `julia.yml` | `ubuntu-latest` | no |
| `test` | `julia.yml` | self-hosted | yes — adds CUDA, runs the CUDA testset |
| `gpu-preflight` | `julia.yml` | self-hosted | yes — host + container `nvidia-smi` |
| `cuda-visuals` | `julia.yml` | self-hosted | yes — runs CUDA visual kernels |
| `build` | `docker-build.yml` | `ubuntu-latest` | no |
| `publish` | `docker-publish.yml` | `ubuntu-latest` | no |

Only jobs that genuinely need the GPU run on the self-hosted host. The two
Docker jobs were moved to GitHub-hosted runners in #50: `docker build` executes
the image's `RUN` instructions but never needs a GPU, and `Dockerfile:49`
already tolerates `nvidia-smi` being absent.

`test-cpu` is the only job that runs for **fork** pull requests, and it is the
only job whose result can be trusted as a gate on external contributions.

## Runner registration

| | |
|---|---|
| Name | `Surrogate-Viz-sm120` |
| Labels | `self-hosted, Linux, X64, GPU, docker, julia-cuda, sm120` |
| Mode | persistent (**not** ephemeral — see below) |
| Supervision | systemd **user** service, `github-runner-surrogate-viz` |
| Working dir | `~/actions-runner/Surrogate_Viz.jl-runner` |

Workflows request `gpu`; the runner registers `GPU`. That is fine — GitHub
states all runner labels are case-insensitive.

Supervision is a *user* service, not a system one, so no root and no `sudo` is
involved:

```bash
systemctl --user status  github-runner-surrogate-viz
systemctl --user restart github-runner-surrogate-viz
journalctl --user -u github-runner-surrogate-viz -f
```

`loginctl` lingering is enabled for the account, so the service survives logout
and starts at boot. `svc.sh install` is deliberately **not** used: it writes a
unit to `/etc/systemd/system` and needs root, which would mean either an
interactive password prompt on every runner operation or a `NOPASSWD` sudoers
rule. Given the shared-host exposure described below, granting passwordless root
to the account that executes CI jobs is not a trade worth making.

### Do not re-enable ephemeral mode without a supervisor

The registration was previously created with `--ephemeral`. An ephemeral runner
accepts one job and then de-registers itself; `run-ephemeral.sh` exists to
re-register in a loop, but nothing kept it alive. The result was that after its
last job the runner silently vanished from GitHub while the local `.runner` file
still claimed it was configured — so `config.sh` refused with "already
configured" *and* `config.sh remove` demanded a removal token it could no longer
validate.

If that state recurs, `--local` is the escape:

```bash
cd ~/actions-runner/Surrogate_Viz.jl-runner
./config.sh remove --local          # clears LOCAL config without contacting GitHub
TOKEN=$(gh api -X POST repos/rmems/Surrogate_Viz.jl/actions/runners/registration-token --jq .token)
./config.sh --url https://github.com/rmems/Surrogate_Viz.jl --token "$TOKEN" \
  --name Surrogate-Viz-sm120 --labels gpu,sm120,julia-cuda,docker --unattended --replace
systemctl --user restart github-runner-surrogate-viz
```

`--labels` takes only the custom labels; `self-hosted,Linux,X64` are added
automatically.

### A job that queues forever looks identical to a broken runner

If no online runner carries every label in a job's `runs-on` list, GitHub keeps
the job **queued for 24 hours** and then cancels it. `timeout-minutes` does not
help: it bounds how long a job may *run*, not how long it may wait. PR #48 is
the worked example — its self-hosted jobs sat queued for exactly 24h00m01s.

So a label typo produces a day-long silent hang, not a fast failure. Verify
labels after any re-registration:

```bash
gh api repos/rmems/Surrogate_Viz.jl/actions/runners \
  --jq '.runners[] | {name, status, labels: [.labels[].name]}'
```

**Do not copy `runs-on` blocks between this repo and `rmems/corinth-canal`.**
They target the same physical machine but spell the architecture label
differently — `sm120` here, `sm_120` there. A copied snippet will never match a
runner and will hang rather than fail.

## Controls that are in place

- **Fork PRs cannot reach the GPU runner.** Every self-hosted job carries:

  ```yaml
  if: >-
    github.event_name == 'push' ||
    github.event.pull_request.head.repo.full_name == github.repository
  ```

  A pull request from a fork has a different `head.repo.full_name`, so the job
  is skipped.

- **No `pull_request_target`.** That trigger runs with the base repository's
  secrets against the *fork's* code. It is not used anywhere in this repo.

- **No outsider-triggerable dispatch.** No `workflow_dispatch` or
  `issue_comment` trigger exists, so there is no comment-driven path to the
  runner.

- **Dedicated labels.** GPU jobs select `gpu, sm120, julia-cuda`, which no
  hosted runner carries, so a job cannot land there by accident.

- **Least-privilege token.** `julia.yml` sets `permissions: contents: read` at
  workflow level, re-declared per job on `gpu-preflight` and `cuda-visuals`. The
  only job that ever holds `packages: write` is `publish`, and it no longer runs
  on this host.

## What is NOT mitigated

These are real, open exposures. They are recorded here rather than omitted,
because a threat model that lists only the controls is misleading.

### The host is shared across repositories with no isolation

`~/actions-runner/` holds **eight** per-repo runner directories
(Surrogate_Viz.jl, corinth-canal, silicon-hdl, blackwell-kernel-lab,
myelin-accelerator, LiquidCortex.jl, Theseus-Quarry, XAIDissect_Viz), all owned
by the same OS user, with several listeners live at once. Jobs run directly on
the host — no container, no user namespace, no per-job sandbox.

Consequently, code executing in **any** job on this machine runs as the account
that owns **every** repository's runner credentials. This includes code you did
not write: a compromised or typosquatted Julia package resolved during
`Pkg.instantiate()`, or a third-party action pinned to a tag its maintainer can
move.

The fork guard scopes *who can trigger* a GPU job. It does nothing about what
that job can reach once running.

Related: `corinth-canal-runner/.secrets` is a symlink to the account's real
secrets file, and stale `.credentials.bak.corinth` / `.runner.bak.corinth`
(mode 0644) remain in the bare `~/actions-runner`. Those backups carry the auth
scheme and client id, not the runner's private key — no
`.credentials_rsaparams` backup exists — so they are stale-file hygiene rather
than a leaked identity, but they should be deleted.

Reducing this properly means one unprivileged OS user per runner, or
container-per-job execution. Both are host-level changes outside this
repository.

### The Julia depot is shared

Self-hosted jobs run `Pkg.instantiate()` / `Pkg.add("CUDA")` against the
account's single `~/.julia` with no per-repo `JULIA_DEPOT_PATH`. A tampered
package resolved by any repository on this host persists for the next job of any
other repository.

### Most actions are pinned to movable tags

`actions/checkout@v4`, `julia-actions/setup-julia@v2`, `julia-actions/cache@v2`,
`docker/build-push-action@v5` and `docker/login-action@v3` are tag-pinned. A tag
can be repointed by its maintainer or by whoever compromises that account.
`docker/setup-buildx-action` is pinned to a full commit SHA; the rest are not
yet.

### `gpu-preflight` is green whenever Docker works, regardless of the GPU

The job runs under `set -eux`, and three commands are deliberately unguarded
(`julia.yml:69-71`):

```bash
[ -n "$DOCKER_BIN" ]
"$DOCKER_BIN" version
"$DOCKER_BIN" info | sed -n '1,80p'
```

So it *does* fail — but only when Docker itself is missing or its daemon is
unreachable. Every GPU check after that point ends in `|| true` or an `echo`
fallback, including the `--gpus all` passthrough tests and the explicit "GPU
passthrough check FAILED" branch, which prints diagnostics and then exits 0.

The practical consequence: a green `gpu-preflight` means *Docker is alive on
the host*. It is not evidence the GPU is usable. Read the log — or rely on
`cuda-visuals`, which does hard-assert via `CUDA.functional()` in
`run_cuda_visuals.sh:53`.

### There is no branch protection

`branches/main/protection` returns 404 and no rulesets are configured, so **no
check is required to merge** — including `test-cpu` and every security bot.
Merge decisions are entirely human.

## If you are reviewing a PR that touches `.github/workflows/`

A workflow change is the one edit that can rewrite these controls, and it takes
effect on the PR branch itself. Check specifically that a PR does not: add
`pull_request_target`; remove or weaken a fork guard; broaden `permissions:`;
add a `workflow_dispatch`/`issue_comment` trigger; or move a job onto the
self-hosted labels that does not need a GPU.

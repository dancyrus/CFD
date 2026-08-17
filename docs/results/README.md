# Committed results

**Results get committed, not reported in chat** (CLAUDE.md). Every
acceptance-ladder rung and every benchmark writes its measured numbers into
this directory through the `cfd-results` crate — automatically, from inside
the test, *before* the test's own asserts, so emission cannot be skipped and
a failing run still records its numbers (with `"pass": false`). Commit the
updated files together with the change that produced them. This directory
must never be gitignored.

## Files

One file per suite per machine: `<suite>-<machine>.json`.

- `ladder-<machine>.json` — the acceptance ladder (T0–T8, the general-geometry
  rungs G0–G3, and the graded-grid guards), written by `cargo test -p cfd-core
  --test acceptance --test ladder -- --include-ignored` and `cargo test -p
  cfd-geom --test t0_rasterizer`.
- `grading-bench-<machine>.json` — the grid-grading benchmark, written by
  `cargo test -p cfd-ui grading_bench -- --include-ignored`.
- `contour-<machine>.json` — the contour-generator swap (cone equivalence,
  bell vs published geometry, RS-25 cone-vs-bell report), written by
  `cargo test -p cfd-ui contour` plus the slow before/after comparison
  `cargo test -p cfd-ui rs25_before_after -- --include-ignored`.
- `historical-presets-<machine>.json` — the historical engine presets
  (1943–1961) and their CEA-backed propellant classes: cone geometry, domain
  fit, separation at each preset's default altitude, the Redstone throat area
  against its published diameter, and the regression pin on the six earlier
  presets. Written by `cargo test -p cfd-ui historical_presets`, plus the
  run-time benchmark `cargo test -p cfd-ui historical_preset_cost --
  --include-ignored`.

One companion file here is **not** machine-keyed JSON and is not written by
`cfd-results`: `propellant-cea.md` is the output of `tools/propellant_cea.py`,
an offline thermochemistry computation whose result depends on the reactant
data and the CEA/cantera version, not on the hardware. It is a table, not a
measurement, so it has no `machine` and no pass/fail. Regenerate it with the
command recorded in its own header; the write-up that consumes it is
`historical-presets-v1.md`.

The **machine label is derived from the hardware** (CPU brand + logical core
count, slugged — e.g. `apple-m1-8c`), never from a flag or hostname, so the
same machine always writes the same file and results are diffable across
time. A file is one coherent snapshot of one commit on one machine: records
from a new commit start the file fresh; re-running a test at the same commit
replaces its row.

## Schema

Keep this stable so files can be diffed across time. Extend it here if you
must — do **not** invent a second format.

```json
{
  "commit": "git rev-parse HEAD at record time",
  "timestamp": "ISO 8601 UTC, last record wins",
  "machine": "hardware-derived slug, e.g. apple-m1-8c",
  "tests": [
    {
      "id": "T4",
      "name": "human-readable test name",
      "expected": "the pass criterion: a number for a plain threshold, a string for a band (\"0.35-0.70\", \">= 1.90\", \"45.344 +/- 1.5\")",
      "actual": "the measured value (number)",
      "units": "what the number is (\"L1(rho)\", \"deg\", \"floor activations\")",
      "pass": true
    }
  ],
  "benchmarks": [
    {
      "case": "what was solved",
      "setting": "the configuration within the case",
      "cells": 64000,
      "steps_per_sec": 108.0,
      "seconds_to_steady": 65.0
    }
  ],
  "notes": ["anything that needs a human — recorded caveats, known startup floor contact, criterion definitions"]
}
```

Conventions:

- `expected`/`actual` are JSON numbers when scalar, strings when a band or
  qualitative. Never put units inside the value — `units` carries them.
- `seconds_to_steady` is wall-clock seconds to the §9 *visual* steady
  criterion unless a note says otherwise.
- One `tests` entry per asserted quantity: a rung asserting three numbers
  writes three rows (`T7-beta`, `T7-p`, `T7-mach`).
- Recorded-but-not-asserted measurements go in `notes`, not in `tests` with a
  fake pass.
- A row measured on a flow that has **not** converged must say so. Assert on a
  mean over a **declared settled window**, never on one instant, and record in
  `notes`: the window (steps and non-dimensional time), the residual at the end
  against `RESIDUAL_CONVERGED`, `StepInfo::converged`, and the `Confidence` the
  report carries. A window that carries an assert also needs its own steadiness
  evidence — the drift of the window mean between its halves, and that the
  residual is not growing — as separate rows. The RS-25 20 km C_f comparison is
  the worked example: its single-instant ancestor read +0.0218 at step 5511 and
  the opposite sign before step ~2000.

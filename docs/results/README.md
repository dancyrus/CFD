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

# Documentation map

This directory separates **current runtime contracts** from **historical design records** so old plans cannot be mistaken for the code that runs today.

## Current / authoritative

- [`../README.md`](../README.md) — setup, commands, active U6/U9 scenario overview.
- [`KAGGLE.md`](KAGGLE.md) — clean Kaggle setup and preflight procedure.
- [`U6_PROVENANCE.md`](U6_PROVENANCE.md) — source/provenance table for U6/U9 parameters and research adaptations.
- [`PAPER_FIDELITY.md`](PAPER_FIDELITY.md) — paper-faithful versus adapted behavior.
- [`U6_U9_SCENARIO_AUDIT_2026-09-12.md`](U6_U9_SCENARIO_AUDIT_2026-09-12.md) — current scenario-flow audit and verification record.
- [`WANDB_DIAGNOSTICS.md`](WANDB_DIAGNOSTICS.md) — metrics/logging interpretation.
- [`FILE_GUIDE.md`](FILE_GUIDE.md) — source-tree guide.

## Legacy paper reproduction

- [`legacy/PAPER_MAPPING.md`](legacy/PAPER_MAPPING.md) — mapping for the historical heterogeneous `f*_m*` reproduction scenarios. These scenarios remain loadable but are not the active U6/U9 research sweep.

## Historical / superseded

Everything under [`archive/`](archive/) is retained only for design history. It may describe old action spaces, termination semantics, buffers, metrics, or scenario scope and is **not** a runtime contract. When archive text conflicts with current code or the authoritative documents above, the current code/tests and authoritative documents win.

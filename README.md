# Cross-Guided Dual-Scale Electricity Forecasting

Code and selected aggregate results for **Cross-Guided Dual-Scale Learning with a Dynamic Industry Graph for Next-Day Electricity Consumption Forecasting**.

This release contains the experiment implementation and the aggregate results corresponding to **Tables 5-16** of the supplied revised manuscript. It contains **no raw spreadsheets, prepared observations, dated true/predicted values, trained weights, manuscript PDF or image assets**.

## Forecasting protocol

- Target: next-day grid-side total electricity consumption, not a 96-step trajectory.
- Both branches use 42 historical days, ending on day t-1. Each day has 96 quarter-hourly power measurements.
- Main target dates: training 2024-02-12 to 2024-11-26 (289); validation 2024-11-27 to 2025-01-27 (62); testing 2025-01-28 to 2025-03-31 (63).
- Main and fusion comparisons retain all five seeds: **42, 123, 2024, 2025, 3407**.
- Module/graph controls use the matched three-seed subset **42, 123, 3407**, as labeled in their tables.
- Noise tests freeze model seed 42 and average metrics over perturbation seeds 101, 202 and 303. These are not additional training seeds.
- Input normalization and target transforms are fitted on training samples. The main target is normalized daily energy change, restored with its previous-day baseline before physical-unit conversion.
- Power is reported in **GW**; energy, MAE and RMSE in **GWh**; MAPE in **%**.

The main five-seed MAE is **16.314 +/- 0.618 GWh**. This is different from the three-seed ablation reference and the fixed-seed noise reference. The historical-window selection stage fixes learning rate 0.0005 and reports **validation** metrics in Table 5; subsequent model comparisons use validation-selected configurations and report **test** metrics.

## What is released

| Directory | Contents |
|---|---|
| `src/hubei_revision/` | Model, data-window construction, normalization, training, physical-unit recovery and analyses |
| `configs/` | Frozen protocol and local-baseline configurations |
| `scripts/` | Experiment entry points and aggregate-result export/verification |
| `paper_results/` | Only aggregate CSVs matching manuscript Tables 5-16 |
| `docs/` | Data schema, implementation provenance and reproduction instructions |
| `vendor/tslib/` | Version-pinned public reference source and its upstream license |

The CSV identifier `Ours` denotes Ours-DualScale. `Legacy_` identifiers denote the explicitly qualified local controls in Table 8. They are not authenticated reproductions of the historically named literature methods. See [implementation provenance](docs/MODEL_PROVENANCE.md).

## Reproduction

Use Python 3.10 and the pinned dependencies:

```bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

The original audited `model_space.csv` and `unit_audit.json` must be supplied locally in `data_private/` by an authorized data holder. They are intentionally absent from this repository. See [the field specification](docs/FEATURE_SCHEMA.md) and [reproduction instructions](docs/REPRODUCING.md). The code consumes the prepared schema; this release does not claim to rerun every historical Excel preparation step.

After authorized local data preparation:

```bash
python scripts/setup_reference_sources.py --source astgcn
python scripts/run_history_controls.py --dry-run
python scripts/run_history_controls.py
python scripts/smoke_and_audit.py
python scripts/run_suite.py --groups E01,E02,E03,E04,E05,E06,E08
python scripts/run_legacy_addendum.py
python scripts/run_analysis.py --groups E03,E07,E09,E10,E11
python scripts/summarize_results.py
python scripts/export_paper_results.py
```

The full experimental suite includes additional archived controls. Only the explicitly mapped aggregate tables are included in this concise release. ASTGCN-r is obtained using its pinned downloader and is not redistributed here. No additional images are needed to inspect the released tables.

To verify the public package without private observations or training dependencies:

```bash
python scripts/verify_release.py
```

The supplied results are archived experimental outputs; running the verification command checks released-file integrity, not new training. Reproduction on the original observations requires authorized data access and the recorded software/hardware environment.

## Paper correspondence

See [the table mapping](docs/PAPER_RESULTS_MAP.md) and `release_manifest.json` for the manuscript snapshot SHA256, numerical source hashes and result-file checksums. Only table-level aggregates are published. The selected six-seed illustration is not part of these aggregate tables or this release's evaluation protocol.

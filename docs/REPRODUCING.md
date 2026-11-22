# Reproduction instructions

## Data boundary

Private observations are not redistributed. An authorized holder must provide the audited `data_private/model_space.csv` and `data_private/unit_audit.json`. The feature meanings and all 23 industry/sales node names are documented in `FEATURE_SCHEMA.md`.

The CSV has one unique chronological row per observation day. It supplies the quarter-hourly power values, daily grid energy, industry/sales quantities, historical Wuhan temperatures and calendar features. The main experiment uses 456 observed days. Windows end on t-1, while the target and auxiliary labels refer to day t. The source observations and fitted artifacts must remain outside public version control.

The prepared internal convention is original numerical values divided by 1000. After target inversion and previous-day-baseline recovery, internal energy is multiplied by 10 exactly once to obtain GWh. Power has the corresponding conversion to GW. The fixed protocol's data SHA256 verifies the authorized original prepared file; replacement data require a separately versioned protocol and fresh training.

Observation dates do not supply actual publication or revision timestamps. The code's delay analyses are simulated conditions, not verification of a real operational publication schedule.

## Dependencies and sources

Use the dependency installation command in the main README. The recorded numerical environment used PyTorch 2.2.2 with CUDA 11.8 and an NVIDIA GeForce RTX 3070 Ti Laptop GPU. Download ASTGCN separately if executing the complete reference suite. TSLib files required by the archived implementations are already included with their upstream license.

## Experiment sequence

1. Place authorized data locally and inspect the frozen configurations.
2. Run the history-driver dry run and unit/temporal audit.
3. Run the historical-window study separately. It fixes learning rate 0.0005.
4. Run E01 validation selection, E02 statistical/tree references, E03 repeated seeds and E04-E06 controls.
5. Run E08 seasonal folds, each with its own pre-test normalization and validation selection.
6. Run the local comparison addendum and the post-training analyses.
7. Regenerate local aggregate tables with `summarize_results.py`, then export the paper subset with `export_paper_results.py`.

Do not reinitialize the frozen protocol after training begins. The complete suite contains extra archived controls; their existence does not mean they appear in every manuscript table. Five-seed, three-seed and fixed-checkpoint groups are kept separate.

Generated `results/`, `experiments/`, `tables/`, `reports/` and `manifests/` directories are local outputs and are ignored. Only the allowlisted aggregate CSVs in `paper_results/` are published. The file-integrity verifier can run without private data or numerical dependencies.

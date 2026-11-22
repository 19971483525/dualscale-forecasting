# Implementation provenance

The full proposed model is the checkpoint-compatible implementation used for the revised 42/42-day experiment suite. Its numerical source is preserved verbatim in this release and checked against the frozen protocol hashes.

The original model also contains a flat-history path. Removing a named branch does not remove all access to its historical variables. An unused legacy two-stage head remains registered in the original constructor; registered parameter counts and executed computation should be distinguished.

The six general forecasting references use the pinned [Time-Series-Library](https://github.com/thuml/Time-Series-Library) source at commit `4e938a1767106324dd753b2a44832bf870a0252e`, with the upstream license retained. Their adapter receives `[B,42,153]` daily sequences and seven target-day calendar features. The four-output readout predicts daily energy change and peak/valley/mean-power auxiliary statistics. PatchTST preserves channel-independent processing; its patch length is 12 historical **days**, with stride six days.

ASTGCN-r is an additional feature-graph adaptation of the authors' recent component, pinned to commit `2e7a4faa2a6f89da8d1cb37acb7e267c9bc87296`. It is retrieved separately by `setup_reference_sources.py`. Its graph uses training-only absolute correlations between 153 features, rather than physical electricity-grid topology.

## Retained local comparisons

These code labels match the qualified local implementations in manuscript Table 8. They are not authenticated reproductions of the original named papers.

| Local identifier | Actual retained backbone | Width |
|---|---|---:|
| MG-ASTL | ConvInception | 80 |
| TL-STLF | Two-block Mixer, without a transfer-learning stage | 80 |
| LoadFormer | Two-layer patch Transformer, seven-day patches | 80 |
| STGNN-Load | Two-layer inverted variable Transformer | 80 |
| ASTGCN-Load | ConvInception; separate from ASTGCN-r | 96 |
| GAT-LoadMixer | Three-block Mixer, without graph-attention layers | 72 |

Native reference and local comparison configurations are kept distinct. The paper's five-seed, matched three-seed and fixed-checkpoint perturbation results are labeled separately in the result mapping.

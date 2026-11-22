# Manuscript table mapping

The aggregate rows below were checked against the supplied revised manuscript snapshot. Only printed table-level results are included.

| Manuscript table | Released CSV | Rows | Statistical scope |
|---|---|---:|---|
| 5 | [`table_05_history_window_validation.csv`](../paper_results/table_05_history_window_validation.csv) | 7 | Validation only; five seeds; fixed learning rate 0.0005. |
| 6 | [`table_06_general_forecasting.csv`](../paper_results/table_06_general_forecasting.csv) | 7 | Test; all five training seeds; sample SD. |
| 7 | [`table_07_selected_configurations.csv`](../paper_results/table_07_selected_configurations.csv) | 7 | Selected configurations; hidden width 96 for these models. |
| 8 | [`table_08_local_deep_learning.csv`](../paper_results/table_08_local_deep_learning.csv) | 7 | Test; five seeds; local labels do not authenticate named-paper reproduction. |
| 9 | [`table_09_statistical_machine_learning.csv`](../paper_results/table_09_statistical_machine_learning.csv) | 9 | Test; Ours is the five-seed mean; other rows are individual baseline evaluations. |
| 10 | [`table_10_module_ablation.csv`](../paper_results/table_10_module_ablation.csv) | 10 | Test; matched seeds 42, 123, 3407; sample SD. |
| 11 | [`table_11_fusion_strategies.csv`](../paper_results/table_11_fusion_strategies.csv) | 9 | Test; all five seeds; complete registered parameter counts. |
| 12 | [`table_12_graph_topk.csv`](../paper_results/table_12_graph_topk.csv) | 6 | Validation and test are labeled separately; three matched seeds; density includes diagonals. |
| 13 | [`table_13_graph_relation_controls.csv`](../paper_results/table_13_graph_relation_controls.csv) | 6 | Test; three matched seeds; actual parameter counts, not asserted exact matches. |
| 14 | [`table_14_graph_operating_conditions.csv`](../paper_results/table_14_graph_operating_conditions.csv) | 7 | Test; matched three seeds; date subsets overlap where applicable; small subsets are descriptive. |
| 15 | [`table_15_seasonal_forecasting.csv`](../paper_results/table_15_seasonal_forecasting.csv) | 3 | Four 30-day next-day forecast windows; five seeds per model and fold; sample SD. |
| 16 | [`table_16_input_noise_robustness.csv`](../paper_results/table_16_input_noise_robustness.csv) | 13 | Fixed model seed 42; metrics averaged over three perturbation seeds, not training seeds. |

The paper identifier `Ours-DualScale` is stored as `Ours`; `Legacy_` prefixes identify the qualified local controls.
Numbers are exported to manuscript precision. This release does not publish individual dates, observed consumption or forecast traces.

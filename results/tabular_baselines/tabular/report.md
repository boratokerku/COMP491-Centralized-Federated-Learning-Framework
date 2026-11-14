# Comparison: tabular

Settings: `{"seeds": [42, 43, 44], "rounds": null, "baselines_only": true}`

| Case | F1 | ROC-AUC | PR-AUC | Accuracy |
|---|---|---|---|---|
| `centralized` | 0.507 ± 0.006 | 0.797 ± 0.003 | 0.466 ± 0.007 | 0.865 ± 0.002 |
| `local_only` | 0.446 ± 0.014 | 0.769 ± 0.007 | 0.410 ± 0.009 | 0.844 ± 0.009 |
| `local_only_client_test` | 0.431 ± 0.016 | 0.751 ± 0.005 | 0.383 ± 0.009 | 0.836 ± 0.012 |

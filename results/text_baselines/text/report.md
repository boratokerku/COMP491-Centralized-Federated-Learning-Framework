# Comparison: text

Settings: `{"seeds": [42, 43, 44], "rounds": null, "baselines_only": true}`

| Case | F1 | ROC-AUC | PR-AUC | Accuracy |
|---|---|---|---|---|
| `centralized` | 0.854 ± 0.011 | 0.985 ± 0.002 | 0.924 ± 0.024 | 0.961 ± 0.003 |
| `local_only` | 0.769 ± 0.030 | 0.969 ± 0.006 | 0.837 ± 0.043 | 0.935 ± 0.007 |
| `local_only_client_test` | 0.752 ± 0.011 | 0.972 ± 0.000 | 0.833 ± 0.030 | 0.929 ± 0.006 |

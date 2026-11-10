# Data

## Tabular dataset: UCI Bank Marketing

| Field | Value |
|---|---|
| Source | https://archive.ics.uci.edu/dataset/222/bank+marketing |
| License | CC BY 4.0 |
| Download date | 2026-10-26 |
| File used | `bank-additional-full.csv` from `bank-additional.zip` (`;` separated) |
| SHA-256 | `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8` |
| Local path | `data/tabular/raw/bank-additional-full.csv` (gitignored) |
| Dataset citation | Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5K306 |
| Paper | Moro, S., Cortez, P., Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems 62:22–31. doi:10.1016/j.dss.2014.03.001 |

The UCI zip contains two versions:
- `bank.zip` → `bank-full.csv`: 45,211 rows, 16 features. These are the numbers shown on the UCI page.
- `bank-additional.zip` → `bank-additional-full.csv`: 41,188 rows, 20 features, including 5 socio-economic indicators. **We use this one.** It is the file analysed in the paper.

Download (from the repository root; the data is never committed):
```bash
python backend/scripts/download_bank_marketing.py
```
The script downloads the archive from UCI, verifies the SHA-256 of the archive and of the CSV, and saves only `bank-additional-full.csv` to `data/tabular/raw/`.

This is a placeholder. The dataset may later be swapped for the advisors' bank data.
To do that, add a new config under `backend/configs/datasets/`. The preprocessing code stays the same.

## EDA summary

Reproduce with: `python backend/scripts/eda_bank_marketing.py`

- **Samples:** 41,188. **Columns:** 21 (20 features + label `y`)
- **Types:** 10 numeric (5 int, 5 float) and 10 categorical
- **Missing values:** there are no NaNs, but categorical columns use the `"unknown"` token:

  | column | unknown |
  |---|---|
  | default | 8,597 |
  | education | 1,731 |
  | housing | 990 |
  | loan | 990 |
  | job | 330 |
  | marital | 80 |

  Decision: we keep `"unknown"` as its own category and do not impute it.
- **Label (`y`):** no 36,548 (88.7%) and yes 4,640 (11.3%). The classes are **imbalanced**,
  so we report F1/AUC rather than accuracy alone.
- **Categorical values:**
  - job (12): admin., blue-collar, technician, services, management, retired, entrepreneur, self-employed, housemaid, unemployed, student, unknown
  - marital (4): married, single, divorced, unknown
  - education (8): university.degree, high.school, basic.9y, professional.course, basic.4y, basic.6y, unknown, illiterate
  - default (3): no, unknown, yes (only 3 rows are "yes")
  - housing / loan (3): yes, no, unknown
  - contact (2): cellular, telephone
  - month (10): mar–dec (there are no jan or feb rows)
  - day_of_week (5): mon–fri
  - poutcome (3): nonexistent, failure, success

### Notes and decisions
- **`duration` is dropped.** Its value is only known after the call ends, so it leaks the label.
- **`pdays = 999`** means the client was not contacted before (about 96% of rows). It is treated as a sentinel: 999 is replaced by the mean and an extra column `pdays_is_sentinel` (0/1) is added.
- **Categories are fixed in the config.** In FL, every client must produce the same one-hot feature space,
  so encoders use the category lists written in the config instead of learning them from local data.

## Client partitions (proposal T1.4 / D1.5)

Reproduce with: `python backend/scripts/partition_tabular.py --num-clients 4 --seed 42`
(code: shared partitioner `backend/src/cfl/tasks/partition.py`, config wrapper `backend/src/cfl/tasks/tabular/partition.py`).

- The rows are shuffled with the seed first (the file is ordered by date).
- A **stratified 20% test set** is held out for experiments (server-side evaluation and baselines). It is never part of any client's data.
- The remaining 80% is split into **4 IID, label-stratified** client partitions.
- Each client partition is split again into **train (70%), val (10%) and test (20%)**, all stratified:
  - `train`: local training.
  - `val`: per-round federated evaluation of the received global model and local early stopping / tuning.
  - `test`: final per-client metrics; not used for any decision during training.
- Files are written to `data/tabular/partitions/bank_marketing/` (gitignored): `client_<i>/train.csv`, `client_<i>/val.csv`, `client_<i>/test.csv`, `test.csv` and `summary.json`.
- The same seed always produces the same partitions (covered by `backend/tests/test_tabular_partition.py`).
- The partitions hold raw rows. Pre-processing runs on each client, following `backend/configs/datasets/bank_marketing.yaml`.
- Each client folder is copied only to its own client machine and mounted into the client container. Raw data is never sent to the gateway or the coordinator.

| split | rows | positive (yes) | ratio |
|---|---|---|---|
| client_0 train / val / test | 5,766 / 824 / 1,648 | 649 / 93 / 186 | 0.1126 / 0.1129 / 0.1129 |
| client_1 train / val / test | 5,766 / 824 / 1,648 | 649 / 93 / 186 | 0.1126 / 0.1129 / 0.1129 |
| client_2 train / val / test | 5,765 / 824 / 1,648 | 649 / 93 / 186 | 0.1126 / 0.1129 / 0.1129 |
| client_3 train / val / test | 5,765 / 824 / 1,648 | 649 / 93 / 186 | 0.1126 / 0.1129 / 0.1129 |
| held-out test | 8,238 | 928 | 0.1126 |

A non-IID partition (label/feature skew) is optional (T7.2) and will be added later.

## Text dataset: UCI SMS Spam Collection

| Field | Value |
|---|---|
| Source | https://archive.ics.uci.edu/dataset/228/sms+spam+collection |
| License | CC BY 4.0 |
| Download date | 2026-11-04 |
| File used | `SMSSpamCollection` from `sms+spam+collection.zip` (one message per line: `<ham\|spam>\t<text>`, no header, no quoting) |
| SHA-256 (zip) | `1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3` |
| SHA-256 (file) | `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d` |
| Local path | `data/text/raw/SMSSpamCollection` (gitignored) |
| Dataset citation | Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI ML Repository. https://doi.org/10.24432/C5CC84 |

Download (from the repository root; the data is never committed):
```bash
python backend/scripts/download_sms_spam.py
```

- **Samples:** 5,574 messages; ham 4,827 (86.6%), spam 747 (13.4%). Longest message: 910 characters.
- **Duplicates removed:** 403 messages are exact copies of an earlier one (mostly spam templates); no text appears with both labels. Without de-duplication the same message could land in a client's train split and in a test split and inflate results. After de-duplication: **5,171 messages**, ham 4,518, spam 653 (12.6%).
- Pre-processing is the hashing tokenizer in `backend/src/cfl/tasks/text/tokenizer.py` (contract: `planning/examples/sms_spam_manifest.json`). Nothing is fitted on data, so no vocabulary has to be shared.

### Client partitions

Reproduce with: `python backend/scripts/partition_text.py --num-clients 4 --seed 42`
(code: `backend/src/cfl/tasks/text/data.py` and the shared partitioner `backend/src/cfl/tasks/partition.py`).

Same scheme as the tabular dataset, applied after de-duplication: stratified 20% held-out test set, the rest split into 4 IID label-stratified clients, each split into train (70%) / val (10%) / test (20%). Files go to `data/text/partitions/sms_spam/` (gitignored) as CSV with `label,text` columns.

| split | rows | positive (spam) | ratio |
|---|---|---|---|
| client_0 train / val / test | 723 / 104 / 207 | 91 / 13 / 26 | 0.1259 / 0.1250 / 0.1256 |
| client_1 train / val / test | 723 / 104 / 207 | 91 / 13 / 26 | 0.1259 / 0.1250 / 0.1256 |
| client_2 train / val / test | 723 / 104 / 207 | 92 / 13 / 26 | 0.1272 / 0.1250 / 0.1256 |
| client_3 train / val / test | 723 / 104 / 207 | 92 / 13 / 26 | 0.1272 / 0.1250 / 0.1256 |
| held-out test | 1,035 | 131 | 0.1266 |

The dataset is small: each client's val split holds only 13 spam messages, so per-client val metrics are noisy. Report per-client results with that in mind, and prefer the held-out test set for comparisons between runs.

### Initial federated text hyperparameter screen

Reproduce with `python backend/scripts/tune_text.py` (or, from `backend/`,
`python scripts/tune_text.py`). The script changes one setting at a time from the baseline,
uses the same seed-42 data partitions, and runs seeds 42, 43, and 44 for 20 rounds each.
It records final held-out F1, ROC-AUC, PR-AUC, accuracy, best round F1, and an F1 stability
round. Stability is the first round after which every remaining global F1 is within 0.005
of that run's final F1. Raw run records and aggregate results are in
`results/text_tuning/runs.csv`, `results/text_tuning/summary.csv`, and
`results/text_tuning/results.json`.

| Setting | Learning rate | Local epochs | Batch size | Class weights | Final F1 (mean ± SD) | ROC-AUC (mean ± SD) | PR-AUC (mean ± SD) | Accuracy (mean ± SD) | Stable round (mean) |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| Baseline | 0.010 | 1 | 64 | [1.0, 6.9] | 0.862 ± 0.010 | 0.988 ± 0.001 | 0.937 ± 0.015 | 0.963 ± 0.003 | 18.7 |
| Learning rate 0.005 | 0.005 | 1 | 64 | [1.0, 6.9] | 0.853 ± 0.012 | 0.988 ± 0.002 | 0.944 ± 0.010 | 0.960 ± 0.003 | 18.3 |
| Local epochs 2 | 0.010 | 2 | 64 | [1.0, 6.9] | 0.865 ± 0.012 | 0.986 ± 0.001 | 0.908 ± 0.020 | 0.964 ± 0.003 | 14.0 |
| Batch size 32 | 0.010 | 1 | 32 | [1.0, 6.9] | 0.863 ± 0.004 | 0.986 ± 0.001 | 0.924 ± 0.008 | 0.963 ± 0.002 | 20.0 |
| Spam weight 5.0 | 0.010 | 1 | 64 | [1.0, 5.0] | **0.868 ± 0.011** | 0.988 ± 0.001 | 0.940 ± 0.015 | **0.965 ± 0.003** | 17.0 |

All 15 runs completed all 20 rounds and all four clients verified the final model. The
weight-5 setting had the highest mean final F1 and accuracy in this small screen, so it is
the provisional candidate in `sim_text.yaml` and `demo_text.yaml`. This is exploratory:
each setting has only three seeds and all runs reuse one fixed partition. Because these
metrics use the held-out test set, the comparison should not be treated as an unbiased final
test after selecting a setting. Re-evaluate the selected configuration on a fresh test set
or use validation metrics for future tuning; report the test-set reuse as a limitation.

### SMS training baselines

Run just the baselines, without repeating the federated runs, from the repository root with
`python backend/scripts/run_comparison.py --tasks text --baselines-only --out results/text_baselines`.
The shared comparison runner reports two local-training baselines against the server's
held-out test set: one centralized model trained
on the concatenated client training splits, and one local model per client trained only on that
client's training split. It also evaluates each local model on that same client's private test
split. The `local_only_client_test` case in `results/text_baselines/text/results.json` contains
the mean metrics across clients plus each client's individual metrics; the report and CSV include the mean,
standard deviation across seeds, and worst/best client F1. The two local-only test
views answer different questions: held-out test scores are directly comparable across methods,
while client-test scores show performance on each client's own distribution. Client test scores
must not be pooled with the server held-out scores or interpreted as an additional independent
test set.

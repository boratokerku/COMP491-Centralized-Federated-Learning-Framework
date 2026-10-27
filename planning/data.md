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
(code: `backend/src/cfl/tasks/tabular/partition.py`).

- The rows are shuffled with the seed first (the file is ordered by date).
- A **stratified 20% test set** is held out for experiments (server-side evaluation and baselines). It is never part of any client's data.
- The remaining 80% is split into **4 IID, label-stratified** client partitions.
- Each client partition is split again into **train (80%) and a local test split (20%)**, used for federated evaluation.
- Files are written to `data/tabular/partitions/bank_marketing/` (gitignored): `client_<i>/train.csv`, `client_<i>/test.csv`, `test.csv` and `summary.json`.
- The same seed always produces the same partitions (covered by `backend/tests/test_tabular_partition.py`).
- The partitions hold raw rows. Pre-processing runs on each client, following `backend/configs/datasets/bank_marketing.yaml`.
- Each client folder is copied only to its own client machine and mounted into the client container. Raw data is never sent to the gateway or the coordinator.

| split | rows | positive (yes) | ratio |
|---|---|---|---|
| client_0 train / local test | 6,590 / 1,648 | 742 / 186 | 0.1126 / 0.1129 |
| client_1 train / local test | 6,590 / 1,648 | 742 / 186 | 0.1126 / 0.1129 |
| client_2 train / local test | 6,589 / 1,648 | 742 / 186 | 0.1126 / 0.1129 |
| client_3 train / local test | 6,589 / 1,648 | 742 / 186 | 0.1126 / 0.1129 |
| held-out test | 8,238 | 928 | 0.1126 |

A non-IID partition (label/feature skew) is optional (T7.2) and will be added later.

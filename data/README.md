# Data Directory (`data/`)

This directory stores local datasets, raw downloads, and simulated client partitions for the Centralized Federated Learning (CFL) framework.

> **CRITICAL SECURITY RULE:** All data files and partitions in this directory (except this `README.md`) are gitignored. Raw private data and client partitions must **never** be committed to the repository.

---

## Directory Structure

```
data/
├── tabular/
│   ├── raw/                      # Downloaded raw dataset files (e.g. bank-additional-full.csv)
│   └── partitions/               # Partitioned client silos and server test set
│       └── bank_marketing/       # Output for bank_marketing dataset
│           ├── client_0/
│           │   ├── train.csv     # Client 0 private local training set (80%)
│           │   └── test.csv      # Client 0 local evaluation test set (20%)
│           ├── client_1/
│           │   ├── train.csv
│           │   └── test.csv
│           ├── ...
│           ├── client_{N-1}/
│           ├── test.csv          # Server-side held-out test set (20% of full data)
│           └── summary.json      # Partition settings & per-client class balance
└── image/                        # Reserved for second modality (MedMNIST)
```

---

## 1. Downloading Tabular Data (UCI Bank Marketing)

To download the primary tabular dataset (`bank-additional-full.csv`), run the download script from the repository root:

```bash
python backend/scripts/download_bank_marketing.py
```

The script:
1. Downloads the official archive from the UCI Machine Learning Repository.
2. Verifies the SHA-256 checksum of the archive (`74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8`).
3. Extracts and places `bank-additional-full.csv` (41,188 rows, 21 columns) into `data/tabular/raw/`.

Dataset license, citations, and EDA details are documented in [`planning/data.md`](../planning/data.md).

---

## 2. Seeded IID Client Partitioning (`partition_tabular.py`)

The partition script (`backend/scripts/partition_tabular.py`) splits the tabular dataset into:
1. **Server-Side Held-Out Test Set (`test.csv`):** Reserved strictly for server-side evaluation and baselines (default: 20% of total rows, label-stratified). This data is never sent to any client.
2. **IID Client Partitions (`client_<i>/`):** The remaining 80% is partitioned equally among $N$ clients using stratified sampling to preserve identical class ratios.
3. **Local Train/Test Splits:** Each client's partition is further split into **local train (80%)** and **local test (20%)** for federated evaluation.

The script is entirely **config-driven** via YAML (e.g. `backend/configs/datasets/bank_marketing.yaml`). It contains zero hardcoded dataset logic, ensuring immediate compatibility when switching to new banking datasets.

### Basic Usage

From the repository root:

```bash
python backend/scripts/partition_tabular.py --num-clients 4 --seed 42
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|:---|:---|:---|:---|
| `--config` | `Path` | `backend/configs/datasets/bank_marketing.yaml` | Path to dataset configuration YAML |
| `--num-clients` | `int` | `4` | Number of client organizations to simulate |
| `--seed` | `int` | `42` | Random seed for deterministic reproducibility |
| `--test-size` | `float` | `0.2` | Fraction of total data reserved for server held-out test set |
| `--local-test-size` | `float` | `0.2` | Fraction of each client partition reserved for local test set |
| `--out` | `Path` | `data/tabular/partitions/<dataset>` | Destination directory for partition CSVs |

### Example Console Summary Output

When executed, the script prints a formatted summary table of sample counts and class balance:

```
==============================================================================
Tabular IID Partition Summary: bank_marketing (seed=42, clients=4)
==============================================================================
Partition                       Total Rows    Positive Class    Positive Ratio
------------------------------------------------------------------------------
client_0 (train)                     6,590               742            11.26%
client_0 (local test)                1,648               186            11.29%
client_1 (train)                     6,590               742            11.26%
client_1 (local test)                1,648               186            11.29%
client_2 (train)                     6,589               742            11.26%
client_2 (local test)                1,648               186            11.29%
client_3 (train)                     6,589               742            11.26%
client_3 (local test)                1,648               186            11.29%
------------------------------------------------------------------------------
server (held-out test)               8,238               928            11.26%
==============================================================================
Total partitioned rows:             41,188
Partitions written to:              data/tabular/partitions/bank_marketing
==============================================================================
```

### Reproducibility & Testing

The partitioning is completely deterministic. Running the script multiple times with the same `--seed` produces identical row assignments and files.

Run automated tests to verify partitioning correctness and reproducibility:

```bash
pytest backend/tests/test_tabular_partition.py
```

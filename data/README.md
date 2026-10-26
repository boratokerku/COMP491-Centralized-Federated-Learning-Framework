# data/

Local datasets and client partitions go here. Everything in this folder except this README is
gitignored: raw data must never be committed.

Suggested layout:

```
data/
  tabular/
    raw/            # original download
    partitions/     # client_0.csv, client_1.csv, ... + test.csv
  image/
```

Tabular data (UCI Bank Marketing): run `python backend/scripts/download_bank_marketing.py`
from the repository root. Source, license, checksum and EDA: [planning/data.md](../planning/data.md).

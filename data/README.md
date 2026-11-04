# data/

Local datasets and client partitions go here. Everything in this folder except this README is
gitignored: raw data must never be committed.

Suggested layout:

```
data/
  tabular/
    raw/            # original download
    partitions/     # <dataset>/client_<i>/{train,val,test}.csv + test.csv + summary.json
  text/
    raw/            # SMSSpamCollection
    partitions/     # sms_spam/client_<i>/{train,val,test}.csv + test.csv + summary.json
```

Tabular data (UCI Bank Marketing): run `python backend/scripts/download_bank_marketing.py`
from the repository root, then `python backend/scripts/partition_tabular.py`.

Text data (UCI SMS Spam Collection): run `python backend/scripts/download_sms_spam.py`, then
`python backend/scripts/partition_text.py`.

Source, license, checksums, EDA and partition tables: [planning/data.md](../planning/data.md).

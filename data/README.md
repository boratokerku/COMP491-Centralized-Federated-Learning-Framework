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

Record each dataset's source, license and download date in the team's shared notes.

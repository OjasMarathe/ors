# Note on file sizes

Two files are large enough that GitHub will warn on push:

- `01-supreme-court/data/sc_judgments_metadata.csv` — 16 MB
- `03-tipitaka/corpus/tipitaka_sutta.txt` — 16 MB

Both are well under GitHub's 100 MB hard limit, so a normal push works. If you would
rather keep the repository light, either track them with Git LFS:

```
git lfs track "*.csv" "03-tipitaka/corpus/*.txt"
```

or gzip them and adjust the loader paths — every consumer of these files reads them
line by line, so switching to `gzip.open` is a one-line change.

Rank tables over 1 MB in `rank-tables/` are already gzipped. They are fully
regenerable from the corpora, so deleting them costs nothing but a rerun.

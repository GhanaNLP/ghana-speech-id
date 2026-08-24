"""Pull id/language/ipa/duration/split from ghana-speech-ipa without fetching the audio.

The parquet files embed FLAC bytes in `audio` (~50 GB). Parquet is columnar, so projecting
the text columns turns this into HTTP range requests over those column pages only. DuckDB's
httpfs does that pushdown well and parallelises across files; pyarrow-over-HfFileSystem did
not (minutes per file vs seconds).
"""
import time

import duckdb
from huggingface_hub import get_token

REPO = "ghananlpcommunity/ghana-speech-ipa"
OUT = "/mnt/volume_d2wey28/projects/ghana-speech-id/data/ipa_text.parquet"

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET enable_progress_bar=false;")
tok = get_token()
if tok:
    con.execute(f"CREATE SECRET hf1 (TYPE huggingface, TOKEN '{tok}');")

glob = f"hf://datasets/{REPO}/**/*.parquet"
t0 = time.time()
con.execute(f"""
COPY (
  SELECT id, language, ipa, duration,
         -- the split lives in the file name (train-000NN / validation-000NN), not a column
         CASE WHEN regexp_matches(filename, 'validation') THEN 'validation' ELSE 'train' END AS split
  FROM read_parquet('{glob}', filename=true, union_by_name=true)
) TO '{OUT}' (FORMAT parquet, COMPRESSION zstd);
""")
print(f"copied in {time.time()-t0:.0f}s", flush=True)

r = con.execute(f"""
  SELECT split, count(*) n, count(DISTINCT language) langs,
         round(sum(duration)/3600, 1) hours
  FROM read_parquet('{OUT}') GROUP BY split ORDER BY split
""").fetchall()
for row in r:
    print(f"  {row[0]:12} rows={row[1]:>7}  langs={row[2]:>3}  hours={row[3]}")

print("\nrows per language:")
for lang, n, h, mu in con.execute(f"""
  SELECT language, count(*) n, round(sum(duration)/3600,1) h,
         round(avg(length(ipa) - length(replace(ipa,' ','')) + 1),1) mean_units
  FROM read_parquet('{OUT}') GROUP BY language ORDER BY n DESC
""").fetchall():
    print(f"  {lang:26} {n:>7}  {h:>7} h  {mu:>6} units/clip")

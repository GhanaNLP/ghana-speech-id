import duckdb
c = duckdb.connect()
F = "data/eval_ipa_gh.parquet"
q = c.execute(f"""
  SELECT CASE WHEN duration < 4 THEN 'a: <4s'
              WHEN duration < 7 THEN 'b: 4-7s'
              WHEN duration < 10 THEN 'c: 7-10s'
              WHEN duration < 15 THEN 'd: 10-15s'
              WHEN duration < 25 THEN 'e: 15-25s'
              ELSE 'f: >25s' END AS bucket,
         count(*) AS n,
         round(avg(duration),1) AS mean_s,
         round(avg(len(string_split(trim(ipa),' '))),1) AS mean_units,
         round(avg(len(string_split(trim(ipa),' '))/duration),2) AS ups,
         round(100.0*sum(CASE WHEN len(string_split(trim(ipa),' ')) < 3 THEN 1 ELSE 0 END)/count(*),1) AS pct_empty
  FROM read_parquet('{F}') GROUP BY bucket ORDER BY bucket
""").fetchall()
print(f"{'bucket':11} {'n':>6} {'mean_s':>7} {'units':>7} {'units/s':>8} {'% <3 units':>11}")
for b, n, ms, mu, ups, pe in q:
    print(f"{b:11} {n:6d} {ms:7.1f} {mu:7.1f} {ups:8.2f} {pe:10.1f}%")
print("\nreal speech runs 8-13 units/s (ghana-ipa-asr batch.py)")

print("\nsame check on the English decode, which WAS cropped to ~6.9s before decoding:")
q2 = c.execute("""
  SELECT count(*) n, round(avg(duration),1) s, round(avg(len(string_split(trim(ipa),' '))),1) u,
         round(avg(len(string_split(trim(ipa),' '))/duration),2) ups,
         round(100.0*sum(CASE WHEN len(string_split(trim(ipa),' ')) < 3 THEN 1 ELSE 0 END)/count(*),1) pe
  FROM read_parquet('data/english_ipa_gh.parquet')
""").fetchone()
print(f"  n={q2[0]} mean {q2[1]}s  {q2[2]} units  {q2[3]} units/s  {q2[4]}% under 3 units")

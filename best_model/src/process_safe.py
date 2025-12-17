from __future__ import annotations

import random
import re
import zlib
from pathlib import Path


MODE = "BUILD"


MIN_RATIO = 0.24


ANALYZE_N_FILES = 20000
SEED = 0


MAX_OUTPUT_GB = 2.0


MAX_FILE_BYTES = 5_000_000


TRIM_LEADING_REST_LINES = True
MAX_LEADING_REST_LINES = 32


PRINT_EVERY = 500


RAW_ABC_DIR = Path("data/abc")
OUT_DIR = Path("data/V3")

TRAIN_TXT = OUT_DIR / "train.txt"
STATS_TSV = OUT_DIR / "gzip_stats.tsv"
SAMPLES_DIR = OUT_DIR / "sanity_samples"

SONG_DELIMITER = "\n<|endoftext|>\n"


REST_ONLY = re.compile(r"^[zZ0-9/|:\[\]()<>\-]+$")


def iter_abc_paths(root: Path):
    yield from root.rglob("*.abc")


def reservoir_sample_paths(root: Path, k: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    sample: list[Path] = []
    n = 0

    print(f"[INDEX] Walking {root.as_posix()} to sample {k} files...")

    for p in iter_abc_paths(root):
        n += 1
        if len(sample) < k:
            sample.append(p)
        else:
            j = rng.randint(1, n)
            if j <= k:
                sample[j - 1] = p

        if n % (PRINT_EVERY * 20) == 0:
            print(f"[INDEX] Seen {n} files...")

    print(f"[INDEX] Total .abc files found: {n}")
    return sample


def keep_and_normalize_line(line: str) -> tuple[str | None, bool]:
    """
    Returns (normalized_line_or_None, was_T_removed)

    Keep:
      - all voices/music
      - all %% directives (%%MIDI etc.)
    Remove:
      - ALL T: lines
      - single-% comments (but keep %%)
      - blank lines
      - literal "..." lines
      - trailing "\" (line-wrap artifact)
    """
    s = line.strip()
    if not s:
        return None, False


    if s == "...":
        return None, False


    if s.startswith("T:"):
        return None, True


    if s.startswith("%") and not s.startswith("%%"):
        return None, False


    if s.endswith("\\"):
        s = s[:-1].rstrip()
        if not s:
            return None, False

    return s, False


def is_rest_only_music_line(s: str) -> bool:
    """
    Identify rest-only *music* lines (not headers).
    We only use this at the very beginning to reduce dead-air intros.
    """

    if re.match(r"^[A-Za-z]:", s):
        return False
    if s.startswith("%%") or s.startswith("V:"):
        return False

    t = "".join(s.split())
    if not t:
        return False

    if "z" not in t.lower():
        return False
    return bool(REST_ONLY.match(t))


def compute_gzip_ratio_stream(path: Path) -> tuple[float, int, int, int, bool]:
    """
    Memory-safe streaming gzip ratio on cleaned+normalized text:
      ratio = gzip_len / raw_len

    IMPORTANT: entropy bytes are whitespace-stripped so junk whitespace
    can't artificially raise ratio and sneak through the filter.
    """
    comp = zlib.compressobj(level=9, wbits=31)

    raw_len = 0
    gz_len = 0
    t_removed = 0
    seen_any = False

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s, was_T = keep_and_normalize_line(line)
            if was_T:
                t_removed += 1
            if s is None:
                continue


            s_ent = "".join(s.split())
            if not s_ent:
                continue

            b = (s_ent + "\n").encode("utf-8", errors="replace")
            raw_len += len(b)
            gz_len += len(comp.compress(b))
            seen_any = True

    gz_len += len(comp.flush())

    if not seen_any or raw_len == 0:
        return 0.0, 0, gz_len, t_removed, True

    return (gz_len / raw_len), raw_len, gz_len, t_removed, False


def write_cleaned_abc(path: Path, out_path: Path, ratio: float):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    leading_rest_skipped = 0
    trimming = TRIM_LEADING_REST_LINES

    with out_path.open("w", encoding="utf-8", newline="\n") as out, \
         path.open("r", encoding="utf-8", errors="replace") as f:

        out.write(f"% source: {path.as_posix()}\n")
        out.write(f"% gzip_ratio: {ratio:.4f}\n")
        out.write("% NOTE: T: removed; %% directives preserved; entropy normalized.\n\n")

        for line in f:
            s, _ = keep_and_normalize_line(line)
            if s is None:
                continue

            if trimming and is_rest_only_music_line(s):
                leading_rest_skipped += 1
                if leading_rest_skipped <= MAX_LEADING_REST_LINES:
                    continue

            if trimming and not is_rest_only_music_line(s):
                trimming = False

            out.write(s + "\n")


def quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(round(q * (len(sorted_vals) - 1)))
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return sorted_vals[idx]


def run_analyze():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    paths = reservoir_sample_paths(RAW_ABC_DIR, ANALYZE_N_FILES, SEED)
    print(f"[ANALYZE] Processing {len(paths)} sampled files...")

    ratios: list[float] = []
    records: list[tuple[float, Path]] = []

    unreadable = 0
    empty = 0
    skipped_large = 0
    t_removed_total = 0

    for i, p in enumerate(paths, 1):
        try:
            if MAX_FILE_BYTES is not None and p.stat().st_size > MAX_FILE_BYTES:
                skipped_large += 1
                continue

            r, raw_b, gz_b, t_removed, is_empty = compute_gzip_ratio_stream(p)
            t_removed_total += t_removed

            if is_empty:
                empty += 1
                continue

            ratios.append(r)
            records.append((r, p))

        except Exception:
            unreadable += 1
            continue

        if i % PRINT_EVERY == 0:
            kept_so_far = sum(1 for x in ratios if x >= MIN_RATIO)
            print(f"[ANALYZE] {i}/{len(paths)} | usable={len(ratios)} | kept@{MIN_RATIO:.2f}={kept_so_far} | skipped_large={skipped_large}")

    ratios_sorted = sorted(ratios)
    usable = len(ratios)
    kept = sum(1 for r in ratios if r >= MIN_RATIO)

    print("\n=== V3 ANALYZE SUMMARY ===")
    print(f"Sampled files:         {len(paths)}")
    print(f"Usable songs:          {usable}")
    print(f"Unreadable/Errors:     {unreadable}")
    print(f"Empty after clean:     {empty}")
    if MAX_FILE_BYTES is None:
        print("Skipped huge files:    disabled")
    else:
        print(f"Skipped huge files:    {skipped_large} (>{MAX_FILE_BYTES} bytes)")
    print(f"T: lines removed:      {t_removed_total}")
    print(f"MIN_RATIO:             {MIN_RATIO:.3f}")
    print(f"Kept fraction:         {kept}/{max(1, usable)}  ({(kept/max(1, usable)*100):.1f}%)")
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        print(f"  p{int(q*100):02d}: {quantile(ratios_sorted, q):.4f}")


    if records:
        records.sort(key=lambda x: x[0])

        for j, (r, p) in enumerate(records[:5]):
            write_cleaned_abc(p, SAMPLES_DIR / "dropped_low_ratio" / f"worst_{j:02d}.abc", r)

        for j, (r, p) in enumerate(records[-5:][::-1]):
            write_cleaned_abc(p, SAMPLES_DIR / "kept_high_ratio" / f"best_{j:02d}.abc", r)

        below = None
        above = None
        for r, p in records:
            if r < MIN_RATIO:
                below = (r, p)
            elif above is None and r >= MIN_RATIO:
                above = (r, p)

        if below is not None:
            r, p = below
            write_cleaned_abc(p, SAMPLES_DIR / "threshold_band" / "closest_below_threshold.abc", r)
        if above is not None:
            r, p = above
            write_cleaned_abc(p, SAMPLES_DIR / "threshold_band" / "closest_above_threshold.abc", r)

        print(f"\n[ANALYZE] Wrote sanity samples to: {SAMPLES_DIR.as_posix()}")
        print("[ANALYZE] Re-check threshold_band samples in abcjs now (should parse).")


def run_build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    max_bytes = None
    if MAX_OUTPUT_GB is not None:
        max_bytes = int(MAX_OUTPUT_GB * (1024 ** 3))

    total = 0
    kept = 0
    dropped = 0
    unreadable = 0
    empty = 0
    skipped_large = 0
    t_removed_total = 0

    print(f"[BUILD] Writing to {TRAIN_TXT.as_posix()}")
    print(f"[BUILD] MIN_RATIO={MIN_RATIO:.3f} | MAX_OUTPUT_GB={MAX_OUTPUT_GB}")

    with STATS_TSV.open("w", encoding="utf-8", newline="\n") as f_stats, \
         TRAIN_TXT.open("w", encoding="utf-8", newline="\n") as f_out:

        f_stats.write("path\tratio\traw_bytes\tgz_bytes\tkept\n")

        for p in iter_abc_paths(RAW_ABC_DIR):
            total += 1

            try:
                if MAX_FILE_BYTES is not None and p.stat().st_size > MAX_FILE_BYTES:
                    skipped_large += 1
                    continue

                r, raw_b, gz_b, t_removed, is_empty = compute_gzip_ratio_stream(p)
                t_removed_total += t_removed

                if is_empty:
                    empty += 1
                    continue

                is_kept = (r >= MIN_RATIO)
                f_stats.write(f"{p.as_posix()}\t{r:.6f}\t{raw_b}\t{gz_b}\t{1 if is_kept else 0}\n")

                if not is_kept:
                    dropped += 1
                    continue


                if max_bytes is not None and f_out.tell() >= max_bytes:
                    print("[BUILD] Reached MAX_OUTPUT_GB cap. Stopping.")
                    break


                leading_rest_skipped = 0
                trimming = TRIM_LEADING_REST_LINES

                with p.open("r", encoding="utf-8", errors="replace") as fin:
                    wrote_any = False
                    for line in fin:
                        s, _ = keep_and_normalize_line(line)
                        if s is None:
                            continue

                        if trimming and is_rest_only_music_line(s):
                            leading_rest_skipped += 1
                            if leading_rest_skipped <= MAX_LEADING_REST_LINES:
                                continue
                        if trimming and not is_rest_only_music_line(s):
                            trimming = False

                        f_out.write(s + "\n")
                        wrote_any = True

                if wrote_any:
                    f_out.write(SONG_DELIMITER)
                    kept += 1
                else:
                    empty += 1

            except Exception:
                unreadable += 1
                continue

            if total % PRINT_EVERY == 0:
                print(f"[BUILD] seen={total} kept={kept} dropped={dropped} skipped_large={skipped_large} out_bytes={f_out.tell()}")

    print("\n=== V3 BUILD SUMMARY ===")
    print(f"Total files seen:      {total}")
    print(f"Kept songs:            {kept}")
    print(f"Dropped (ratio):       {dropped}")
    print(f"Unreadable/Errors:     {unreadable}")
    print(f"Empty after clean:     {empty}")
    if MAX_FILE_BYTES is None:
        print("Skipped huge files:    disabled")
    else:
        print(f"Skipped huge files:    {skipped_large} (>{MAX_FILE_BYTES} bytes)")
    print(f"T: lines removed:      {t_removed_total}")
    print(f"Outputs:")
    print(f"  - {TRAIN_TXT.as_posix()}")
    print(f"  - {STATS_TSV.as_posix()}")


def main():
    if not RAW_ABC_DIR.exists():
        raise FileNotFoundError(f"Missing input dir: {RAW_ABC_DIR.as_posix()}")

    print("========================================")
    print(f"V3 process_safe.py | MODE={MODE}")
    print(f"INPUT:  {RAW_ABC_DIR.as_posix()}")
    print(f"OUTPUT: {OUT_DIR.as_posix()}")
    print("========================================")

    if MODE.upper() == "ANALYZE":
        run_analyze()
    elif MODE.upper() == "BUILD":
        run_build()
    else:
        raise ValueError("MODE must be 'ANALYZE' or 'BUILD'")


if __name__ == "__main__":
    main()

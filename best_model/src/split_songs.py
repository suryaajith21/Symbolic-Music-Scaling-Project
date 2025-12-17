from __future__ import annotations
from pathlib import Path
import random

IN_PATH = Path("data/V3/train.txt")
OUT_DIR = Path("data/V3")
SEED = 0
TRAIN_FRAC = 0.98
VAL_FRAC = 0.01
TEST_FRAC = 0.01

DELIM = "<|endoftext|>"

def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_out = OUT_DIR / "train_split.txt"
    val_out   = OUT_DIR / "val.txt"
    test_out  = OUT_DIR / "test.txt"

    with IN_PATH.open("r", encoding="utf-8", errors="replace") as f, \
         train_out.open("w", encoding="utf-8", newline="\n") as f_tr, \
         val_out.open("w", encoding="utf-8", newline="\n") as f_va, \
         test_out.open("w", encoding="utf-8", newline="\n") as f_te:

        buf = []
        n = 0
        n_tr = n_va = n_te = 0

        for line in f:
            if line.strip() == DELIM:
                song = "".join(buf)
                buf = []
                if not song.strip():
                    continue

                r = random.random()
                if r < TRAIN_FRAC:
                    out = f_tr; n_tr += 1
                elif r < TRAIN_FRAC + VAL_FRAC:
                    out = f_va; n_va += 1
                else:
                    out = f_te; n_te += 1

                out.write(song)
                out.write("\n" + DELIM + "\n")
                n += 1

            else:
                buf.append(line)

    print("=== SPLIT SUMMARY ===")
    print("songs total:", n)
    print("train:", n_tr, "val:", n_va, "test:", n_te)
    print("wrote:", train_out, val_out, test_out)

if __name__ == "__main__":
    main()

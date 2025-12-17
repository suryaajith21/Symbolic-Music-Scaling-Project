from __future__ import annotations
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

DATA_DIR = Path("data/V3")
TOK_PATH = DATA_DIR / "tokenizer_bpe_4096.json"

FILES = [
    ("train_split.txt", "train.bin"),
    ("val.txt", "val.bin"),
    ("test.txt", "test.bin"),
]

DELIM = "<|endoftext|>"

def encode_file(tok: Tokenizer, in_path: Path, out_path: Path):
    eot_id = tok.token_to_id(DELIM)
    assert eot_id is not None, "EOT token missing in tokenizer"

    ids = []
    with in_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == DELIM:
                ids.append(eot_id)
            else:

                enc = tok.encode(line)
                ids.extend(enc.ids)

    arr = np.array(ids, dtype=np.uint16)
    out_path.write_bytes(arr.tobytes())
    print(f"{in_path.name}: tokens={len(arr):,} -> {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")
    return len(arr)

def main():
    tok = Tokenizer.from_file(str(TOK_PATH))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for in_name, out_name in FILES:
        in_path = DATA_DIR / in_name
        out_path = DATA_DIR / out_name
        if not in_path.exists():
            raise FileNotFoundError(in_path)
        encode_file(tok, in_path, out_path)

if __name__ == "__main__":
    main()

from pathlib import Path
from tokenizers import Tokenizer


DATA_DIR = Path("data/V3")
TOK_PATH = DATA_DIR / "tokenizer_bpe_4096.json"
DELIM = "<|endoftext|>"


FILES = [
    ("Train",      DATA_DIR / "train.txt"),
    ("Validation", DATA_DIR / "val.txt"),
    ("Test",       DATA_DIR / "test.txt")
]

def count_tokens_in_file(path, tokenizer):
    if not path.exists():
        return "File Not Found"

    total_tokens = 0
    buf = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == DELIM:
                text = "".join(buf)
                buf = []
                if text.strip():

                    total_tokens += len(tokenizer.encode(text).ids)
            else:
                buf.append(line)

    return total_tokens

def main():
    if not TOK_PATH.exists():
        print(f"Error: Tokenizer not found at {TOK_PATH}")
        return


    tokenizer = Tokenizer.from_file(str(TOK_PATH))


    print(f"Tokenizer Vocab Size: {tokenizer.get_vocab_size()}")
    print("\n" + "="*35)
    print(f"{'DATASET':<15} | {'TOTAL TOKENS':<15}")
    print("="*35)


    for label, path in FILES:
        count = count_tokens_in_file(path, tokenizer)

        if isinstance(count, int):
            print(f"{label:<15} | {count:,}")
        else:
            print(f"{label:<15} | {count}")

    print("="*35)

if __name__ == "__main__":
    main()

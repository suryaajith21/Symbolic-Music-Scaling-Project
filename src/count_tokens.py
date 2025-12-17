import os
from pathlib import Path


DATA_DIR = Path("data/processed")
FILES_TO_CHECK = [
    ("Train (Colab)", DATA_DIR / "train_colab.txt"),
    ("Validation",    DATA_DIR / "val.txt"),
    ("Test",          DATA_DIR / "test.txt")
]

def count_tokens(file_path):
    """
    Counts characters in the file.
    """
    if not file_path.exists():
        return None, "File not found"

    try:

        file_size = file_path.stat().st_size
        if file_size == 0:
            return 0, "Empty file"

        count = 0
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
        return count, "OK"

    except Exception as e:
        return None, str(e)

def main():
    print(f"{'DATASET':<15} | {'TOKENS (CHARS)':<15}")
    print("-" * 50)

    total_tokens = 0

    for label, path in FILES_TO_CHECK:
        count, status = count_tokens(path)

        if count is not None:

            count_str = f"{count:,}"
            print(f"{label:<15} | {count_str:<15}")


            if "Full" not in label and "Empty" not in status:
                total_tokens += count
        else:
            print(f"{label:<15} | {'N/A':<15}")




if __name__ == "__main__":
    main()

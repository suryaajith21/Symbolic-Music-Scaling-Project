import os
import random
import pickle
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

SOURCE_DIR = Path("data/abc")
OUTPUT_DIR = Path("data/processed")
MIN_FILE_LENGTH = 50

def clean_abc_data(abc_content):
    """
    Parses ABC text to remove metadata (Titles, Composers)
    while keeping musical structure (Headers X, M, L, K, V).
    """
    if not abc_content:
        return None

    lines = abc_content.splitlines()
    cleaned_lines = []

    keep_headers = ('X:', 'M:', 'L:', 'K:', 'V:', 'P:')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if len(line) > 1 and line[1] == ':':
            if line.startswith(keep_headers):
                cleaned_lines.append(line)
        elif not line.startswith('%'):
            cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)

    if len(result) < MIN_FILE_LENGTH:
        return None

    return result

def get_dataset_stats(text_data):
    """Computes vocab and frequency stats for the report."""
    counts = Counter(text_data)
    vocab = sorted(counts.keys())
    vocab_size = len(vocab)

    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    return counts, vocab, stoi, itos

def save_visualization(counts, output_path):
    """Generates a histogram of the top 30 tokens for the report."""
    most_common = counts.most_common(30)
    chars, freqs = zip(*most_common)

    labels = []
    for c in chars:
        if c == '\n': labels.append('\\n')
        elif c == ' ': labels.append('space')
        else: labels.append(c)

    plt.figure(figsize=(12, 6))
    plt.bar(labels, freqs, color='#4C72B0')
    plt.title(f'Top 30 Character Frequencies (Total Vocab: {len(counts)})')
    plt.ylabel('Count')
    plt.xlabel('Token')
    plt.grid(axis='y', alpha=0.3)

    plot_file = output_path / "vocab_dist.png"
    plt.savefig(plot_file)
    print(f"-> Saved visualization to {plot_file}")
    plt.close()

def process_and_save(file_paths, output_file):
    """Reads files, cleans them, and writes to a single text file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_content = []

    print(f"Processing {len(file_paths)} files for {output_file.name}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for path in tqdm(file_paths):
            raw_text = path.read_text(encoding='utf-8', errors='ignore')
            clean_text = clean_abc_data(raw_text)

            if clean_text:

                entry = clean_text + "\n\n"
                f.write(entry)
                all_content.append(entry)

    return "".join(all_content)

def main():
    print("Scanning for .abc files...")
    all_files = list(SOURCE_DIR.glob("*.abc"))
    if not all_files:
        print("Error: No files found.")
        return

    random.seed(42)
    random.shuffle(all_files)

    n_total = len(all_files)
    idx_train = int(n_total * 0.98)
    idx_val = int(n_total * 0.99)

    splits = {
        'train': all_files[:idx_train],
        'val': all_files[idx_train:idx_val],
        'test': all_files[idx_val:]
    }

    full_train_text = process_and_save(splits['train'], OUTPUT_DIR / "train.txt")
    process_and_save(splits['val'], OUTPUT_DIR / "val.txt")
    process_and_save(splits['test'], OUTPUT_DIR / "test.txt")

    print("REPORT DATA OUTPUT:")

    counts, vocab, stoi, itos = get_dataset_stats(full_train_text)

    meta = {
        'vocab_size': len(vocab),
        'stoi': stoi,
        'itos': itos,
    }
    with open(OUTPUT_DIR / 'meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

    print(f"\n[Stats] Vocab Size: {len(vocab)}")
    print(f"[Stats] Total Training Characters: {len(full_train_text):,}")

    print("\n[Tokenization Example]")
    example_str = "M:4/4\nL:1/8"
    encoded = [stoi[c] for c in example_str if c in stoi]
    print(f"Original: {repr(example_str)}")
    print(f"Encoded:  {encoded}")

    save_visualization(counts, OUTPUT_DIR)

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

INPUT_FILE = Path("data/processed/train.txt")
OUTPUT_DIR = Path("data/processed")

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

def main():

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        text_data = f.read()

    counts = Counter(text_data)
    vocab = sorted(counts.keys())
    vocab_size = len(vocab)

    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}

    print(" REPORT REQUIRED INFO:")

    print(f"\nDataset Stats:")
    print(f"• Total Character Count: {len(text_data):,}")
    print(f"• Vocabulary Size:       {vocab_size}")
    print(f"• Most Common Token:     '{counts.most_common(1)[0][0]}' (Count: {counts.most_common(1)[0][1]:,})")

    print("\nTokenization Scheme Example:")
    print("Input String: 'M:4/4\\nL:1/8'")

    example_str = "M:4/4\nL:1/8"

    encoded = [stoi[c] for c in example_str if c in stoi]

    print(f"• Original: {repr(example_str)}")
    print(f"• Encoded:  {encoded}")
    save_visualization(counts, OUTPUT_DIR)

if __name__ == "__main__":
    main()

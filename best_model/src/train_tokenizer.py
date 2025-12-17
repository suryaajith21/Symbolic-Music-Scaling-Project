from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing


IN_PATH = Path("data/V3/train.txt")
OUT_DIR = Path("data/V3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER_JSON = OUT_DIR / "tokenizer_bpe_4096.json"

VOCAB_SIZE = 4096
SPECIAL_TOKENS = ["<|endoftext|>"]


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing: {IN_PATH.as_posix()}")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<unk>"] + SPECIAL_TOKENS,
        show_progress=True,
    )


    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        pair="$A $B",
        special_tokens=[(tok, tokenizer.token_to_id(tok)) for tok in ["<unk>"] + SPECIAL_TOKENS],
    )

    tokenizer.save(str(TOKENIZER_JSON))
    print(f"Saved tokenizer to: {TOKENIZER_JSON.as_posix()}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"eot id: {tokenizer.token_to_id('<|endoftext|>')}")


if __name__ == "__main__":
    main()

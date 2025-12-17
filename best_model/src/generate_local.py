

from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from types import SimpleNamespace
import re

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[2]
SRC_DIR = THIS.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_DIR))


DATA_DIR = REPO_ROOT / "data" / "V3"
TOKENIZER_PATH = DATA_DIR / "tokenizer_bpe_4096.json"
CKPT_PATH = REPO_ROOT / "checkpoints/full_step_0011000.pt"
OUT_DIR = DATA_DIR / "conditional_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)


PREFIX_DICT = {
    "1_Song": """X: 1
M: 4/4
L: 1/8
Q: 1/4=120
K: C
V: 1
%%clef treble
""",

    "2_Song": """X: 1
M: 3/4
L: 1/8
Q: 1/4=90
K: Am
V: 1
%%clef treble
""",

    "3_Song": """X: 1
M: 6/8
L: 1/8
Q: 1/4=160
K: G
V: 1
%%clef treble
""",

    "4_Song": """X: 1
M: 4/4
L: 1/8
K: D
V: 1
%%clef treble
[D2F2A2]""",

    "5_Song": """X: 1
M: 4/4
L: 1/8
K: Fm
V: 1
%%clef treble
"""
}


SAMPLES_PER_PREFIX = 1
MAX_NEW_TOKENS = 1024
TEMP = 0.95
TOP_P = 0.90
REP_PENALTY = 1.1


def _load_tokenizer(tokenizer_path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(tokenizer_path))

def load_checkpoint(ckpt_path, device):
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")


    cfg = ckpt["cfg"] if "cfg" in ckpt else ckpt["config"]
    if isinstance(cfg, dict):
        config = SimpleNamespace(**cfg)
    else:
        config = cfg


    sys.path.append(str(REPO_ROOT))
    from model import GPT

    model = GPT(config)


    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    new_sd = {}
    unwanted = "_orig_mod."
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")
        if new_k.startswith(unwanted):
            new_k = new_k[len(unwanted):]
        new_sd[new_k] = v

    model.load_state_dict(new_sd)
    model.to(device)
    model.eval()
    return model, config

@torch.no_grad()
def sample_next_token(logits, temp, top_p, rep_penalty, ctx_ids):

    if rep_penalty != 1.0 and ctx_ids is not None:
        check_window = 128
        check_ids = set(ctx_ids[-check_window:]) if len(ctx_ids) > 0 else set()
        for token_id in check_ids:
            if logits[token_id] < 0:
                logits[token_id] *= rep_penalty
            else:
                logits[token_id] /= rep_penalty

    logits = logits / max(temp, 1e-6)


    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

@torch.no_grad()
def generate_with_prefix(model, config, tokenizer, device, prefix_str):
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    if eot_id is None: eot_id = 0


    prefix_ids = tokenizer.encode(prefix_str).ids
    ids = [eot_id] + prefix_ids


    for _ in range(MAX_NEW_TOKENS):
        ctx = ids[-config.block_size:]
        x = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)

        logits, _ = model(x)
        next_logits = logits[0, -1, :]

        next_id = sample_next_token(next_logits, TEMP, TOP_P, REP_PENALTY, ids)

        if next_id == eot_id:
            break

        ids.append(next_id)


    text = tokenizer.decode(ids)


    text = text.replace("<|endoftext|>", "")



    matches = list(re.finditer(r"(?m)^[ \t]*X\s*:", text))
    if len(matches) >= 2:
        text = text[:matches[1].start()]

    return text

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not CKPT_PATH.exists():
        print(f"CRITICAL ERROR: Checkpoint not found at {CKPT_PATH}")
        return


    tokenizer = _load_tokenizer(TOKENIZER_PATH)
    model, config = load_checkpoint(CKPT_PATH, device)

    print(f"Generating {SAMPLES_PER_PREFIX} samples for EACH of the {len(PREFIX_DICT)} prefixes.")
    print(f"Output folder: {OUT_DIR}")
    print("="*60)


    for name, prefix_str in PREFIX_DICT.items():
        print(f"\n>>> Processing Prefix: {name}")
        print("-" * 30)

        for i in range(1, SAMPLES_PER_PREFIX + 1):
            print(f"   Generating {name} sample {i}...")

            abc_content = generate_with_prefix(model, config, tokenizer, device, prefix_str)


            fname = OUT_DIR / f"{name}_sample_{i}.abc"
            with open(fname, "w", encoding="utf-8") as f:
                f.write(abc_content)

            print(f"   Saved: {fname.name}")

    print("\n" + "="*60)
    print("All generations complete.")

if __name__ == "__main__":
    main()

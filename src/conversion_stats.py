from pathlib import Path

MIDI_ROOT = Path("data/raw_midi/lmd_full")
ABC_ROOT  = Path("data/abc")


midi_paths = []
for ext in ("*.mid", "*.midi", "*.MID", "*.MIDI"):
    midi_paths.extend(MIDI_ROOT.rglob(ext))

abc_paths = list(ABC_ROOT.rglob("*.abc"))

print(f"MIDI files found: {len(midi_paths)}")
print(f"ABC files found:  {len(abc_paths)}")


midi_stems = [p.stem for p in midi_paths]
abc_stems  = {p.stem for p in abc_paths}

matched = [s for s in midi_stems if s in abc_stems]
missing = [p for p in midi_paths if p.stem not in abc_stems]

attempted = len(midi_paths)
succeeded = len(matched)

print(f"\nMatched (by stem): {succeeded} / {attempted} = {100*succeeded/attempted:.2f}%")


MIN_ABC_BYTES = 200
abc_size = {p.stem: p.stat().st_size for p in abc_paths}

tiny = [s for s in matched if abc_size.get(s, 0) < MIN_ABC_BYTES]
valid = succeeded - len(tiny)

print(f"ABC < {MIN_ABC_BYTES} bytes among matched: {len(tiny)}")
print(f"Valid-ish conversions: {valid} / {attempted} = {100*valid/attempted:.2f}%")


print("\nSample missing MIDI→ABC (first 20):")
for p in missing[:20]:
    print("  ", p)

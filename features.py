from collections import Counter

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def clean_sequence(sequence: str) -> str:
    seq = "".join(str(sequence).split()).upper()
    if not seq:
        raise ValueError("Protein sequence cannot be empty.")
    invalid = sorted(set(seq) - set(AMINO_ACIDS))
    if invalid:
        raise ValueError("Unsupported amino-acid character(s): " + ", ".join(invalid))
    return seq

def extract_features(sequence: str) -> dict:
    seq = clean_sequence(sequence)
    counts = Counter(seq)
    features = {f"frac_{aa}": counts.get(aa, 0) / len(seq) for aa in AMINO_ACIDS}
    features["sequence_length"] = len(seq)
    return features

def feature_names() -> list[str]:
    return [f"frac_{aa}" for aa in AMINO_ACIDS] + ["sequence_length"]

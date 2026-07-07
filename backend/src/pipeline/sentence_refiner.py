"""
Sentence Refiner — Rule-Based Syllable-to-Sentence Refinement Module
=======================================================================
Converts a raw decoded word (e.g., "MAKAN") from the BCI pipeline into a
natural-language sentence appropriate for assistive communication contexts.

This module is intentionally rule-based (dictionary lookup), NOT a
generative LLM backend. For a closed vocabulary of 10 words, a fixed
mapping is deterministic, faster, and does not depend on an external API —
this is a final design decision for this study, not a placeholder awaiting
an LLM integration.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# REFINEMENT TABLE
# ---------------------------------------------------------------------------
# Maps uppercase decoded word → formal assistive-communication sentence.
# These are clinically appropriate Indonesian phrasings for each of the
# 10 target vocabulary words used in the BCI syllable taxonomy.
REFINEMENT_TABLE: dict[str, str] = {
    "MAKAN":  "Saya ingin makan.",
    "MINUM":  "Saya ingin minum.",
    "BERAK":  "Saya ingin buang air besar.",
    "PIPIS":  "Saya ingin buang air kecil.",
    "MANDI":  "Saya ingin mandi.",
    "BOSAN":  "Saya merasa bosan.",
    "LELAH":  "Saya merasa lelah.",
    "SAKIT":  "Saya merasa sakit.",
    "TIDUR":  "Saya ingin tidur.",
    "SAYANG": "Saya sayang kamu.",
}

# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------
def refine_sentence_rule_based(raw_word: str) -> str:
    """
    Refine a raw BCI-decoded word into a communicative sentence.

    The function normalises the input to uppercase and performs a direct lookup
    in the refinement table. If the word is not found (out-of-vocabulary or
    decoding error), the raw word is returned unchanged with a prefix note.

    Args:
        raw_word (str): Decoded word string from the BCI word assembler
                        (e.g., 'MAKAN', 'Tidur', 'sakit').

    Returns:
        str: A natural-language refinement of the decoded intent.
    """
    key = raw_word.strip().upper()
    return REFINEMENT_TABLE.get(key, f"[Unrecognised intent: {raw_word}]")


def get_confidence_label(confidence: float) -> str:
    """
    Convert a numeric confidence score to a human-readable category.

    Args:
        confidence (float): Confidence value in [0, 100].

    Returns:
        str: One of 'High', 'Medium', or 'Low'.
    """
    if confidence >= 85.0:
        return "High"
    if confidence >= 65.0:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_cases = list(REFINEMENT_TABLE.keys()) + ["UNKNOWN_WORD", "sakit", "Makan"]
    print("Sentence Refiner — Rule-Based Refinement Table Dry-Run")
    print("=" * 50)
    for word in test_cases:
        result = refine_sentence_rule_based(word)
        print(f"  {word:<12} -> {result}")
    print("=" * 50)

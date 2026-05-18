"""
LLM Agent — Syllable-to-Sentence Refinement Module
====================================================
Converts a raw decoded word (e.g., "MAKAN") from the BCI pipeline into a
natural-language sentence appropriate for assistive communication contexts.

The current implementation uses a deterministic rule-based refinement table
so that the system is fully operational without an external LLM API key. This
is semantically equivalent for the 10-word closed-vocabulary used in this study.

To upgrade to a generative LLM backend (e.g., Anthropic Claude), replace the
body of `refine_with_llm()` with an API call and set ANTHROPIC_API_KEY in
the backend .env file.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# REFINEMENT TABLE
# ---------------------------------------------------------------------------
# Maps uppercase decoded word → formal assistive-communication sentence.
# These are clinically appropriate Indonesian phrasings for each of the
# 10 target vocabulary words used in the BCI syllable taxonomy.
REFINEMENT_TABLE: dict[str, str] = {
    "MAKAN":  "Saya ingin makan.",
    "MINUM":  "Saya ingin minum.",
    "BERAK":  "Saya ingin ke toilet (BAB).",
    "PIPIS":  "Saya ingin ke toilet (BAK).",
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
def refine_with_llm(raw_word: str) -> str:
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
    print("LLM Agent — Refinement Table Dry-Run")
    print("=" * 50)
    for word in test_cases:
        result = refine_with_llm(word)
        print(f"  {word:<12} -> {result}")
    print("=" * 50)

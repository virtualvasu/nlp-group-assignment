"""
Entry point.

    python main.py evaluate   -> Parts 1-4: build/load data, run accuracy
                                  evaluation + Speed Demon benchmark
    python main.py cli        -> Part 5: interactive terminal corrector
    python main.py demo       -> quick sanity check on the assignment's
                                  example sentences (no user input needed)

Running with no argument does `evaluate` followed by `demo`.
"""

import sys

from src.data_prep import build_and_cache_all
from src.corrector import SpellingCorrector
from src.evaluate import run_full_evaluation
from src.cli import run_cli

EXAMPLE_SENTENCES = [
    "I hav a good feeling about this.",
    "This is a test sentnce.",
    "I would like to sea the world.",
    "Please meat me at the station.",
]


def build_corrector():
    data = build_and_cache_all()
    corrector = SpellingCorrector(
        vocab=data["vocab"],
        freq=data["freq"],
        bigram_counts=data["bigram_counts"],
        unigram_counts=data["unigram_counts"],
        vocab_size=data["vocab_size"],
    )
    return corrector, data


def demo(corrector):
    print("\n" + "=" * 60)
    print("Demo on the assignment's example sentences")
    print("=" * 60)
    for s in EXAMPLE_SENTENCES:
        corrected, changes = corrector.correct_sentence(s)
        print(f"\nInput:     {s}")
        print(f"Corrected: {corrected}")
        print(f"Changes:   {changes if changes else 'none'}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "cli":
        corrector, _ = build_corrector()
        run_cli(corrector)
        return

    corrector, data = build_corrector()

    if mode in ("evaluate", "all"):
        run_full_evaluation(corrector, data["sentences"], data["vocab"])

    if mode in ("demo", "all"):
        demo(corrector)


if __name__ == "__main__":
    main()

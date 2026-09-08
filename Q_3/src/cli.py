"""
Part 5: Live Interactive Application
=======================================
A continuous terminal CLI: type a sentence, press Enter, get the corrected
sentence back with changed words highlighted, plus the latency of the
correction. Type "exit" to quit.
"""

import time

# ANSI color codes. Falls back gracefully -- if the terminal doesn't
# support ANSI, the codes are just harmless characters around the word,
# so we ALSO wrap changed words in **asterisks** as the assignment allows.
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"


def highlight(word):
    return f"{BOLD}{GREEN}**{word}**{RESET}"


def run_cli(corrector):
    print("=" * 60)
    print("Spelling Corrector -- interactive mode")
    print("Type a sentence and press Enter. Type 'exit' to quit.")
    print("=" * 60)

    while True:
        try:
            sentence = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if sentence.lower() == "exit":
            print("Goodbye!")
            break
        if not sentence:
            continue

        start = time.perf_counter()
        corrected, changes = corrector.correct_sentence(sentence)
        latency = time.perf_counter() - start

        if changes:
            display = corrected
            for original, new_word in changes:
                display = display.replace(new_word, highlight(new_word), 1)
            print(f"Corrected: {display}")
            print(f"Changed:   {', '.join(f'{o} -> {n}' for o, n in changes)}")
        else:
            print(f"Corrected: {corrected}")
            print("Changed:   (no changes)")

        print(f"Latency:   {latency * 1000:.2f} ms")


if __name__ == "__main__":
    from src.data_prep import build_and_cache_all
    from src.corrector import SpellingCorrector

    data = build_and_cache_all()
    corrector = SpellingCorrector(
        vocab=data["vocab"],
        freq=data["freq"],
        bigram_counts=data["bigram_counts"],
        unigram_counts=data["unigram_counts"],
        vocab_size=data["vocab_size"],
    )
    run_cli(corrector)

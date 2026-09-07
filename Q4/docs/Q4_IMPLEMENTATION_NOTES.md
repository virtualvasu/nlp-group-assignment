# Q4 Implementation Notes

## Integration boundary

The original Q1 implementation exposes `TrigramLanguageModel`, `FeatureBasedPOSTagger`, and `JointBeamSearchDecoder`; Q4 loads the serialized English trigram + MEMM artifacts and constructs the same joint decoder with max word length 20, beam width 5, alpha 1.0 and beta 1.0.

The original Q3 implementation exposes `SpellingCorrector`, which internally uses Method A and Method B candidate generation. Q4 calls its non-word correction during the per-token layer and its real-word correction during the trigger interval.

## Live processing order

For every completed input token:

1. Detect OOV/unusually long input.
2. Run the reused Q1 joint segmentation/POS decoder when suspicious.
3. If a merged OOV token is split into multiple valid words, emit `[SEGMENT-ALERT]` and continue with the recovered words.
4. Run the reused Q3 non-word spelling corrector on remaining OOV words and emit `[SPELL-ALERT]` when it changes a word.
5. Append the resulting words to the one shared live stream.
6. Every `N=10` accepted output words, run the trigger grammar + Q3 real-word check.

The token timer measures only steps 1–4. The trigger timer is measured separately. Alert messages display the corresponding measured latency instead of a default `0.00 ms`.

## Why the merged-token generator is necessary

The source passage is otherwise already whitespace-tokenized, so segmentation would have little work to do. With `p=0.08`, adjacent words are occasionally concatenated before entering the editor. The reused Q1 joint decoder can then recover multiple words and POS tags.

## Grammar trigger

Every 10 accepted output words, the editor checks the latest 10-word window. It uses the Q4 shared smoothed bigram/trigram models for local sequence plausibility and invokes the unchanged Q3 real-word correction routine, whose bigram model remains the trained Q3 model.

## PCFG

Penn Treebank trees are converted to Chomsky Normal Form before PCFG induction. The parser indexes lexical, unary and binary productions and runs a Viterbi CKY chart. Unary closure is applied after lexical initialization and binary-combination steps. Unknown lexical words receive a tiny backoff probability under the reconciled POS so parsing fails gracefully rather than crashing.

## Tagset mapping

Q1 uses the Brown/Universal POS tags while the Penn Treebank grammar uses PTB tags. Q4 maps common Universal tags to PTB tags, including `PART -> RP` and legacy `PRT -> RP`. The mapping loses some fine-grained information; for example Universal `VERB` collapses PTB distinctions such as VB/VBD/VBG/VBN/VBP/VBZ. This is documented as a source of possible PCFG accuracy loss.

## Final method selection

For each finalized sentence, Q4 first attempts the PCFG parse. If parsing succeeds and its normalized log probability is not a strong passage-level outlier, PCFG is selected. Otherwise the trigram score is preferred when finite, with bigram as the final fallback. The verdict is a model-based plausibility label, not a claim of perfect linguistic grammaticality.

## Speed Demon interpretation

The benchmark intentionally disables grammar triggers so that the full 1,000-word measurement isolates the added per-token segmentation + spelling layer. The same 1,000 tokens are then sent through the grammar/real-word checker in isolation. If Q1 OOV decoding dominates runtime, the report should say so and discuss practical throttling/optimization (for example, only decoding suspicious tokens, increasing the trigger interval, or moving expensive work off the keystroke path) rather than hiding the measured overhead.

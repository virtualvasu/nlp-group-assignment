import random
import nltk


def ensure_corpora():
    for name, path in [('gutenberg', 'corpora/gutenberg'),
                       ('brown', 'corpora/brown'),
                       ('reuters', 'corpora/reuters')]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass


def sample_passage(seed=None, min_sentences=5, max_sentences=8):
    """Sample 5–8 contiguous sentences from one corpus document/file."""
    if min_sentences < 1 or max_sentences < min_sentences:
        raise ValueError('Invalid sentence-count bounds.')

    ensure_corpora()
    rng = random.Random(seed)
    n = rng.randint(min_sentences, max_sentences)

    # Keep the sampled passage inside one source file. This avoids the old
    # flattened-corpus bug where a slice could cross a document boundary.
    sources = []
    try:
        from nltk.corpus import gutenberg
        for fid in gutenberg.fileids():
            sentences = [list(s) for s in gutenberg.sents(fileids=[fid])]
            if len(sentences) >= n:
                sources.append((f'gutenberg:{fid}', sentences))
    except Exception:
        pass

    if not sources:
        try:
            from nltk.corpus import brown
            for fid in brown.fileids():
                sentences = [list(s) for s in brown.sents(fileids=[fid])]
                if len(sentences) >= n:
                    sources.append((f'brown:{fid}', sentences))
        except Exception:
            pass

    if not sources:
        try:
            from nltk.corpus import reuters
            for fid in reuters.fileids():
                sentences = [list(s) for s in reuters.sents(fileids=[fid])]
                if len(sentences) >= n:
                    sources.append((f'reuters:{fid}', sentences))
        except Exception:
            pass

    if not sources:
        raise RuntimeError('No NLTK passage corpus is available with a sufficiently long document.')

    _, sentences = rng.choice(sources)
    start = rng.randint(0, len(sentences) - n)
    return sentences[start:start + n]

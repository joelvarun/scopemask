"""Byte-pair tokenizer trained on the MiniPy corpus.

Subword tokenization is what makes scope-aware masking non-trivial: a legal name is
spelled across several tokens, so the mask at step k has to answer "is there any
legal name whose spelling continues with this token" rather than "is this token a
legal name".

Pre-tokenization splits text into runs of identifier characters and runs of
non-identifier characters, so merges produce both identifier fragments (scale, _conf,
ter) and glue tokens (space equals space, close paren newline indent).
"""

from collections import defaultdict

from .minipy import IDENT_CHARS

EOS = chr(0)


def pretokenize(text):
    out = []
    i = 0
    n = len(text)
    while i < n:
        ident = text[i] in IDENT_CHARS
        j = i
        while j < n and (text[j] in IDENT_CHARS) == ident:
            j += 1
        out.append(text[i:j])
        i = j
    return out


def _word_pairs(sym):
    return [(sym[k], sym[k + 1]) for k in range(len(sym) - 1)]


def train_bpe(texts, n_merges=2400):
    """Incremental BPE over word types. Returns the ordered list of merged strings."""
    counts = defaultdict(int)
    for text in texts:
        for w in pretokenize(text):
            counts[w] += 1

    words = []
    freqs = []
    for w, c in counts.items():
        words.append(list(w))
        freqs.append(c)

    pair_counts = defaultdict(int)
    pair_where = defaultdict(set)
    for wid, sym in enumerate(words):
        f = freqs[wid]
        for p in _word_pairs(sym):
            pair_counts[p] += f
            pair_where[p].add(wid)

    merges = []
    for _ in range(n_merges):
        best = None
        best_c = 0
        for p, c in pair_counts.items():
            if c > best_c:
                best_c = c
                best = p
        if best is None or best_c < 2:
            break
        merged = best[0] + best[1]
        merges.append(merged)
        touched = list(pair_where[best])
        for wid in touched:
            sym = words[wid]
            f = freqs[wid]
            old = _word_pairs(sym)
            new_sym = []
            k = 0
            while k < len(sym):
                if k + 1 < len(sym) and sym[k] == best[0] and sym[k + 1] == best[1]:
                    new_sym.append(merged)
                    k += 2
                else:
                    new_sym.append(sym[k])
                    k += 1
            if new_sym == sym:
                continue
            words[wid] = new_sym
            new = _word_pairs(new_sym)
            for p in old:
                pair_counts[p] -= f
                if pair_counts[p] <= 0:
                    pair_counts.pop(p, None)
                    pair_where.pop(p, None)
                elif wid in pair_where.get(p, ()):
                    pass
            for p in set(old):
                s = pair_where.get(p)
                if s is not None:
                    s.discard(wid)
            for p in new:
                pair_counts[p] += f
                pair_where[p].add(wid)
        pair_counts.pop(best, None)
        pair_where.pop(best, None)
    return merges


class Tokenizer:
    """Greedy longest-match subword tokenizer over a fixed vocabulary."""

    def __init__(self, vocab):
        self.vocab = list(vocab)
        self.index = {s: i for i, s in enumerate(self.vocab)}
        self.eos_id = self.index[EOS]
        self.max_len = max(len(s) for s in self.vocab if s != EOS)

    @classmethod
    def train(cls, texts, n_merges=2400):
        chars = set()
        for text in texts:
            chars.update(text)
        merges = train_bpe(texts, n_merges=n_merges)
        vocab = [EOS] + sorted(chars) + [m for m in merges if m not in chars]
        seen = set()
        uniq = []
        for s in vocab:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return cls(uniq)

    def encode(self, text, add_eos=False):
        ids = []
        for word in pretokenize(text):
            i = 0
            n = len(word)
            while i < n:
                hi = min(n, i + self.max_len)
                while hi > i:
                    tid = self.index.get(word[i:hi])
                    if tid is not None:
                        ids.append(tid)
                        i = hi
                        break
                    hi -= 1
                else:
                    raise KeyError("no vocabulary entry for " + repr(word[i]))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids if i != self.eos_id)

    def __len__(self):
        return len(self.vocab)

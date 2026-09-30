"""A deliberately small local generator: an interpolated backoff token n-gram model.

The point of the experiment is not that this model is good. It is that it fails the
way a large model fails on unfamiliar code: it produces well-formed syntax and then
names a variable that is not in scope, because it learned the shape of the corpus
rather than the scope of this particular program.

Two components are mixed:

NgramLM    a global backoff n-gram fitted on the training corpus. It knows what code
           looks like, including where identifiers end, but it has never seen the
           header it is now completing.

InContextLM an n-gram fitted online on the current program only, header included.
           This is the induction-head behaviour that keeps real models mostly in
           scope: they repeat identifiers they have already seen. Its weight is the
           copy_weight knob, and raising it emulates a stronger generator.

Neither component is allowed to look at the legal-name set, so the baseline is never
secretly told the answer.
"""

import numpy as np


class BackoffLM:
    """Interpolated n-gram over token ids. order=4 means contexts up to 3 tokens."""

    def __init__(self, vocab_size, order=4, weights=None):
        self.V = vocab_size
        self.order = order
        if weights is None:
            # longest context first, then shorter, then unigram, then uniform
            weights = _default_weights(order)
        self.weights = weights
        self.tables = [{} for _ in range(order)]
        self.uni = np.full(vocab_size, 1.0 / vocab_size)

    def fit(self, sequences, pad_id):
        counts = [{} for _ in range(self.order)]
        uni = np.zeros(self.V)
        pad = (pad_id,) * (self.order - 1)
        for seq in sequences:
            hist = pad
            for tid in seq:
                uni[tid] += 1.0
                for k in range(1, self.order):
                    ctx = hist[len(hist) - k:]
                    table = counts[k].setdefault(ctx, {})
                    table[tid] = table.get(tid, 0) + 1
                hist = (hist + (tid,))[1:]
        total = uni.sum()
        if total > 0:
            self.uni = uni / total
        for k in range(1, self.order):
            self.tables[k] = _compact(counts[k])
        return self

    def add(self, out, hist):
        """Accumulate the interpolated distribution into out. hist is a tuple."""
        w = self.weights
        for k in range(self.order - 1, 0, -1):
            if len(hist) < k:
                continue
            entry = self.tables[k].get(hist[len(hist) - k:])
            if entry is not None:
                out[entry[0]] += w[k] * entry[1]
        out += w[0] * self.uni
        return out


class InContextLM:
    """Online n-gram over the tokens of the current program only."""

    def __init__(self, vocab_size, order=3):
        self.V = vocab_size
        self.order = order
        self.tables = [{} for _ in range(order)]
        self.uni = np.zeros(vocab_size)
        self.total = 0.0

    def observe(self, hist, tid):
        for k in range(1, self.order):
            if len(hist) < k:
                continue
            ctx = hist[len(hist) - k:]
            table = self.tables[k].setdefault(ctx, {})
            table[tid] = table.get(tid, 0) + 1
        self.uni[tid] += 1.0
        self.total += 1.0

    def probs(self, hist):
        """Longest available in-context match, backing off to the token bag."""
        for k in range(self.order - 1, 0, -1):
            if len(hist) < k:
                continue
            table = self.tables[k].get(hist[len(hist) - k:])
            if table:
                out = np.zeros(self.V)
                norm = float(sum(table.values()))
                for tid, c in table.items():
                    out[tid] = c / norm
                return out
        if self.total <= 0:
            return None
        return self.uni / self.total


def _default_weights(order, head=0.88):
    """Geometric interpolation weights, longest context first.

    head is deliberately large: a light backoff tail is what stops the model from
    scattering probability onto identifier fragments that no real name contains.
    """
    w = {}
    share = 1.0
    for k in range(order - 1, 0, -1):
        w[k] = share * head
        share -= w[k]
    w[0] = share
    return w


def _compact(counts):
    out = {}
    for ctx, table in counts.items():
        ids = np.fromiter(table.keys(), dtype=np.int64, count=len(table))
        vals = np.fromiter(table.values(), dtype=np.float64, count=len(table))
        out[ctx] = (ids, vals / vals.sum())
    return out

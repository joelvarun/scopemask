"""Two token-mask builders for the MiniPy automaton.

Both answer the same question: which vocabulary entries can be appended at this
decoder state without making the program unfinishable.

NaiveMasker replays every vocabulary entry through the automaton, character by
character, with early exit on the first dead character. That is what a direct
implementation of scope-aware masking looks like, and it costs O(V) automaton steps
per decoded token.

FastMasker splits the vocabulary once, at construction time, into

  pure  : entries made only of identifier characters (scale, _conf, ter, 12)
  mixed : everything else (space equals space, close paren newline indent, comma)

The pure half is the large half, and it is also the half whose legality depends only
on the legal-name set. So the pure half is answered by

  a static prefix mask, cached per partial identifier and reused for the whole run,
  because the module level namespace never changes
  +
  a dynamic overlay walked from the handful of names currently in local scope

and the mixed half is small enough to replay directly. The cost per decoded token
becomes O(|mixed| + |locals| * ident_len) automaton steps plus two vectorized passes,
with no per-token work over the identifier half of the vocabulary.

FastMasker.mask and NaiveMasker.mask must agree on every reachable state, in every
namespace mode. That is asserted in tests/test_masker.py and re-checked on every
sampled state during the benchmark run.
"""

from bisect import bisect_left

import numpy as np

from .minipy import (ALL_METHODS, ANY_TYPE, DIGITS, IDENT_CHARS, IDENT_START,
                     LEX_TYPE, MAX_BODY_LINES, MAX_IDENT, MAX_INT_DIGITS,
                     METHODS, MODE_STRICT, MODE_SYNTAX,
                     RETURN_WORD, step)

ATOM_MODES = ("ATOM1", "ATOM2", "GARG")


class NaiveMasker:
    """Replay every vocabulary entry through the automaton."""

    def __init__(self, tok, ns):
        self.tok = tok
        self.ns = ns
        self.V = len(tok.vocab)
        self.eos_id = tok.eos_id
        self.step_calls = 0
        self.index_lookups = 0

    def mask(self, st):
        out = np.zeros(self.V, dtype=bool)
        if st.done:
            out[self.eos_id] = True
            return out
        ns = self.ns
        calls = 0
        for tid, s in enumerate(self.tok.vocab):
            if tid == self.eos_id:
                continue
            cur = st
            for ch in s:
                calls += 1
                cur = step(cur, ch, ns)
                if cur is None:
                    break
            if cur is not None:
                out[tid] = True
        self.step_calls += calls
        return out


class FastMasker:
    """Static/dynamic factorized mask builder. Same answers, much less work."""

    def __init__(self, tok, ns):
        self.tok = tok
        self.ns = ns
        vocab = tok.vocab
        V = len(vocab)
        self.V = V
        self.eos_id = tok.eos_id

        self.tok_len = np.array([len(s) for s in vocab], dtype=np.int32)
        pure = np.zeros(V, dtype=bool)
        digit = np.zeros(V, dtype=bool)
        pure_start = np.zeros(V, dtype=bool)
        for tid, s in enumerate(vocab):
            if tid == self.eos_id or not s:
                continue
            if all(c in IDENT_CHARS for c in s):
                pure[tid] = True
                if s[0] in IDENT_START:
                    pure_start[tid] = True
                if all(c in DIGITS for c in s):
                    digit[tid] = True
        self.pure = pure
        self.digit = digit
        self.pure_start = pure_start
        self.digit_atom = digit & (self.tok_len <= MAX_INT_DIGITS)
        self.mixed_ids = [t for t in range(V) if not pure[t] and t != self.eos_id]
        self.n_pure = int(pure.sum())
        self.n_mixed = len(self.mixed_ids)

        self.static_names = sorted(ns.all_names())
        self._static_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.step_calls = 0
        self.index_lookups = 0

    def _static_with_prefix(self, buf):
        names = self.static_names
        i = bisect_left(names, buf)
        n = len(names)
        while i < n and names[i].startswith(buf):
            yield names[i]
            i += 1

    def _add_pieces(self, names, buf, out):
        """Mark every pure entry that extends buf along one of names."""
        index = self.tok.index
        pure = self.pure
        max_len = self.tok.max_len
        room = MAX_IDENT - len(buf)
        cut = len(buf)
        seen = set()
        for name in names:
            if not name.startswith(buf):
                continue
            suffix = name[cut:]
            lim = min(len(suffix), max_len, room)
            for L in range(1, lim + 1):
                piece = suffix[:L]
                if piece in seen:
                    continue
                seen.add(piece)
                self.index_lookups += 1
                tid = index.get(piece)
                if tid is not None and pure[tid]:
                    out[tid] = True

    def _static_mask(self, buf):
        cached = self._static_cache.get(buf)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        out = np.zeros(self.V, dtype=bool)
        self._add_pieces(self._static_with_prefix(buf), buf, out)
        self._static_cache[buf] = out
        return out

    def _free_ident(self, buf):
        """Any identifier continuation. Used by the grammar-only view."""
        room = MAX_IDENT - len(buf)
        base = self.pure_start if buf == "" else self.pure
        return base & (self.tok_len <= room)

    def _name_position(self, st, out):
        mode = self.ns.mode
        if mode == MODE_SYNTAX:
            np.logical_or(out, self._free_ident(st.buf), out=out)
            return
        np.logical_or(out, self._static_mask(st.buf), out=out)
        if mode == MODE_STRICT:
            self._add_pieces(
                [n for n in st.env if n.startswith(st.buf)], st.buf, out)

    def warm(self, bufs):
        """Pre-populate the static prefix cache. One-time cost, like a grammar compile."""
        for buf in bufs:
            self._static_mask(buf)

    def mask(self, st):
        out = np.zeros(self.V, dtype=bool)
        if st.done:
            out[self.eos_id] = True
            return out

        mode = st.mode
        if mode == "TARGET":
            if st.line >= MAX_BODY_LINES:
                self._add_pieces([RETURN_WORD], st.buf, out)
            else:
                np.logical_or(out, self._free_ident(st.buf), out=out)
        elif mode == "ATTR":
            if st.recv_type == ANY_TYPE:
                np.logical_or(out, self._free_ident(st.buf), out=out)
            elif st.recv_type == LEX_TYPE:
                self._add_pieces(ALL_METHODS, st.buf, out)
            else:
                self._add_pieces(sorted(METHODS[st.recv_type]), st.buf, out)
        elif mode == "RET":
            self._name_position(st, out)
        elif mode in ATOM_MODES:
            if st.numeric:
                room = MAX_INT_DIGITS - len(st.buf)
                np.logical_or(out, self.digit & (self.tok_len <= room), out=out)
            else:
                if st.buf == "":
                    np.logical_or(out, self.digit_atom, out=out)
                self._name_position(st, out)

        ns = self.ns
        vocab = self.tok.vocab
        calls = 0
        for tid in self.mixed_ids:
            cur = st
            for ch in vocab[tid]:
                calls += 1
                cur = step(cur, ch, ns)
                if cur is None:
                    break
            if cur is not None:
                out[tid] = True
        self.step_calls += calls
        return out

    def composition(self):
        return {"vocab_size": self.V, "pure_ident_tokens": self.n_pure,
                "mixed_tokens": self.n_mixed}

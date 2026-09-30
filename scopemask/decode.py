"""Decoding loops for the three arms of the experiment.

syntax    Grammar-constrained decoding. Output always parses, identifiers are free.
lexicon   Grammar plus a flat project lexicon. Output always parses and every
          identifier is a real name from somewhere in the project, but nothing checks
          whether it is in scope here or whether the receiver has that method. This is
          the realistic baseline: fluent, plausible code that reads the wrong variable.
scopemask Grammar plus scope plus coarse types, folded into the token mask.

The first two are paired with rejection: generate, check, and if the program does not
resolve, throw it away and generate again. Every token of every rejected attempt is
paid for. scopemask needs no retries, because an invalid program is unreachable.
"""

import numpy as np

from . import corpus, minipy
from .lm import BackoffLM, InContextLM
from .masker import FastMasker, NaiveMasker
from .tokenizer import Tokenizer
from .validate import check

LM_ORDER = 5
COPY_ORDER = 4
ARMS = ("syntax", "lexicon", "scopemask")


class Engine:
    """Holds the corpus, tokenizer, generator and the three maskers."""

    def __init__(self, n_programs=4000, n_merges=2000, seed=7, order=LM_ORDER):
        texts, _programs, ns = corpus.make_corpus(n_programs, seed=seed)
        self.ns = ns
        self.tok = Tokenizer.train(texts, n_merges=n_merges)
        self.order = order
        self.V = len(self.tok)
        sequences = [self.tok.encode(t, add_eos=True) for t in texts]
        self.lm = BackoffLM(self.V, order=order).fit(
            sequences, pad_id=self.tok.eos_id)
        views = {
            "scopemask": ns,
            "lexicon": ns.lexicon_view(corpus.LOCAL_POOL),
            "syntax": ns.syntax_view(),
        }
        self.views = views
        self.maskers = {name: FastMasker(self.tok, view)
                        for name, view in views.items()}
        self.naive = {name: NaiveMasker(self.tok, view)
                      for name, view in views.items()}

    def new_task(self, rng):
        """A held-out header to complete. Returns (header, params, prompt_ids)."""
        header, _body, params = corpus.make_program(rng, self.ns)
        prompt = header + minipy.NL + minipy.INDENT
        return header, params, self.tok.encode(prompt)

    def _seed_context(self, prompt_ids):
        ic = InContextLM(self.V, order=COPY_ORDER)
        hist = ()
        for tid in prompt_ids:
            ic.observe(hist, tid)
            hist = (hist + (tid,))[-(self.order - 1):]
        return ic, hist

    def sample(self, params, prompt_ids, rng, copy_weight=0.0, arm="scopemask",
               max_tokens=160, collect=None):
        """Sample one completion. Returns (tokens_sampled, body_text)."""
        masker = self.maskers[arm]
        ns = masker.ns
        tok = self.tok
        eos = tok.eos_id
        ic, hist = self._seed_context(prompt_ids)
        state = minipy.start_state(params)

        pieces = []
        steps = 0
        while steps < max_tokens:
            probs = np.zeros(self.V)
            self.lm.add(probs, hist)
            probs /= probs.sum()
            if copy_weight > 0.0:
                copied = ic.probs(hist)
                if copied is not None:
                    probs = (1.0 - copy_weight) * probs + copy_weight * copied
            if collect is not None:
                collect.append(state)
            allowed = masker.mask(state)
            if not allowed.any():
                raise AssertionError("empty mask in arm " + arm)
            probs = probs * allowed
            total = probs.sum()
            if total <= 0.0:
                probs = allowed.astype(np.float64)
                total = probs.sum()
            probs = probs / total

            tid = int(np.searchsorted(np.cumsum(probs), rng.random()))
            if tid >= self.V:
                tid = self.V - 1
            steps += 1
            if tid == eos:
                break
            piece = tok.vocab[tid]
            pieces.append(piece)
            ic.observe(hist, tid)
            hist = (hist + (tid,))[-(self.order - 1):]
            state = minipy.step_text(state, piece, ns)
            if state is None:
                raise AssertionError("decoder left the automaton in arm " + arm)
        return steps, minipy.INDENT + "".join(pieces)

    def is_valid(self, header, params, body_text):
        """Strict automaton verdict, independent ast verdict, and the reason."""
        rest = body_text[len(minipy.INDENT):]
        state = minipy.step_text(minipy.start_state(params), rest, self.ns)
        automaton_ok = state is not None and state.done
        ast_ok, reason = check(header, body_text, params, self.ns)
        return automaton_ok, ast_ok, reason

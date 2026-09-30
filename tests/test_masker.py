"""The factorized masker must agree with the naive one on every reachable state."""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scopemask.decode import Engine

ENGINE = Engine(n_programs=1200, n_merges=800)
ARMS = ("syntax", "lexicon", "scopemask")


def _states(arm, n_programs=6, copy_weight=0.8, seed=5):
    rng = np.random.default_rng(seed)
    headers = random.Random(seed + 1)
    collected = []
    for _ in range(n_programs):
        _header, params, prompt = ENGINE.new_task(headers)
        ENGINE.sample(params, prompt, rng, copy_weight=copy_weight, arm=arm,
                      collect=collected)
    return collected


def test_fast_and_naive_masks_agree():
    for arm in ARMS:
        fast = ENGINE.maskers[arm]
        naive = ENGINE.naive[arm]
        states = _states(arm)
        assert len(states) > 50
        for state in states:
            assert np.array_equal(fast.mask(state), naive.mask(state)), state.mode


def test_masks_are_never_empty():
    for arm in ARMS:
        fast = ENGINE.maskers[arm]
        for state in _states(arm, n_programs=4):
            assert fast.mask(state).any()


def test_scopemask_is_a_subset_of_grammar_only():
    """Every arm enforces the same grammar, so scopemask can only remove tokens."""
    strict = ENGINE.maskers["scopemask"]
    syntax = ENGINE.maskers["syntax"]
    lexicon = ENGINE.maskers["lexicon"]
    for arm in ARMS:
        for state in _states(arm, n_programs=3):
            assert not (strict.mask(state) & ~syntax.mask(state)).any()
            assert not (lexicon.mask(state) & ~syntax.mask(state)).any()


def test_scopemask_narrows_the_lexicon_when_locals_come_from_the_pool():
    """The flat lexicon is not a superset in general.

    A decoder is free to bind a local the lexicon has never heard of, and from then
    on scopemask admits a name that the lexicon view rejects. Where every bound name
    does come from the pool, which is the case inside the corpus, scopemask is the
    strictly tighter mask.
    """
    from scopemask import corpus, minipy
    strict = ENGINE.maskers["scopemask"]
    lexicon = ENGINE.maskers["lexicon"]
    headers = random.Random(3)
    narrowed = 0
    for _ in range(5):
        _header, body, params = corpus.make_program(headers, ENGINE.ns)
        rest = corpus.body_text(body)[len(minipy.INDENT):]
        state = minipy.start_state(params)
        for ch in rest:
            tight = strict.mask(state)
            wide = lexicon.mask(state)
            assert not (tight & ~wide).any()
            if int(wide.sum()) > int(tight.sum()):
                narrowed += 1
            state = minipy.step(state, ch, ENGINE.ns)
            assert state is not None
    assert narrowed > 0


def test_vocabulary_split_covers_everything():
    fast = ENGINE.maskers["scopemask"]
    composition = fast.composition()
    total = composition["pure_ident_tokens"] + composition["mixed_tokens"]
    assert total == composition["vocab_size"] - 1  # the end of sequence entry


def test_static_prefix_cache_is_reused():
    fast = ENGINE.maskers["scopemask"]
    fast.cache_hits = 0
    fast.cache_misses = 0
    states = _states("scopemask", n_programs=5, seed=9)
    for state in states:
        fast.mask(state)
    for state in states:
        fast.mask(state)
    assert fast.cache_hits > fast.cache_misses

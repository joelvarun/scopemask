"""End to end properties of the three decoding arms."""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scopemask import minipy
from scopemask.decode import Engine
from scopemask.validate import audit

ENGINE = Engine(n_programs=1200, n_merges=800)


def _run(arm, n=25, copy_weight=0.8, seed=2):
    rng = np.random.default_rng(seed)
    headers = random.Random(seed + 1)
    out = []
    for _ in range(n):
        header, params, prompt = ENGINE.new_task(headers)
        steps, body = ENGINE.sample(params, prompt, rng, copy_weight=copy_weight,
                                    arm=arm)
        out.append((header, params, body, steps))
    return out


def test_scopemask_output_always_resolves():
    for header, params, body, _steps in _run("scopemask"):
        automaton_ok, ast_ok, reason = ENGINE.is_valid(header, params, body)
        assert automaton_ok, body
        assert ast_ok, reason


def test_both_graders_always_agree():
    for arm in ("syntax", "lexicon", "scopemask"):
        for header, params, body, _steps in _run(arm, n=20):
            automaton_ok, ast_ok, reason = ENGINE.is_valid(header, params, body)
            assert automaton_ok == ast_ok, (arm, reason, body)


def test_relaxed_arms_always_parse():
    """The grammar is enforced in every arm, so the output is always Python."""
    import ast as ast_module
    for arm in ("syntax", "lexicon"):
        for header, _params, body, _steps in _run(arm, n=20):
            ast_module.parse(header + minipy.NL + body)


def test_scopemask_has_no_unresolved_use_positions():
    total = 0
    for header, params, body, _steps in _run("scopemask"):
        uses, bad = audit(header, body, params, ENGINE.ns)
        total += uses
        assert bad == 0
    assert total > 50


def test_baseline_does_have_unresolved_use_positions():
    bad_total = 0
    for header, params, body, _steps in _run("lexicon"):
        _uses, bad = audit(header, body, params, ENGINE.ns)
        bad_total += bad
    assert bad_total > 0


def test_sampling_is_reproducible():
    first = _run("scopemask", n=5, seed=31)
    second = _run("scopemask", n=5, seed=31)
    assert [row[2] for row in first] == [row[2] for row in second]

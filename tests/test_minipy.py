"""The automaton must accept exactly the programs the ast resolver accepts."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scopemask import corpus, minipy
from scopemask.validate import check

NS = corpus.build_namespace()
TEXTS, PROGRAMS, _NS = corpus.make_corpus(200, ns=NS)


def accepts(view, params, body_lines):
    rest = corpus.body_text(body_lines)[len(minipy.INDENT):]
    state = minipy.step_text(minipy.start_state(params), rest, view)
    return state is not None and state.done


def test_strict_accepts_every_generated_program():
    for _header, body, params in PROGRAMS:
        assert accepts(NS, params, body)


def test_relaxed_views_are_supersets():
    lexicon = NS.lexicon_view(corpus.LOCAL_POOL)
    syntax = NS.syntax_view()
    for _header, body, params in PROGRAMS:
        assert accepts(lexicon, params, body)
        assert accepts(syntax, params, body)


def test_ast_resolver_agrees_with_automaton():
    for header, body, params in PROGRAMS:
        ok, reason = check(header, corpus.body_text(body), params, NS)
        assert ok, reason


def test_out_of_scope_name_is_rejected():
    header, body, params = PROGRAMS[0]
    broken = list(body)
    broken[-1] = "return not_bound_anywhere"
    assert not accepts(NS, params, broken)
    ok, _reason = check(header, corpus.body_text(broken), params, NS)
    assert not ok


def test_wrong_method_for_receiver_is_rejected():
    params = [("xs", "list"), ("s", "str"), ("d", "dict")]
    header = "def f(xs, s, d):"
    good = ["a = xs.count()", "return a"]
    bad = ["a = xs.upper()", "return a"]
    assert accepts(NS, params, good)
    assert not accepts(NS, params, bad)
    assert check(header, corpus.body_text(good), params, NS)[0]
    assert not check(header, corpus.body_text(bad), params, NS)[0]


def test_keywords_are_not_valid_targets():
    params = [("xs", "list"), ("s", "str"), ("d", "dict")]
    assert not accepts(NS, params, ["def = 1 + 1", "return def"])
    assert not accepts(NS.syntax_view(), params, ["class = 1 + 1", "return class"])


def test_calling_a_non_callable_is_rejected():
    params = [("xs", "list"), ("s", "str"), ("d", "dict")]
    assert not accepts(NS, params, ["a = xs(1)", "return a"])
    assert accepts(NS, params, ["a = len(xs)", "return a"])


def test_body_line_budget_forces_a_return():
    params = [("xs", "list"), ("s", "str"), ("d", "dict")]
    too_long = ["a" + str(i) + " = 1 + 1" for i in range(minipy.MAX_BODY_LINES + 1)]
    too_long.append("return a0")
    assert not accepts(NS, params, too_long)

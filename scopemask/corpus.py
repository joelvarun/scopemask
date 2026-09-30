"""Namespace and training-corpus generation for MiniPy.

The corpus is what the n-gram generator in scopemask.lm learns from. Two properties
matter for the experiment:

1. The static namespace (module level functions) is shared by every program, so a
   generator that emits a static name is almost always right.
2. Local names are drawn from a large pool and only a handful are bound per program,
   so a generator that emits a local name copied from some other program is almost
   always wrong. That is the hallucinated-variable failure mode, in miniature.
"""

import random

from . import minipy
from .minipy import BUILTINS, METHODS, NL, VALUE_TYPES

VERBS = [
    "load", "save", "parse", "build", "merge", "split", "scan", "emit",
    "fetch", "push", "pull", "trim", "pack", "wrap", "bind", "probe",
    "route", "check", "align", "sort", "rank", "pick", "drop", "fold",
    "hash", "mask", "patch", "queue", "read", "write", "index", "count",
    "clamp", "scale", "shift", "group", "flush", "purge", "widen", "seal",
]

NOUNS = [
    "config", "header", "record", "token", "buffer", "chunk", "frame",
    "batch", "table", "cache", "entry", "field", "label", "metric",
    "sample", "window", "stream", "block", "edge", "node", "path",
    "query", "route", "shard", "slice", "stamp", "value", "vector",
    "weight", "bound", "limit", "cursor", "digest", "handle", "offset",
    "region", "schema", "series", "bucket", "column",
]

LOCAL_POOL = [
    "acc", "buf", "cnt", "cur", "dst", "idx", "key", "out", "pos", "res",
    "row", "seg", "tmp", "val", "total", "items", "names", "parts", "chunk",
    "block", "entry", "label", "value", "delta", "ratio", "score", "depth",
    "width", "limit", "bound", "order", "group", "frame", "stack", "queue",
    "index", "offset", "cursor", "head", "tail", "span", "mark", "hint",
    "slot", "unit", "edge", "leaf", "pair",
]

QUALIFIERS = ["", "_raw", "_safe", "_view", "_meta", "_pair", "_ref", "_span"]

RETURN_TYPE_CYCLE = ("num", "list", "str", "dict")


def build_namespace(seed=17):
    """Deterministic static namespace of module level functions."""
    functions = {}
    i = 0
    for verb in VERBS:
        for noun in NOUNS:
            for qual in QUALIFIERS:
                name = verb + "_" + noun + qual
                functions[name] = RETURN_TYPE_CYCLE[i % len(RETURN_TYPE_CYCLE)]
                i += 1
    return minipy.Namespace(functions)


def _atom(rng, locals_, allow_int=True):
    if allow_int and rng.random() < 0.18:
        return str(rng.randint(0, 999)), "num"
    name = rng.choice(locals_)
    return name, None


def make_program(rng, ns, min_lines=2, max_lines=minipy.MAX_BODY_LINES):
    """Generate one valid MiniPy program. Returns (header, body_lines, params)."""
    fname = rng.choice(VERBS) + "_" + rng.choice(NOUNS) + rng.choice(QUALIFIERS)
    n_params = 3
    param_names = rng.sample(LOCAL_POOL, n_params)
    types = rng.sample(list(VALUE_TYPES), n_params)
    params = list(zip(param_names, types))
    header = "def " + fname + "(" + ", ".join(param_names) + "):"

    env = dict(params)
    unused = [n for n in LOCAL_POOL if n not in env]
    rng.shuffle(unused)
    static_names = sorted(ns.functions)

    body = []
    n_lines = rng.randint(min_lines, max_lines)
    for _ in range(n_lines):
        target = unused.pop()
        form = rng.random()
        bound = sorted(env)
        if form < 0.40:
            a, _t = _atom(rng, bound)
            b, _t2 = _atom(rng, bound)
            op = rng.choice("+-*")
            body.append(target + " = " + a + " " + op + " " + b)
            env[target] = "num"
        elif form < 0.70:
            recv_pool = [n for n in bound if env[n] in METHODS]
            recv = rng.choice(recv_pool)
            method = rng.choice(sorted(METHODS[env[recv]]))
            body.append(target + " = " + recv + "." + method + "()")
            env[target] = METHODS[env[recv]][method]
        else:
            if rng.random() < 0.25:
                fn = rng.choice(sorted(BUILTINS))
            else:
                fn = rng.choice(static_names)
            arg, _t = _atom(rng, bound)
            body.append(target + " = " + fn + "(" + arg + ")")
            env[target] = ns.return_type(fn)
    body.append("return " + target)
    return header, body, params


def make_corpus(n_programs=4000, seed=7, ns=None):
    """Returns (texts, programs) where texts are full program sources."""
    if ns is None:
        ns = build_namespace()
    rng = random.Random(seed)
    texts = []
    programs = []
    for _ in range(n_programs):
        header, body, params = make_program(rng, ns)
        texts.append(minipy.render(header, body))
        programs.append((header, body, params))
    return texts, programs, ns


def body_text(body_lines):
    return "".join(minipy.INDENT + line + NL for line in body_lines)

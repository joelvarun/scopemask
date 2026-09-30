"""MiniPy: a small but real Python subset, plus a character-level automaton for it.

The automaton is the ground truth for what "legal next character" means. It tracks
three things at once:

1. Syntax, meaning which statement form we are inside.
2. Scope, meaning which names are bound at this point in the program.
3. Coarse types, so attribute access can be checked against the receiver.

A token is legal at a state if and only if every character of the token can be
stepped through this automaton without hitting a dead end. That definition is the
contract both maskers in scopemask.masker must satisfy.

The same automaton runs in three modes, which are the three arms of the benchmark:

MODE_SYNTAX   grammar only. The output always parses. Any identifier is accepted
              anywhere, which is what a plain grammar-constrained decoder gives you.
MODE_LEXICON  grammar plus a flat project lexicon. Every identifier is a name that
              exists somewhere in the project, but scope and types are ignored. This
              is the realistic baseline: plausible code that reads the wrong variable.
MODE_STRICT   grammar plus scope plus coarse types. This is scopemask.
"""

import keyword

NL = chr(10)
INDENT = "    "

LETTERS = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
IDENT_START = frozenset(LETTERS + "_")
IDENT_CHARS = frozenset(LETTERS + DIGITS + "_")
OPERATORS = frozenset("+-*")
KEYWORDS = frozenset(keyword.kwlist) | frozenset(keyword.softkwlist)

MAX_IDENT = 20
MAX_INT_DIGITS = 3
MAX_BODY_LINES = 5

VALUE_TYPES = ("list", "str", "dict", "num")

MODE_STRICT = "strict"
MODE_SYNTAX = "syntax"
MODE_LEXICON = "lexicon"

# Receiver placeholders for the two relaxed modes.
ANY_TYPE = "any"
LEX_TYPE = "lex"

# Coarse method tables. Query-style methods only, so the inferred type stays honest.
METHODS = {
    "list": {"count": "num", "index": "num", "copy": "list", "pop": "num"},
    "str": {"upper": "str", "lower": "str", "strip": "str", "split": "list", "find": "num"},
    "dict": {"keys": "list", "values": "list", "copy": "dict", "get": "num"},
    "num": {"bit_length": "num", "conjugate": "num"},
}

ALL_METHODS = tuple(sorted({m for table in METHODS.values() for m in table}))

BUILTINS = {
    "len": "num", "sum": "num", "min": "num", "max": "num", "abs": "num",
    "round": "num", "sorted": "list", "list": "list", "dict": "dict",
    "str": "str", "int": "num", "float": "num", "set": "list",
    "any": "num", "all": "num", "repr": "str", "bool": "num",
}

RETURN_WORD = "return"


def legal_ident(s):
    return len(s) > 0 and s not in KEYWORDS


class Namespace:
    """The name environment the automaton checks against.

    In MODE_STRICT the names held here are the static, module level half of the legal
    set: large and identical for every program. That is what makes the prefix mask
    cache in scopemask.masker worth having, since only the small local half changes
    while a program is being decoded.
    """

    def __init__(self, functions, extra_names=(), mode=MODE_STRICT):
        self.mode = mode
        self.functions = dict(functions)
        names = set(self.functions) | set(BUILTINS)
        if mode == MODE_LEXICON:
            names |= set(extra_names)
        self.names = frozenset(names)
        if mode == MODE_SYNTAX:
            self.prefixes = frozenset()
        else:
            prefixes = set()
            for name in self.names:
                for i in range(1, len(name) + 1):
                    prefixes.add(name[:i])
            self.prefixes = frozenset(prefixes)
        if mode == MODE_LEXICON:
            self.callables = self.names
        else:
            self.callables = frozenset(set(self.functions) | set(BUILTINS))

    def syntax_view(self):
        return Namespace(self.functions, mode=MODE_SYNTAX)

    def lexicon_view(self, extra_names):
        return Namespace(self.functions, extra_names=extra_names, mode=MODE_LEXICON)

    def all_names(self):
        return self.names

    def is_callable(self, name):
        if self.mode == MODE_SYNTAX:
            return legal_ident(name)
        return name in self.callables

    def return_type(self, name):
        if name in self.functions:
            return self.functions[name]
        if name in BUILTINS:
            return BUILTINS[name]
        return "num"

    def has_prefix(self, s):
        if self.mode == MODE_SYNTAX:
            return True
        return s in self.prefixes

    def is_name(self, s):
        if self.mode == MODE_SYNTAX:
            return legal_ident(s)
        return s in self.names


class State:
    """Decoder-visible parse state. Treated as immutable: step returns a new one.

    env is shared by reference between states and only copied when a name is bound,
    which keeps the per-character clone cheap enough for the naive masker to be a
    fair baseline rather than a straw man.
    """

    __slots__ = ("mode", "buf", "numeric", "lit", "litpos", "litnext",
                 "target", "recv_type", "pending_type", "line", "done", "env")

    def __init__(self, mode, buf, numeric, lit, litpos, litnext,
                 target, recv_type, pending_type, line, done, env):
        self.mode = mode
        self.buf = buf
        self.numeric = numeric
        self.lit = lit
        self.litpos = litpos
        self.litnext = litnext
        self.target = target
        self.recv_type = recv_type
        self.pending_type = pending_type
        self.line = line
        self.done = done
        self.env = env

    def clone(self, **kw):
        return State(
            kw.get("mode", self.mode),
            kw.get("buf", self.buf),
            kw.get("numeric", self.numeric),
            kw.get("lit", self.lit),
            kw.get("litpos", self.litpos),
            kw.get("litnext", self.litnext),
            kw.get("target", self.target),
            kw.get("recv_type", self.recv_type),
            kw.get("pending_type", self.pending_type),
            kw.get("line", self.line),
            kw.get("done", self.done),
            kw.get("env", self.env),
        )

    def signature(self):
        return (self.mode, self.buf, self.numeric, self.lit, self.litpos,
                self.litnext, self.recv_type, self.line, self.done,
                tuple(sorted(self.env.items())))


def start_state(params):
    """params is a list of (name, type) pairs from the header, already in scope.

    Decoding starts at the first body line with its indent already emitted as part of
    the prompt, so the automaton begins in TARGET rather than waiting for the indent.
    """
    return State("TARGET", "", False, "", 0, "", None, None, None,
                 0, False, dict(params))


def _resolvable_prefix(st, ns, s):
    mode = ns.mode
    if mode == MODE_SYNTAX:
        return True
    if s in ns.prefixes:
        return True
    if mode == MODE_LEXICON:
        return False
    for name in st.env:
        if name.startswith(s):
            return True
    return False


def _resolvable_name(st, ns, s):
    mode = ns.mode
    if mode == MODE_SYNTAX:
        return legal_ident(s)
    if s in ns.names:
        return True
    if mode == MODE_LEXICON:
        return False
    return s in st.env


def _end_statement(st, name, typ):
    env = dict(st.env)
    env[name] = typ
    return st.clone(mode="LIT", buf="", numeric=False, lit=INDENT, litpos=0,
                    litnext="TARGET", target=None, recv_type=None,
                    pending_type=None, line=st.line + 1, env=env)


def _open_attribute(st, ns):
    if ns.mode == MODE_SYNTAX:
        if not legal_ident(st.buf):
            return None
        return st.clone(mode="ATTR", buf="", recv_type=ANY_TYPE)
    if ns.mode == MODE_LEXICON:
        if not _resolvable_name(st, ns, st.buf):
            return None
        return st.clone(mode="ATTR", buf="", recv_type=LEX_TYPE)
    typ = st.env.get(st.buf)
    if typ not in METHODS:
        return None
    return st.clone(mode="ATTR", buf="", recv_type=typ)


def _extend_name(st, ns, ch):
    if st.buf == "" and ch not in IDENT_START:
        return None
    if len(st.buf) >= MAX_IDENT:
        return None
    nb = st.buf + ch
    if not _resolvable_prefix(st, ns, nb):
        return None
    return st.clone(buf=nb)


def step(st, ch, ns):
    """Advance the automaton by one character. Returns a new State, or None if dead."""
    if st.done:
        return None
    mode = st.mode

    if mode == "LIT":
        if ch != st.lit[st.litpos]:
            return None
        pos = st.litpos + 1
        if pos < len(st.lit):
            return st.clone(litpos=pos)
        nxt = st.litnext
        if nxt == "ENDSTMT":
            return _end_statement(st, st.target, st.pending_type)
        return st.clone(mode=nxt, buf="", numeric=False, lit="", litpos=0, litnext="")

    if mode == "TARGET":
        if ch in IDENT_CHARS:
            if st.buf == "" and ch not in IDENT_START:
                return None
            if len(st.buf) >= MAX_IDENT:
                return None
            nb = st.buf + ch
            if st.line >= MAX_BODY_LINES and not RETURN_WORD.startswith(nb):
                return None
            return st.clone(buf=nb)
        if ch == " ":
            if st.buf == RETURN_WORD:
                return st.clone(mode="RET", buf="")
            if st.line >= MAX_BODY_LINES or not legal_ident(st.buf):
                return None
            return st.clone(mode="LIT", buf="", target=st.buf,
                            lit="= ", litpos=0, litnext="ATOM1")
        return None

    if mode == "RET":
        if ch in IDENT_CHARS:
            return _extend_name(st, ns, ch)
        if ch == NL:
            if not _resolvable_name(st, ns, st.buf):
                return None
            return st.clone(mode="DONE", buf="", done=True)
        return None

    if mode in ("ATOM1", "ATOM2", "GARG"):
        if st.buf == "":
            if ch in DIGITS:
                return st.clone(buf=ch, numeric=True)
            if ch in IDENT_START:
                return _extend_name(st, ns, ch)
            return None
        if st.numeric:
            if ch in DIGITS:
                if len(st.buf) >= MAX_INT_DIGITS:
                    return None
                return st.clone(buf=st.buf + ch)
            return _close_atom(st, ns, ch, True)
        if ch in IDENT_CHARS:
            return _extend_name(st, ns, ch)
        return _close_atom(st, ns, ch, False)

    if mode == "OP":
        if ch in OPERATORS:
            return st.clone(mode="LIT", buf="", lit=" ", litpos=0, litnext="ATOM2")
        return None

    if mode == "ATTR":
        return _step_attribute(st, ns, ch)

    return None


def _close_atom(st, ns, ch, numeric):
    """Terminate the atom being read in ATOM1, ATOM2 or GARG.

    A name atom may end three ways, and each has its own legality rule: as a value it
    must resolve, before a dot it must be a receiver, before an open paren it must be
    callable.
    """
    mode = st.mode
    if mode == "ATOM1":
        if ch == ".":
            return _open_attribute(st, ns)
        if ch == "(":
            if numeric or not ns.is_callable(st.buf):
                return None
            return st.clone(mode="GARG", buf="", numeric=False,
                            pending_type=ns.return_type(st.buf))
        if ch == " ":
            if not numeric and not _resolvable_name(st, ns, st.buf):
                return None
            return st.clone(mode="OP", buf="", numeric=False)
        return None
    if not numeric and not _resolvable_name(st, ns, st.buf):
        return None
    if mode == "ATOM2":
        if ch == NL:
            return _end_statement(st, st.target, "num")
        return None
    if ch == ")":
        return st.clone(mode="LIT", buf="", numeric=False,
                        lit=NL, litpos=0, litnext="ENDSTMT")
    return None


def _step_attribute(st, ns, ch):
    recv = st.recv_type
    if recv == ANY_TYPE:
        if ch in IDENT_CHARS:
            if st.buf == "" and ch not in IDENT_START:
                return None
            if len(st.buf) >= MAX_IDENT:
                return None
            return st.clone(buf=st.buf + ch)
        if ch == "(":
            if not legal_ident(st.buf):
                return None
            return st.clone(mode="LIT", buf="", pending_type="num",
                            lit=")" + NL, litpos=0, litnext="ENDSTMT")
        return None
    table = ALL_METHODS if recv == LEX_TYPE else METHODS[recv]
    if ch in IDENT_CHARS:
        if len(st.buf) >= MAX_IDENT:
            return None
        nb = st.buf + ch
        for m in table:
            if m.startswith(nb):
                return st.clone(buf=nb)
        return None
    if ch == "(":
        if st.buf not in table:
            return None
        typ = "num" if recv == LEX_TYPE else METHODS[recv][st.buf]
        return st.clone(mode="LIT", buf="", pending_type=typ,
                        lit=")" + NL, litpos=0, litnext="ENDSTMT")
    return None


def step_text(st, text, ns):
    for ch in text:
        st = step(st, ch, ns)
        if st is None:
            return None
    return st


def render(header, body_lines):
    return header + NL + "".join(INDENT + line + NL for line in body_lines)

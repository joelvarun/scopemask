"""An independent validity checker, written over Python's own ast module.

scopemask.minipy decides legality with a character automaton. This module decides it
again from the parsed syntax tree, with no shared code path. The benchmark compares
the two on every sampled program, so a claim that constrained output is valid does
not rest on the constraint engine grading its own homework.
"""

import ast

from .minipy import METHODS, NL


def _resolves(name, env, ns):
    return name in env or ns.is_name(name)


def _atom_type(node, env, ns):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return "num", None
    if isinstance(node, ast.Name):
        if not _resolves(node.id, env, ns):
            return None, "unbound name " + node.id
        return env.get(node.id, "function"), None
    return None, "atom form"


def _infer(node, env, ns):
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            return None, "operator"
        for side in (node.left, node.right):
            _t, err = _atom_type(side, env, ns)
            if err:
                return None, err
        return "num", None
    if isinstance(node, ast.Call):
        if node.keywords:
            return None, "keywords"
        func = node.func
        if isinstance(func, ast.Attribute):
            if node.args:
                return None, "method args"
            if not isinstance(func.value, ast.Name):
                return None, "receiver form"
            recv = env.get(func.value.id)
            if recv not in METHODS:
                return None, "receiver type"
            table = METHODS[recv]
            if func.attr not in table:
                return None, "no attribute " + func.attr
            return table[func.attr], None
        if isinstance(func, ast.Name):
            if not ns.is_callable(func.id):
                return None, "not callable " + func.id
            if len(node.args) != 1:
                return None, "arity"
            _t, err = _atom_type(node.args[0], env, ns)
            if err:
                return None, err
            return ns.return_type(func.id), None
        return None, "call form"
    return None, "rhs form"


def check(header, body_text, params, ns):
    """Returns (ok, reason). params gives the declared coarse types of the header."""
    src = header + NL + body_text
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return False, "syntax"
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False, "module shape"
    fn = tree.body[0]
    env = dict(params)
    if len(fn.args.args) != len(params):
        return False, "header arity"
    returned = False
    for stmt in fn.body:
        if returned:
            return False, "statement after return"
        if isinstance(stmt, ast.Return):
            if not isinstance(stmt.value, ast.Name):
                return False, "return form"
            if not _resolves(stmt.value.id, env, ns):
                return False, "unbound return " + stmt.value.id
            returned = True
            continue
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            return False, "statement form"
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            return False, "target form"
        typ, err = _infer(stmt.value, env, ns)
        if err:
            return False, err
        env[target.id] = typ
    if not returned:
        return False, "no return"
    return True, "ok"


def audit(header, body_text, params, ns):
    """Count identifier use positions and how many of them fail to resolve.

    Unlike check, this does not stop at the first error: every use position is scored
    independently, so the per-position error rate can be reported alongside the
    per-program one. Attribute accesses count as use positions too, and fail when the
    receiver type does not carry that method.
    """
    src = header + NL + body_text
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return 0, 0
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return 0, 0
    fn = tree.body[0]
    env = dict(params)
    uses = 0
    bad = 0
    for stmt in fn.body:
        if isinstance(stmt, ast.Return):
            if isinstance(stmt.value, ast.Name):
                uses += 1
                if not _resolves(stmt.value.id, env, ns):
                    bad += 1
            continue
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        value = stmt.value
        for node in ast.walk(value):
            if isinstance(node, ast.Name):
                uses += 1
                if not _resolves(node.id, env, ns):
                    bad += 1
            elif isinstance(node, ast.Attribute):
                uses += 1
                recv = None
                if isinstance(node.value, ast.Name):
                    recv = env.get(node.value.id)
                if recv not in METHODS or node.attr not in METHODS[recv]:
                    bad += 1
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            typ, _err = _infer(value, env, ns)
            env[target.id] = typ if typ else "num"
    return uses, bad

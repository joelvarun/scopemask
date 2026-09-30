"""Side by side demo: the same generator, with and without scope in the mask.

Run:  python examples/demo.py
"""

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scopemask.decode import Engine

RULE = "-" * 78


def show(title, header, body, verdict):
    automaton_ok, ast_ok, reason = verdict
    print(title)
    print(header)
    print(body, end="")
    if ast_ok:
        print("  verdict: resolves (every name is in scope, every method exists)")
    else:
        print("  verdict: rejected, " + reason)
    assert automaton_ok == ast_ok, "graders disagree"
    print()


def explain_state(eng, params, prompt_ids, rng, copy_weight):
    """Decode a few tokens, then show what each mask allows at the same state."""
    states = []
    eng.sample(params, prompt_ids, rng, copy_weight=copy_weight,
               arm="scopemask", collect=states)
    lexicon = eng.maskers["lexicon"]
    strict_masker = eng.maskers["scopemask"]
    best = None
    for state in states:
        if state.mode not in ("ATOM1", "ATOM2", "GARG", "RET") or state.numeric:
            continue
        wide = int(lexicon.mask(state).sum())
        tight = int(strict_masker.mask(state).sum())
        if wide == 0:
            continue
        score = wide - tight
        if best is None or score > best[0]:
            best = (score, state, wide, tight)
    if best is None:
        return
    _score, chosen, wide, tight = best
    lex = lexicon.mask(chosen)
    tight_mask = strict_masker.mask(chosen)
    where = "after " + repr(chosen.buf) if chosen.buf else "at the start of a name"
    print(RULE)
    print("Widest gap between the two masks in that program, " + where
          + ", with locals " + ", ".join(sorted(chosen.env)) + ":")
    print("  grammar plus lexicon admits " + str(wide) + " of " + str(eng.V)
          + " vocabulary entries")
    share = 100.0 * (wide - tight) / wide
    print("  scopemask admits " + str(tight) + ", removing " + str(wide - tight)
          + " (" + format(share, ".1f") + " percent) that cannot continue a name")
    print("  the mask barely moves at any single step, and that is the point:")
    print("  the removed entries are exactly the ones the generator gets wrong")
    removed = [eng.tok.vocab[i] for i in np.nonzero(lex & ~tight_mask)[0][:12]]
    print("  examples of what scopemask removes here: "
          + ", ".join(repr(r) for r in removed))
    print()


def main():
    print("building corpus, tokenizer and generator")
    eng = Engine()
    composition = eng.maskers["scopemask"].composition()
    print("vocabulary " + str(composition["vocab_size"]) + " entries ("
          + str(composition["pure_ident_tokens"]) + " identifier-only, "
          + str(composition["mixed_tokens"]) + " mixed), static namespace "
          + str(len(eng.ns.functions)) + " functions")
    print()

    copy_weight = 0.85
    headers = random.Random(4)
    for _ in range(3):
        header, params, prompt_ids = eng.new_task(headers)
        types = ", ".join(n + " is " + t for n, t in params)
        print(RULE)
        print("task: complete this function. Parameter types: " + types)
        print()
        for arm, title in (("lexicon", "before, grammar plus project lexicon:"),
                           ("scopemask", "after, scopemask:")):
            rng = np.random.default_rng(12)
            steps, body = eng.sample(params, prompt_ids, rng,
                                     copy_weight=copy_weight, arm=arm)
            show(title + " " + str(steps) + " tokens", header, body,
                 eng.is_valid(header, params, body))

    header, params, prompt_ids = eng.new_task(headers)
    explain_state(eng, params, prompt_ids, np.random.default_rng(7), copy_weight)


if __name__ == "__main__":
    main()

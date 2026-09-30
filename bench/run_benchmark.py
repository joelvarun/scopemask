"""Run the scopemask benchmark and write docs/results.json.

Three things are measured, all on the same generator and the same held-out headers:

1. Tokens generated per valid program, for the baseline (grammar plus project
   lexicon, then reject and resample) versus scopemask (grammar plus scope plus
   coarse types in the mask, one pass). This is the headline number.
2. Valid-program rate as a function of program length, which is why rejection
   sampling is the wrong tool here: the rate decays geometrically in the number of
   identifier use positions.
3. Mask construction time per decoded token, naive per-token automaton replay versus
   the static/dynamic factorization.

Every sampled program is graded twice, once by the character automaton and once by an
independent ast-based resolver, and the run fails if the two ever disagree.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scopemask import minipy
from scopemask.decode import Engine
from scopemask.validate import audit

LEVELS = [("weak", 0.50), ("medium", 0.85), ("strong", 0.93)]
HEADER_SEED = 11
SAMPLE_SEED = 3


def run_arm(eng, arm, copy_weight, n_programs, target_valid=None,
            collect_states=None):
    """Sample until target_valid programs resolve, or until n_programs attempts.

    Stopping on a valid count rather than an attempt count keeps the precision of
    tokens-per-valid comparable across generator strengths: a weak generator simply
    needs more attempts to reach the same number of valid programs.
    """
    rng = np.random.default_rng(SAMPLE_SEED)
    headers = random.Random(HEADER_SEED)
    tokens = 0
    valid = 0
    uses = 0
    bad_uses = 0
    attempts = 0
    by_uses = {}
    for _ in range(n_programs):
        if target_valid is not None and valid >= target_valid:
            break
        attempts += 1
        header, params, prompt = eng.new_task(headers)
        steps, body = eng.sample(params, prompt, rng, copy_weight=copy_weight,
                                 arm=arm, collect=collect_states)
        tokens += steps
        automaton_ok, ast_ok, reason = eng.is_valid(header, params, body)
        if automaton_ok != ast_ok:
            raise AssertionError(
                "graders disagree: automaton=" + str(automaton_ok)
                + " ast=" + str(ast_ok) + " (" + reason + ") on" + minipy.NL + body)
        valid += 1 if ast_ok else 0
        n_uses, n_bad = audit(header, body, params, eng.ns)
        uses += n_uses
        bad_uses += n_bad
        bucket = by_uses.setdefault(n_uses, [0, 0])
        bucket[0] += 1
        bucket[1] += 1 if ast_ok else 0
    return {
        "arm": arm,
        "copy_weight": copy_weight,
        "attempts": attempts,
        "valid": valid,
        "tokens": tokens,
        "tokens_per_attempt": tokens / attempts,
        "valid_rate": valid / attempts,
        "tokens_per_valid": (tokens / valid) if valid else None,
        "use_positions": uses,
        "unresolved_use_positions": bad_uses,
        "use_error_rate": (bad_uses / uses) if uses else None,
        "by_uses": {str(k): v for k, v in sorted(by_uses.items())},
    }


def time_masks(eng, states, naive_budget):
    """Per-state mask construction time for both builders, in microseconds."""
    fast = eng.maskers["scopemask"]
    naive = eng.naive["scopemask"]
    # The static prefix cache is a one-time index, so warm it before timing steady
    # state, and report the warm-up separately.
    warm_start = time.perf_counter()
    fast.cache_hits = 0
    fast.cache_misses = 0
    for st in states:
        fast.mask(st)
    warm_seconds = time.perf_counter() - warm_start
    misses = fast.cache_misses

    fast.step_calls = 0
    fast.index_lookups = 0
    start = time.perf_counter()
    for st in states:
        fast.mask(st)
    fast_seconds = time.perf_counter() - start
    fast_steps = fast.step_calls / len(states)
    fast_lookups = fast.index_lookups / len(states)

    subset = states[:naive_budget]
    naive.step_calls = 0
    start = time.perf_counter()
    for st in subset:
        naive.mask(st)
    naive_seconds = time.perf_counter() - start
    naive_steps = naive.step_calls / len(subset)

    mismatches = sum(
        0 if np.array_equal(fast.mask(st), naive.mask(st)) else 1 for st in subset)
    if mismatches:
        raise AssertionError(str(mismatches) + " mask disagreements in timing subset")

    return {
        "states_timed": len(states),
        "naive_states_timed": len(subset),
        "fast_us": 1e6 * fast_seconds / len(states),
        "naive_us": 1e6 * naive_seconds / len(subset),
        "fast_automaton_steps": fast_steps,
        "naive_automaton_steps": naive_steps,
        "fast_index_lookups": fast_lookups,
        "step_reduction": naive_steps / fast_steps,
        "speedup": (naive_seconds / len(subset)) / (fast_seconds / len(states)),
        "cold_fast_us": 1e6 * warm_seconds / len(states),
        "cache_misses_on_cold_pass": misses,
        "verified_states": len(subset),
    }


def main():
    parser = argparse.ArgumentParser(description="run the scopemask benchmark")
    parser.add_argument("--programs", type=int, default=200,
                        help="sampled programs for the scopemask arm per strength")
    parser.add_argument("--target-valid", type=int, default=25,
                        help="valid programs to collect per baseline measurement")
    parser.add_argument("--max-attempts", type=int, default=1500,
                        help="attempt cap per baseline measurement")
    parser.add_argument("--syntax-programs", type=int, default=600,
                        help="sampled programs for the grammar-only reference arm")
    parser.add_argument("--corpus", type=int, default=4000)
    parser.add_argument("--merges", type=int, default=2000)
    parser.add_argument("--naive-budget", type=int, default=150,
                        help="states to time the naive masker on")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = args.out or os.path.join(root, "docs", "results.json")

    t0 = time.perf_counter()
    eng = Engine(n_programs=args.corpus, n_merges=args.merges)
    setup_seconds = time.perf_counter() - t0
    composition = eng.maskers["scopemask"].composition()
    print("corpus and tokenizer ready in " + str(round(setup_seconds, 1)) + " s")
    print("vocabulary " + str(composition["vocab_size"])
          + " entries, " + str(composition["pure_ident_tokens"])
          + " identifier-only, " + str(composition["mixed_tokens"]) + " mixed")
    print("static namespace " + str(len(eng.ns.functions)) + " module level functions")

    levels = []
    states = []
    for label, weight in LEVELS:
        baseline = run_arm(eng, "lexicon", weight, args.max_attempts,
                           target_valid=args.target_valid)
        want = None if states else 800
        collect = states if want else None
        after = run_arm(eng, "scopemask", weight, args.programs,
                        collect_states=collect)
        if want:
            del states[want:]
        ratio = baseline["tokens_per_valid"] / after["tokens_per_valid"]
        levels.append({"label": label, "copy_weight": weight,
                       "baseline": baseline, "scopemask": after,
                       "tokens_per_valid_ratio": ratio})
        print(label + " generator: baseline "
              + str(baseline["valid"]) + " valid in " + str(baseline["attempts"])
              + " attempts, " + str(round(baseline["tokens_per_valid"], 1))
              + " tokens per valid program; scopemask "
              + str(round(after["tokens_per_valid"], 1)) + " ("
              + str(round(ratio, 2)) + "x fewer)", flush=True)

    syntax_ref = run_arm(eng, "syntax", LEVELS[1][1], args.syntax_programs)
    print("grammar-only reference: " + str(syntax_ref["valid"]) + " valid in "
          + str(syntax_ref["attempts"]) + " attempts", flush=True)

    timing = time_masks(eng, states, args.naive_budget)
    print("mask build per decoded token: naive "
          + str(round(timing["naive_us"], 1)) + " us, factorized "
          + str(round(timing["fast_us"], 1)) + " us ("
          + str(round(timing["speedup"], 1)) + "x)", flush=True)
    print("automaton character steps per mask: naive "
          + str(round(timing["naive_automaton_steps"], 1)) + ", factorized "
          + str(round(timing["fast_automaton_steps"], 1)) + " plus "
          + str(round(timing["fast_index_lookups"], 1))
          + " vocabulary lookups", flush=True)

    results = {
        "composition": composition,
        "static_functions": len(eng.ns.functions),
        "lm_order": eng.order,
        "corpus_programs": args.corpus,
        "setup_seconds": setup_seconds,
        "levels": levels,
        "syntax_reference": syntax_ref,
        "mask_timing": timing,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("wrote " + out_path)


if __name__ == "__main__":
    main()

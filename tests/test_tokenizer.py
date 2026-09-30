"""Tokenizer invariants the decoder depends on."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scopemask import corpus, minipy
from scopemask.tokenizer import EOS, Tokenizer, pretokenize

TEXTS, PROGRAMS, NS = corpus.make_corpus(600)
TOK = Tokenizer.train(TEXTS, n_merges=600)


def test_encode_decode_roundtrip():
    for text in TEXTS[:100]:
        assert TOK.decode(TOK.encode(text, add_eos=True)) == text


def test_prompt_and_completion_tokenize_the_same_as_the_whole_program():
    """The decoder is handed a prompt and continues from it, so the split must
    not change the tokenization of either side."""
    for header, body, _params in PROGRAMS[:100]:
        full = minipy.render(header, body)
        prompt = header + minipy.NL + minipy.INDENT
        rest = full[len(prompt):]
        assert TOK.encode(prompt) + TOK.encode(rest) == TOK.encode(full)


def test_pretokenize_partitions_the_text():
    for text in TEXTS[:50]:
        assert "".join(pretokenize(text)) == text


def test_every_character_has_a_single_entry():
    chars = set()
    for text in TEXTS:
        chars.update(text)
    for ch in chars:
        assert ch in TOK.index


def test_end_of_sequence_is_reserved():
    assert TOK.vocab[TOK.eos_id] == EOS
    assert TOK.decode([TOK.eos_id]) == ""
    for text in TEXTS[:20]:
        assert EOS not in text


def test_identifiers_really_do_split_into_pieces():
    """Scope-aware masking is only interesting when names span several entries."""
    names = sorted(NS.functions)[:300]
    pieces = [len(TOK.encode(name)) for name in names]
    assert sum(pieces) / len(pieces) > 1.5

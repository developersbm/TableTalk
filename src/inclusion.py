#done by Sebastian Bastida Marin, Rei Shindo, Logan Mifflin
#This file marks which tokens or lines are kept after fixing

#token_inclusion marks original tokens as kept or changed
#line_inclusion marks original lines as kept or changed

from difflib import SequenceMatcher
from typing import List, Callable

#compute token inclusion flags for original vs final code, using model tokenizer if possible
def token_inclusion(
    initial_tokens: List[str],
    final_code: str,
    tokenizer: Callable[[str], List[str]],
) -> List[int]:
    if not initial_tokens:
        return []

    final_tokens = tokenizer(final_code)
    flags = [0] * len(initial_tokens)

    matcher = SequenceMatcher(None, initial_tokens, final_tokens)
    opcodes = matcher.get_opcodes()

    for op_idx, (tag, i1, i2, j1, j2) in enumerate(opcodes):
        if tag == "equal":
            has_prior_insertion = (
                op_idx > 0 and opcodes[op_idx - 1][0] == "insert"
            )
            label = 0 if has_prior_insertion else 1
            for i in range(i1, i2):
                flags[i] = label

    return flags

#compute line inclusion flags for original vs final code
def line_inclusion(initial_code: str, final_code: str) -> List[int]:
    if not initial_code:
        return []

    init_lines = initial_code.splitlines()
    final_lines = final_code.splitlines()
    flags = [0] * len(init_lines)

    matcher = SequenceMatcher(None, init_lines, final_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                flags[i] = 1

    return flags

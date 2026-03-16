#done by Sebastian Bastida Marin, Rei Shindo, Logan Mifflin
#This file handles loading the huggingface tokenizer and converting text into tokens

#tokenize_result: structure to hold the output of tokenization like text, ids, and tokens
#load_tokenizer: downloads and initializes the specified tokenizer from huggingface
#tokenize: takes standard text and converts it to model token ids and strings
#decode: converts tokenizer ids back into normal string text

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase

GEMMA3_27B_MODEL_ID: str = "google/gemma-3-27b-it"
@dataclass
class TokenizeResult:
    #structure to hold the output of tokenization
    text: str
    token_ids: List[int]
    tokens: List[str]
    num_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_tokens = len(self.token_ids)

    def __repr__(self) -> str:
        preview = self.tokens[:10]
        suffix = " ..." if self.num_tokens > 10 else ""
        return (
            f"TokenizeResult(num_tokens={self.num_tokens}, "
            f"tokens={preview}{suffix})"
        )

#downloads and initializes the specified tokenizer from huggingface
def load_tokenizer(
    model_id: str = GEMMA3_27B_MODEL_ID,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> PreTrainedTokenizerBase:
    hf_token = token or os.environ.get("HF_TOKEN")

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        token=hf_token,
    )
    return tokenizer

#takes standard text and converts it to model token ids and strings
def tokenize(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    add_special_tokens: bool = True,
) -> TokenizeResult:
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__!r}")

    encoding = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_tensors=None,
    )
    token_ids = cast(List[int], encoding["input_ids"])

    tokens = [
        cast(str, tokenizer.convert_ids_to_tokens(tok_id))
        for tok_id in token_ids
    ]

    return TokenizeResult(text=text, token_ids=token_ids, tokens=tokens)



#converts tokenizer ids back into normal string text
def decode(
    token_ids: List[int],
    tokenizer: PreTrainedTokenizerBase,
    skip_special_tokens: bool = True,
) -> str:
    return cast(str, tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens))

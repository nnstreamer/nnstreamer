##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2026 Samsung Electronics
#
# @file    generateModel.py
# @brief   Regenerate the language model fixture of the executorch-llama filter
# @author  MyungJoo Ham <myungjoo.ham@samsung.com>
#
# The model and its tokenizer are committed, so this only runs when they have
# to be rebuilt. Needs the ExecuTorch python package and the torch release it
# pins:
#   pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
#   pip install executorch==1.4.1
#
# The tokenizer is a tiktoken file holding the 256 single bytes, so any text
# encodes to its bytes. The ExecuTorch LLM runner adds the 256 Llama 3 special
# tokens after them, which makes <|end_of_text|> the token 257.
#
# The model is not trained. It predicts the next token from the last one
# alone, with a margin wide enough that sampling at temperature 0.8 always
# picks it: a lowercase letter is followed by the next letter and 'z' by
# <|end_of_text|>, a digit by the next digit (wrapping around, so it never
# ends), and anything else by 'a'. It takes the whole sequence on each call
# (no KV cache), which is what its metadata methods tell the runner. Its
# context is longer than the 128 tokens the filter stops at, so that the limit
# the tests see is the filter's own.
#
# Usage: python3 generateModel.py [output-directory]   (in this directory)

"""Regenerate the executorch-llama fixture: a .pte model and its tokenizer."""

import base64
import os
import sys

import torch
from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim, export

NUM_BYTE_TOKENS = 256
VOCAB_SIZE = NUM_BYTE_TOKENS + 256
EOS_ID = NUM_BYTE_TOKENS + 1
MAX_SEQ_LEN = 256


class NextLetter(torch.nn.Module):
    """Predicts the token that follows the last one of the sequence."""

    def forward(self, tokens):
        """Return the logits of the next token, shaped [1, VOCAB_SIZE]."""
        last = tokens[:, -1]
        is_lower = (last >= ord('a')) & (last < ord('z'))
        is_digit = (last >= ord('0')) & (last <= ord('9'))
        nxt = torch.full_like(last, ord('a'))
        nxt = torch.where(is_digit, ord('0') + torch.remainder(last - ord('0') + 1, 10), nxt)
        nxt = torch.where(last == ord('z'), torch.full_like(last, EOS_ID), nxt)
        nxt = torch.where(is_lower, last + 1, nxt)
        vocab = torch.arange(VOCAB_SIZE, dtype=torch.long).unsqueeze(0)
        return (vocab == nxt.unsqueeze(-1)).to(torch.float32) * 100.0


def write_tokenizer(path):
    """Write a tiktoken vocabulary of the 256 single bytes."""
    with open(path, 'w', encoding='ascii', newline='\n') as f:
        for rank in range(NUM_BYTE_TOKENS):
            f.write(base64.b64encode(bytes([rank])).decode('ascii') + ' ' + str(rank) + '\n')


def main():
    """Export the model and write the tokenizer next to it."""
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        '..', 'test_models', 'models')

    example = (torch.tensor([[ord('a'), ord('b')]], dtype=torch.long),)
    seq = Dim('seq', min=1, max=MAX_SEQ_LEN)
    program = export(NextLetter(), example, dynamic_shapes=({1: seq},))
    edge = to_edge_transform_and_lower(program, constant_methods={
        'get_max_seq_len': MAX_SEQ_LEN,
        'get_max_context_len': MAX_SEQ_LEN,
        'get_vocab_size': VOCAB_SIZE,
        'use_kv_cache': False,
        'enable_dynamic_shape': True,
        'get_eos_ids': [EOS_ID],
    })

    with open(os.path.join(out_dir, 'llm_next_letter.pte'), 'wb') as f:
        f.write(edge.to_executorch().buffer)
    write_tokenizer(os.path.join(out_dir, 'llm_next_letter_tokenizer.model'))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-only
# Copyright (C) 2022 Samsung Electronics
#
# @file generate_random_text.py
# @author Yechan Choi <yechan9.choi@samsung.com>
# @date   1 Aug 2022
# @brief Generate Random Text File
#

from argparse import ArgumentParser
from random import choice, randint
import string


def generate_random_text(length: int,
                         file_name: str) -> None:
    chars = string.ascii_letters + string.digits + '\n'
    text = ''.join([choice(chars) for _ in range(length)])
    with open(file_name, "w") as f:
        f.write(text)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--file_name', type=str, default="random_text.txt")
    parser.add_argument('--length', type=int, default=randint(1e3, 1e6))

    args = parser.parse_args()

    generate_random_text(args.length, args.file_name)

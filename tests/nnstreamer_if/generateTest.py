#!/usr/bin/env python3

##
# SPDX-License-Identifier: LGPL-2.1-only
#
# Copyright (C) 2026 Samsung Electronics
#
# @file generateTest.py
# @brief Generate the input frames and golden results of the tensor_if SSAT cases
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
#
# Each input frame holds two uint8 tensors: tensor 0 is 2:2 and tensor 1 is 3.
# The averages of tensor 0 are whole numbers, so that the uint8 cast of the
# average does not depend on rounding, and two of them are the bounds 10 and 20
# of the range cases.

FRAMES = [
    ([5, 1, 2, 4], [100, 101, 102]),
    ([12, 8, 16, 12], [110, 111, 112]),
    ([20, 20, 20, 20], [120, 121, 122]),
    ([30, 10, 10, 10], [130, 131, 132]),
    ([9, 40, 30, 21], [140, 141, 142]),
    ([10, 18, 9, 3], [150, 151, 152]),
]


##
# @brief Write the frames picked by a condition into a golden file
# @param filename The golden file to write
# @param cond A function of a frame that tells if the frame is in the output
# @param pick A function of a frame that returns the bytes of the output frame
def save_golden(filename, cond, pick):
    with open(filename, 'wb') as file:
        for frame in FRAMES:
            if cond(frame):
                file.write(bytes(pick(frame)))


##
# @brief Return the average of tensor 0 of a frame
# @param frame The input frame
def average(frame):
    return sum(frame[0]) // len(frame[0])


for i, f in enumerate(FRAMES):
    with open('input_%02d.dat' % i, 'wb') as out:
        out.write(bytes(f[0] + f[1]))

# A_VALUE of [0][0] of tensor 0 >= 10, PASSTHROUGH or SKIP
save_golden('passthrough_ge.golden', lambda f: f[0][0] >= 10,
            lambda f: f[0] + f[1])
# A_VALUE of [0][0] of tensor 0 against 10 with the other operators
save_golden('passthrough_gt.golden', lambda f: f[0][0] > 10,
            lambda f: f[0] + f[1])
save_golden('passthrough_le.golden', lambda f: f[0][0] <= 10,
            lambda f: f[0] + f[1])
save_golden('passthrough_eq.golden', lambda f: f[0][0] == 10,
            lambda f: f[0] + f[1])
save_golden('passthrough_ne.golden', lambda f: f[0][0] != 10,
            lambda f: f[0] + f[1])
# Every frame, PASSTHROUGH
save_golden('passthrough_all.golden', lambda f: True, lambda f: f[0] + f[1])
# A_VALUE of [1][1] of tensor 0 < 10, PASSTHROUGH or SKIP
save_golden('passthrough_lt.golden', lambda f: f[0][3] < 10,
            lambda f: f[0] + f[1])
# TENSOR_AVERAGE_VALUE of tensor 0 in [10, 20], TENSORPICK 1 or SKIP
save_golden('pick_range_inclusive.golden',
            lambda f: 10 <= average(f) <= 20, lambda f: f[1])
# TENSOR_AVERAGE_VALUE of tensor 0 in (10, 20), TENSORPICK 1 or SKIP
save_golden('pick_range_exclusive.golden',
            lambda f: 10 < average(f) < 20, lambda f: f[1])
# A_VALUE of [0][0] of tensor 0 >= 10, TENSORPICK 1,0 or TENSORPICK 0
save_golden('pick_then.golden', lambda f: f[0][0] >= 10,
            lambda f: f[1] + f[0])
save_golden('pick_else.golden', lambda f: f[0][0] < 10, lambda f: f[0])

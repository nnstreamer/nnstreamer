#!/usr/bin/env python
##
# SPDX-License-Identifier: LGPL-2.1-only
#
# GTest / SSAT Test Result Aggregator
# Copyright (c) 2019 Samsung Electronics
#
##
# @file   count_test_cases.py
# @brief  A unit test result aggregator
# @author MyungJoo Ham <myungjoo.ham@samsung.com>
# @date   18 Dec 2019
# @bug    No known bugs
# @todo   WIP

import sys
import re
import os


##
# @brief read results of GTest's result xml file
def read_gtest_xml(filename):
    neg = 0
    with open(filename, 'r') as fd:
        lines = fd.readlines()
        for line in lines:
            res = re.match(r'\s*<testcase name="([^"]+_n)"', line)
            if res:
                neg = neg + 1
        fd.seek(0)

        lines = fd.readlines()
        for line in lines:
            res = re.match(r'<testsuites tests="(\d+)" failures="(\d+)" disabled="(\d+)"', line)
            if res:
                return (int(res.group(1)), int(res.group(1)) - int(res.group(2)) - int(res.group(3)),
                        int(res.group(2)), int(res.group(3)), neg)
    return (0, 0, 0, 0, 0)


##
# @brief read results of SSAT summary file
def read_ssat(filename):
    with open(filename, 'r') as fd:
        lines = fd.readlines()
        for line in lines:
            res = re.match(r'passed=(\d+), failed=(\d+), ignored=(\d+), negative=(\d+)', line)
            if res:
                return (int(res.group(1)) + int(res.group(2)) + int(res.group(3)), int(res.group(1)),
                        int(res.group(2)), int(res.group(3)), int(res.group(4)))
    return (0, 0, 0, 0, 0)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage:")
        print(" $ " + sys.argv[0] + " <gtest xml path> <ssat summary path>")
        print("")
        sys.exit(1)

    tg = 0
    pg = 0
    fg = 0
    ig = 0
    ng = 0
    posg = 0

    for r, d, f in os.walk(sys.argv[1]):
        for file in f:
            if os.path.splitext(file)[1] == '.xml':
                (t, p, f, i, n) = read_gtest_xml(os.path.join(r, file))
                tg = tg + t
                pg = pg + p
                fg = fg + f
                ig = ig + i
                ng = ng + n
    posg = pg + fg + ig - ng

    (t, p, f, i, n) = read_ssat(sys.argv[2])

    print("GTest (total " + str(tg) + " cases)")
    print("  Passed: " + str(pg) + " / Failed: " + str(fg) + " / Ignored: " + str(ig) +
          " | Positive: " + str(posg) + " / Negative: " + str(ng))
    print("SSAT (total " + str(t) + " cases)")
    print("  Passed: " + str(p) + " / Failed: " + str(f) + " / Ignored: " + str(i) +
          " | Positive: " + str(t - n) + " / Negative: " + str(n))
    print("Grand Total: " + str(pg + t) + " cases (negatives : " + str(ng + n) + ")")
    print("  Passed: " + str(pg + p) + " / Failed: " + str(fg + f) + " / Ignored: " + str(ig + i))
    sys.exit(0)

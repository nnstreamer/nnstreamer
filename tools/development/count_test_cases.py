#!/usr/bin/env python
##
# GTest / SSAT Test Result Aggregator
# Copyright (c) 2019 Samsung Electronics
#
# You may use this under either LGPL 2.1+ or Apache-2.0.

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

## @brief read results of GTest's result xml file
def readGtestXml(filename):
    neg = 0
    try:
        with open(filename, "r") as f:
            r = f.readlines()
            for line in r:
                res = re.match(r'\s*<testcase name="([^"]+_n)"', line)
                if res:
                    neg = neg + 1
            f.seek(0)

            r = f.readlines()
            for line in r:
                res = re.match(r'<testsuites tests="(\d+)" failures="(\d+)" disabled="(\d+)"', line)
                if res:
                    return (int(res.group(1)), int(res.group(1)) - int(res.group(2)) - int(res.group(3)), int(res.group(2)), int(res.group(3)), neg)
    except:
        print("No gtest results.")
    return (0, 0, 0, 0, 0)

## @brief read results of SSAT summary file
def readSSAT(filename):
    try:
        with open(filename, "r") as f:
            r = f.readlines()
            for line in r:
                res = re.match(r'passed=(\d+), failed=(\d+), ignored=(\d+)', line)
                if res:
                    return (int(res.group(1)) + int(res.group(2)) + int(res.group(3)), int(res.group(1)), int(res.group(2)), int(res.group(3)))
    except:
        print("No SSAT results.")
    return (0, 0, 0, 0)

def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print(" $ "+sys.argv[0]+" <gtest xml path> <ssat summary path>")
        print("")
        return 1

    tg = 0
    pg = 0
    fg = 0
    ig = 0
    ng = 0
    posg = 0

    for r, d, f in os.walk(sys.argv[1]):
        for file in f:
            if os.path.splitext(file)[1] == '.xml':
                (t, p, f, i, n) = readGtestXml(os.path.join(r, file))
                tg = tg + t
                pg = pg + p
                fg = fg + f
                ig = ig + i
                ng = ng + n
    posg = pg + fg + ig - ng

    (t, p, f, i) = readSSAT(sys.argv[2])

    print("GTest (total " + str(tg) + " cases)")
    print("  Passed: " + str(pg) + " / Failed: " + str(fg) + " / Ignored: " + str(ig) + " | Positive: " + str(posg) + " / Negative: " + str(ng))
    print("SSAT (total " + str(t) + " cases)")
    print("  Passed: " + str(p) + " / Failed: " + str(f) + " / Ignored: " + str(i))
    print("Grand Total: " + str(pg + t) + " cases")
    print("  Passed: " + str(pg+p) + " / Failed: " + str(fg + f) + " / Ignored: " + str(ig + i))
    return 0

main()

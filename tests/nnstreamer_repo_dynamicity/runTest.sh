#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author MyungJoo Ham <myungjoo.ham@gmail.com>
## @date Nov 01 2018
## @brief SSAT Test Cases for NNStreamer
##
if [[ "$SSATAPILOADED" != "1" ]]; then
    SILENT=0
    INDEPENDENT=1
    search="ssat-api.sh"
    source $search
    printf "${Blue}Independent Mode${NC}
"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

PATH_TO_PLUGIN="../../build"

if [ "$SKIPGEN" == "YES" ]; then
    echo "Test Case Generation Skipped"
    sopath=$2
else
    echo "Test Case Generation Started"
    python ../nnstreamer_converter/generateGoldenTestResult.py 12
    sopath=$1
fi
convertBMP2PNG

if [[ ! -z "${UNITTEST_DIR}" ]]; then
    TESTBINDIR="${UNITTEST_DIR}"
elif [ ! -d "${PATH_TO_PLUGIN}" ] && [ ! -d "${UNITTEST_DIR}" ]; then
    TESTBINDIR="/usr/lib/nnstreamer/unittest"
else
    TESTBINDIR="../../build/tests"
fi

${TESTBINDIR}/nnstreamer_repo_dynamicity/unittest_repo --gst-plugin-path=../../build

callCompareTest testsequence_1.golden tensorsequence01_1.log 1-1 "Compare 1-1" 1 0
callCompareTest testsequence_2.golden tensorsequence01_2.log 1-2 "Compare 1-2" 1 0
callCompareTest testsequence_3.golden tensorsequence01_3.log 1-3 "Compare 1-3" 1 0
callCompareTest testsequence_4.golden tensorsequence01_4.log 1-4 "Compare 1-4" 1 0
callCompareTest testsequence_5.golden tensorsequence01_5.log 1-5 "Compare 1-5" 1 0
callCompareTest testsequence_6.golden tensorsequence01_6.log 1-6 "Compare 1-6" 1 0
callCompareTest testsequence_7.golden tensorsequence01_7.log 1-7 "Compare 1-7" 1 0
callCompareTest testsequence_8.golden tensorsequence01_8.log 1-8 "Compare 1-8" 1 0
callCompareTest testsequence_9.golden tensorsequence01_9.log 1-9 "Compare 1-9" 1 0
callCompareTest testsequence_10.golden tensorsequence01_10.log 1-10 "Compare 1-10" 1 0

report

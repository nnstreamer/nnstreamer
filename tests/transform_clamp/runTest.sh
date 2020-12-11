#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Dongju Chae <dongju.chae@samsung.com>
## @date Sep 02 2020
## @brief SSAT Test Cases for transform clamp
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
    python3 generateTest.py
    sopath=$1
fi

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=\"test_00.dat\" blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=50:100:1:1 input-type=int8 ! tensor_transform mode=clamp option=-50:50 ! filesink location=\"./result_00.dat\" sync=true" 1 0 0 $PERFORMANCE
callCompareTest result_00.dat test_00.dat.golden 1 "Golden test comparison 1" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=\"test_01.dat\" blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=50:100:1:1 input-type=uint32 ! tensor_transform mode=clamp option=+20:80 ! filesink location=\"./result_01.dat\" sync=true" 2 0 0 $PERFORMANCE
callCompareTest result_01.dat test_01.dat.golden 2 "Golden test comparison 2" 1 0


gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=\"test_02.dat\" blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=50:100:1:1 input-type=float32 ! tensor_transform mode=clamp option=-33.3:+77.7 ! filesink location=\"./result_02.dat\" sync=true" 3 0 0 $PERFORMANCE
callCompareTest result_02.dat test_02.dat.golden 3 "Golden test comparison 3" 1 0

report

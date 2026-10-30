#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author MyungJoo Ham <myungjoo.ham@samsung.com>
## @date Oct 28 2026
## @brief SSAT Test Cases for NNStreamer tensor_filter::hailort on a mock HailoRT
##
## The cases run the sub-plugin built against the mock HailoRT of this
## directory (see mock/hailort_mock.h), so they need neither HailoRT nor a
## Hailo device. The mock reads a text model; its output k is input 0 plus k.
##

if [[ "$SSATAPILOADED" != "1" ]]; then
    SILENT=0
    INDEPENDENT=1
    search="ssat-api.sh"
    source $search
    printf "${Blue}Independent Mode${NC}"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

# NNStreamer and plugins path for test
PATH_TO_PLUGIN="../../build"
PATH_TO_MOCK="${PATH_TO_PLUGIN}/tests/nnstreamer_filter_hailort"

if [[ ! -f ${PATH_TO_MOCK}/libnnstreamer_filter_hailort.so ]]; then
    echo "Cannot find the hailort sub-plugin built on the mock HailoRT"
    # Every test build has it, so its absence there must not pass as a skip.
    if [[ -d ${PATH_TO_PLUGIN} ]]; then
        testResult 1 0 "Mock sub-plugin is built" 0 1
    fi
    report
    exit
fi

# Load the mock sub-plugin, not one built with a real HailoRT.
export NNSTREAMER_FILTERS=${PATH_TO_MOCK}

PATH_TO_MODEL="../test_models/models/hailort_mock_identity.hef"
PATH_TO_TWO_OUTPUTS_MODEL="../test_models/models/hailort_mock_two_outputs.hef"
PATH_TO_TWO_GROUPS_MODEL="../test_models/models/hailort_mock_two_groups.hef"
PATH_TO_INVALID_MODEL="../test_models/labels/labels.txt"

SRC="videotestsrc num-buffers=3 pattern=13 ! video/x-raw,format=RGB,width=4,height=4,framerate=5/1 ! tensor_converter"

# The identity model returns the input as it is.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tee name=t ! queue ! tensor_filter framework=hailort model=\"${PATH_TO_MODEL}\" ! filesink location=\"testcase1.hailort.log\" sync=true t. ! queue ! filesink location=\"testcase1.direct.log\" sync=true" 1 0 0 $PERFORMANCE
callCompareTest testcase1.direct.log testcase1.hailort.log 1 "Compare 1" 0 0
# 3 frames of 3:4:4:1 uint8, so that two empty files cannot pass the comparison
[ "$(wc -c < testcase1.hailort.log)" -eq 144 ]
testResult $? 1-1 "Output size 1" 0 1

# float32 output: HailoRT dequantizes on the host.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tee name=t ! queue ! tensor_filter framework=hailort model=\"${PATH_TO_MODEL}\" custom=output_format:float32 ! filesink location=\"testcase2.hailort.log\" sync=true t. ! queue ! tensor_transform mode=typecast option=float32 ! filesink location=\"testcase2.direct.log\" sync=true" 2 0 0 $PERFORMANCE
callCompareTest testcase2.direct.log testcase2.hailort.log 2 "Compare 2" 0 0
[ "$(wc -c < testcase2.hailort.log)" -eq 576 ]
testResult $? 2-1 "Output size 2" 0 1

# Two outputs in the order of the model: the second one is the input plus 1.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tee name=t ! queue ! tensor_filter framework=hailort model=\"${PATH_TO_TWO_OUTPUTS_MODEL}\" custom=output_format:float32 ! tensor_demux name=d d.src_0 ! queue ! filesink location=\"testcase3.out0.log\" sync=true d.src_1 ! queue ! filesink location=\"testcase3.out1.log\" sync=true t. ! queue ! tensor_transform mode=typecast option=float32 ! filesink location=\"testcase3.direct0.log\" sync=true t. ! queue ! tensor_transform mode=arithmetic option=typecast:float32,add:1.0 ! filesink location=\"testcase3.direct1.log\" sync=true" 3 0 0 $PERFORMANCE
callCompareTest testcase3.direct0.log testcase3.out0.log 3-0 "Compare 3-0" 0 0
callCompareTest testcase3.direct1.log testcase3.out1.log 3-1 "Compare 3-1" 0 0
[ "$(wc -c < testcase3.out0.log)" -eq 576 ] && [ "$(wc -c < testcase3.out1.log)" -eq 576 ]
testResult $? 3-2 "Output size 3" 0 1

# Two filters sharing one virtual device group.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tee name=t ! queue ! tensor_filter framework=hailort model=\"${PATH_TO_MODEL}\" custom=group_id:ssat ! filesink location=\"testcase4.a.log\" sync=true t. ! queue ! tensor_filter framework=hailort model=\"${PATH_TO_MODEL}\" custom=group_id:ssat ! filesink location=\"testcase4.b.log\" sync=true t. ! queue ! filesink location=\"testcase4.direct.log\" sync=true" 4 0 0 $PERFORMANCE
callCompareTest testcase4.direct.log testcase4.a.log 4-0 "Compare 4-0" 0 0
callCompareTest testcase4.direct.log testcase4.b.log 4-1 "Compare 4-1" 0 0
[ "$(wc -c < testcase4.a.log)" -eq 144 ] && [ "$(wc -c < testcase4.b.log)" -eq 144 ]
testResult $? 4-2 "Output size 4" 0 1

# Negative test: the given file is not a model
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_filter framework=hailort model=\"${PATH_TO_INVALID_MODEL}\" ! fakesink" 1_n 0 1 $PERFORMANCE

# Negative test: a model of two network groups
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_filter framework=hailort model=\"${PATH_TO_TWO_GROUPS_MODEL}\" ! fakesink" 2_n 0 1 $PERFORMANCE

# Negative test: an unknown format type in the custom property
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_filter framework=hailort model=\"${PATH_TO_MODEL}\" custom=output_format:int8 ! fakesink" 3_n 0 1 $PERFORMANCE

# Negative test: input dimension does not match the model
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=3 ! video/x-raw,format=RGB,width=8,height=8,framerate=5/1 ! tensor_converter ! tensor_filter framework=hailort model=\"${PATH_TO_MODEL}\" ! fakesink" 4_n 0 1 $PERFORMANCE

# Cleanup
rm -f testcase*.log

report

#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author MyungJoo Ham <myungjoo.ham@samsung.com>
## @date Aug 26 2026
## @brief SSAT Test Cases for NNStreamer tensor_filter::tvm
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

if [[ -d ${PATH_TO_PLUGIN} ]]; then
    filter_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
else
    ini_file="/etc/nnstreamer.ini"
    if [[ ! -f ${ini_file} ]]; then
        echo "Cannot identify nnstreamer.ini"
        report
        exit
    fi

    filters_line=$(grep "^filters" ${ini_file})
    if [[ ${filters_line%%=*} != "filters" ]]; then
        echo "Cannot identify the sub-plugin path from ${ini_file}"
        report
        exit
    fi
    filter_path=${filters_line##*=}
fi

if [[ ! -f ${filter_path}/libnnstreamer_filter_tvm.so ]]; then
    echo "Cannot find the tvm sub-plugin"
    report
    exit
fi

# The test model is prebuilt per architecture, matching ARCH of unittest_filter_tvm.cc.
# CI exercises the x86_64 model only: the gbs armv7l and aarch64 jobs build with
# "unit_test 0" and the pdebuild matrix is amd64, so the other two are not covered there.
case $(uname -m) in
    x86_64) MODEL_ARCH="x86_64" ;;
    aarch64) MODEL_ARCH="aarch64" ;;
    arm*) MODEL_ARCH="arm" ;;
    *) MODEL_ARCH="unsupported" ;;
esac

PATH_TO_MODEL="../test_models/models/tvm_add_one_${MODEL_ARCH}.so_"
PATH_TO_INVALID_MODEL="../test_models/labels/labels.txt"

if [[ ! -f ${PATH_TO_MODEL} ]]; then
    echo "No tvm test model for $(uname -m)"
    report
    exit
fi

# tvm_add_one_*.so_ is a single 'ones_like + add' node: 3:480:640:1 float32 in and out.
# Every pixel value is an integer in [-255, 0] after the offset, so +1.0 and -1.0 cancel
# bit-exactly and the tvm branch must equal the direct branch byte for byte.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=480,height=640,framerate=0/1 ! videoconvert ! video/x-raw,format=RGB ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-255.0 ! tee name=t ! queue ! tensor_filter framework=tvm model=\"${PATH_TO_MODEL}\" custom=device:CPU,num_input_tensors:1 ! tensor_transform mode=arithmetic option=add:-1.0 ! filesink location=\"testcase1.tvm.log\" sync=true t. ! queue ! filesink location=\"testcase1.direct.log\" sync=true" 1 0 0 $PERFORMANCE
callCompareTest testcase1.direct.log testcase1.tvm.log 1 "Compare 1" 0 0

# Pin the output tensor shape and type: 3 * 480 * 640 * sizeof(float32)
[ -s testcase1.tvm.log ] && [ "$(wc -c < testcase1.tvm.log)" -eq 3686400 ]
testResult $? 1 "Output tensor size" 0 1

# num_input_tensors is optional; the sub-plugin infers it from the model
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=480,height=640,framerate=0/1 ! videoconvert ! video/x-raw,format=RGB ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-255.0 ! tee name=t ! queue ! tensor_filter framework=tvm model=\"${PATH_TO_MODEL}\" ! tensor_transform mode=arithmetic option=add:-1.0 ! filesink location=\"testcase2.tvm.log\" sync=true t. ! queue ! filesink location=\"testcase2.direct.log\" sync=true" 2 0 0 $PERFORMANCE
callCompareTest testcase2.direct.log testcase2.tvm.log 2 "Compare 2" 0 0

# Negative test: uint8 input while the model expects float32
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=480,height=640,framerate=0/1 ! videoconvert ! video/x-raw,format=RGB ! tensor_converter ! tensor_filter framework=tvm model=\"${PATH_TO_MODEL}\" custom=device:CPU,num_input_tensors:1 ! fakesink" 1_n 0 1 $PERFORMANCE

# Negative test: input dimension does not match the model
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=320,height=240,framerate=0/1 ! videoconvert ! video/x-raw,format=RGB ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-255.0 ! tensor_filter framework=tvm model=\"${PATH_TO_MODEL}\" custom=device:CPU,num_input_tensors:1 ! fakesink" 2_n 0 1 $PERFORMANCE

# Negative test: the given file exists but is not a tvm module
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=480,height=640,framerate=0/1 ! videoconvert ! video/x-raw,format=RGB ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-255.0 ! tensor_filter framework=tvm model=\"${PATH_TO_INVALID_MODEL}\" custom=device:CPU,num_input_tensors:1 ! fakesink" 3_n 0 1 $PERFORMANCE

# Negative test: unknown device in the custom property
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=480,height=640,framerate=0/1 ! videoconvert ! video/x-raw,format=RGB ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-255.0 ! tensor_filter framework=tvm model=\"${PATH_TO_MODEL}\" custom=device:NPU,num_input_tensors:1 ! fakesink" 4_n 0 1 $PERFORMANCE

# Negative test: num_input_tensors must be greater than 0
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=480,height=640,framerate=0/1 ! videoconvert ! video/x-raw,format=RGB ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-255.0 ! tensor_filter framework=tvm model=\"${PATH_TO_MODEL}\" custom=device:CPU,num_input_tensors:0 ! fakesink" 5_n 0 1 $PERFORMANCE

# Cleanup
rm -f testcase1.tvm.log testcase1.direct.log testcase2.tvm.log testcase2.direct.log

report

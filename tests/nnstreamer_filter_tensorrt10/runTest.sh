#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Suyeon Kim <suyeon5.kim@samsung.com>
## @date Oct 30 2023
## @brief SSAT Test Cases for NNStreamer
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

if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep tensorrt10.so)
        if [[ ! $check ]]; then
            echo "Cannot find TensorRT10 shared lib"
            report
            exit
        fi
    else
        echo "Cannot find ${ini_path}"
    fi
else
    ini_file="/etc/nnstreamer.ini"
    if [[ -f ${ini_file} ]]; then
        path=$(grep "^filters" ${ini_file})
        key=${path%=*}
        value=${path##*=}

        if [[ $key != "filters" ]]; then
            echo "String Error"
            report
            exit
        fi

        if [[ -d ${value} ]]; then
            check=$(ls ${value} | grep tensorrt10.so)
            if [[ ! $check ]]; then
                echo "Cannot find TensorRT10 shared lib"
                report
                exit
            fi
        else
            echo "Cannot find ${value}"
            report
            exit
        fi
    else
        echo "Cannot identify nnstreamer.ini"
        report
        exit
    fi
fi

PATH_TO_MODEL="../test_models/models/add_one_224.onnx"
PATH_TO_INVALID_MODEL="../test_models/labels/labels.txt"
PATH_TO_MISSING_MODEL="../test_models/models/no_such_model.onnx"

if [[ ! -f ${PATH_TO_MODEL} ]]; then
    echo "Cannot find the test model ${PATH_TO_MODEL}"
    report
    exit
fi

# add_one_224.onnx is an identity 1x1 convolution with a bias of 1.0: 3:224:224:1
# float32 in and out. Every pixel is an integer in [0, 255] before the offset, so
# +1.0 is exact even under the TF32 accumulation TensorRT enables by default, and
# the tensorrt10 branch must equal the tensor_transform branch byte for byte.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc num-buffers=1 ! \
    video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! \
    videoconvert ! video/x-raw,format=RGB ! \
    tensor_converter ! \
    tensor_transform mode=transpose option=1:2:0:3 ! \
    tensor_transform mode=typecast option=float32 ! \
    tee name=t ! queue ! \
    tensor_filter framework=tensorrt10 model=${PATH_TO_MODEL} ! \
    filesink location=testcase1.tensorrt10.log sync=true \
    t. ! queue ! \
    tensor_transform mode=arithmetic option=add:1.0 ! \
    filesink location=testcase1.direct.log sync=true" \
    1 0 0 $PERFORMANCE
callCompareTest testcase1.direct.log testcase1.tensorrt10.log 1 "Compare 1" 0 0

# Pin the output tensor shape and type: 3 * 224 * 224 * sizeof(float32)
[ -s testcase1.tensorrt10.log ] && [ "$(wc -c < testcase1.tensorrt10.log)" -eq 602112 ]
testResult $? 1 "Output tensor size" 0 1

# Regression test for issue #4850: reading tensor_filter outputs across a queue
# boundary (the sink thread maps and reads the buffer while the filter runs the
# next inference) segfaulted when the sub-plugin handed CUDA managed memory
# downstream. Multiple buffers are required to overlap the downstream CPU read
# with an active GPU inference. Comparing rather than only draining the branch
# also catches a buffer that is released or overwritten before the sink reads
# it, which corrupts the output without crashing.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc num-buffers=8 ! \
    video/x-raw,format=RGB,width=224,height=224,framerate=30/1 ! \
    videoconvert ! video/x-raw,format=RGB ! \
    tensor_converter ! \
    tensor_transform mode=transpose option=1:2:0:3 ! \
    tensor_transform mode=typecast option=float32 ! \
    tee name=t ! queue ! \
    tensor_filter framework=tensorrt10 model=${PATH_TO_MODEL} ! \
    queue ! \
    filesink location=testcase2.tensorrt10.log sync=false \
    t. ! queue ! \
    tensor_transform mode=arithmetic option=add:1.0 ! \
    filesink location=testcase2.direct.log sync=false" \
    2 0 0 $PERFORMANCE
callCompareTest testcase2.direct.log testcase2.tensorrt10.log 2 "Compare 2" 0 0

# Negative test: unsupported model file extension.
# configure_instance() throws in loadModel() before the cuda stream and
# input buffers are allocated; exercises the _model_path cleanup path.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc num-buffers=1 ! \
    video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! \
    videoconvert ! video/x-raw,format=RGB ! \
    tensor_converter ! \
    tensor_filter framework=tensorrt10 model=${PATH_TO_INVALID_MODEL} ! \
    fakesink" \
    1_n 0 1 $PERFORMANCE

# Negative test: invalid custom property.
# configure_instance() throws before loadModel(); the partially
# configured instance must be cleaned up gracefully.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc num-buffers=1 ! \
    video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! \
    videoconvert ! video/x-raw,format=RGB ! \
    tensor_converter ! \
    tensor_transform mode=transpose option=1:2:0:3 ! \
    tensor_transform mode=typecast option=float32 ! \
    tensor_filter framework=tensorrt10 model=${PATH_TO_MODEL} custom=InvalidProp:1 ! \
    fakesink" \
    2_n 0 1 $PERFORMANCE

# Negative test: the model file does not exist. The onnx parser must report the
# failure instead of the builder writing out an empty engine.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc num-buffers=1 ! \
    video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! \
    videoconvert ! video/x-raw,format=RGB ! \
    tensor_converter ! \
    tensor_transform mode=transpose option=1:2:0:3 ! \
    tensor_transform mode=typecast option=float32 ! \
    tensor_filter framework=tensorrt10 model=${PATH_TO_MISSING_MODEL} ! \
    fakesink" \
    3_n 0 1 $PERFORMANCE

# Negative test: input dimension does not match the model.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc num-buffers=1 ! \
    video/x-raw,format=RGB,width=320,height=240,framerate=0/1 ! \
    videoconvert ! video/x-raw,format=RGB ! \
    tensor_converter ! \
    tensor_transform mode=transpose option=1:2:0:3 ! \
    tensor_transform mode=typecast option=float32 ! \
    tensor_filter framework=tensorrt10 model=${PATH_TO_MODEL} ! \
    fakesink" \
    4_n 0 1 $PERFORMANCE

# Negative test: uint8 input while the model expects float32.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc num-buffers=1 ! \
    video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! \
    videoconvert ! video/x-raw,format=RGB ! \
    tensor_converter ! \
    tensor_transform mode=transpose option=1:2:0:3 ! \
    tensor_filter framework=tensorrt10 model=${PATH_TO_MODEL} ! \
    fakesink" \
    5_n 0 1 $PERFORMANCE

# Cleanup
rm *.log

report

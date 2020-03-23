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

# NNStreamer and plugins path for test
PATH_TO_PLUGIN="../../build"

if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep tensorflow-lite.so)
        if [[ ! $check ]]; then
            echo "Cannot find tensorflow-lite shared lib"
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
            check=$(ls ${value} | grep tensorflow-lite.so)
            if [[ ! $check ]]; then
                echo "Cannot find tensorflow-lite shared lib"
                report
                exit
            fi
        else
            echo "Cannot file ${value}"
            report
            exit
        fi
    else
        echo "Cannot identify nnstreamer.ini"
        report
        exit
    fi
fi

# Decoding 'orange' tests
# Since data type of output tensor is uint8, int8 requires another 'quantization' (such as /2)
PATH_TO_MODEL="../test_models/models/mobilenet_v1_1.0_224_quant.tflite"
PATH_TO_LABEL="../test_models/labels/labels.txt"
PATH_TO_IMAGE="../test_models/data/orange.png"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=\"${PATH_TO_IMAGE}\" ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw, format=RGB, framerate=0/1 ! tensor_converter ! tensor_filter framework=\"tensorflow-lite\" model=\"${PATH_TO_MODEL}\" ! \
tee name=t ! queue ! tensor_transform mode=typecast option=uint8 ! tensor_decoder mode=image_labeling option1=\"${PATH_TO_LABEL}\" ! filesink location=\"tensordecoder.orange.uint8.log\" \
t. ! queue ! tensor_transform mode=typecast option=uint16 ! tensor_decoder mode=image_labeling option1=\"${PATH_TO_LABEL}\" ! filesink location=\"tensordecoder.orange.uint16.log\" \
t. ! queue ! tensor_transform mode=typecast option=uint32 ! tensor_decoder mode=image_labeling option1=\"${PATH_TO_LABEL}\" ! filesink location=\"tensordecoder.orange.uint32.log\" \
t. ! queue ! tensor_transform mode=typecast option=uint64 ! tensor_decoder mode=image_labeling option1=\"${PATH_TO_LABEL}\" ! filesink location=\"tensordecoder.orange.uint64.log\" \
t. ! queue ! tensor_transform mode=arithmetic option=div:2 ! tensor_transform mode=typecast option=int8 ! tensor_decoder mode=image_labeling option1=\"${PATH_TO_LABEL}\" ! filesink location=\"tensordecoder.orange.int8.log\" \
t. ! queue ! tensor_transform mode=typecast option=int16 ! tensor_decoder mode=image_labeling option1=\"${PATH_TO_LABEL}\" ! filesink location=\"tensordecoder.orange.int16.log\" \
t. ! queue ! tensor_transform mode=typecast option=int32 ! tensor_decoder mode=image_labeling option1=\"${PATH_TO_LABEL}\" ! filesink location=\"tensordecoder.orange.int32.log\" \
t. ! queue ! tensor_transform mode=typecast option=int64 ! tensor_decoder mode=image_labeling option1=\"${PATH_TO_LABEL}\" ! filesink location=\"tensordecoder.orange.int64.log\" \
t. ! queue ! tensor_transform mode=typecast option=float32 ! tensor_decoder mode=image_labeling option1=\"${PATH_TO_LABEL}\" ! filesink location=\"tensordecoder.orange.float.log\" \
t. ! queue ! tensor_transform mode=typecast option=float64 ! tensor_decoder mode=image_labeling option1=\"${PATH_TO_LABEL}\" ! filesink location=\"tensordecoder.orange.double.log\"" D1 0 0 $PERFORMANCE
let i=1
for result in tensordecoder.orange.*.log; do
    label=$(cat "${result}")
    if [ "$label" == "orange" ]; then
        testResult 1 D1-${i} "Decoding Orange"
    else
        testResult 0 D1-${i} "Decoding Orange"
    fi
    let i++
done

report

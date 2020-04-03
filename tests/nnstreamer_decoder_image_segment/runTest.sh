#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Yongjoo Ahn <yongjoo1.ahn@samsung.com>
## @date Feb 26 2020
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

PATH_TO_MODEL="../test_models/models/deeplabv3_257_mv_gpu.tflite"

# THIS SHOULD EMIT ERROR
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num_buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=480 ! tee name=t t. ! queue ! mix. t. ! queue ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! tensor_filter framework=tensorflow-lite model=${PATH_TO_MODEL} ! tensor_decoder mode=image_segment option1=tflite-deeplab ! mix. videomixer name=mix sink_0::alpha=0.7 sink_1::alpha=0.6 ! videoconvert ! fakesink" 0_n 0 1

# THIS WON'T FAIL, BUT NOT MUCH MEANINGFUL.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num_buffers=4 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=257,height=257 ! tee name=t t. ! queue ! mix. t. ! queue ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! tensor_filter framework=tensorflow-lite model=${PATH_TO_MODEL} ! tensor_decoder mode=image_segment option1=tflite-deeplab ! mix. videomixer name=mix sink_0::alpha=0.7 sink_1::alpha=0.6 ! videoconvert ! fakesink" 0_p 0 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=257,height=257 ! tee name=t t. ! queue ! mix. t. ! queue ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! tensor_filter framework=tensorflow-lite model=${PATH_TO_MODEL} ! tensor_decoder mode=image_segment option1=tflite-deeplab ! mix. videomixer name=mix sink_0::alpha=0.7 sink_1::alpha=0.6 ! filesink location=test_output.0" 1 0 0 $PERFORMANCE

callCompareTest test_golden.0 test_output.0 1 "test with videotestsrc" 0

rm test_output.*

report

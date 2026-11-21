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
    printf "${Blue}Independent Mode${NC}"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

# NNStreamer and plugins path for test
PATH_TO_PLUGIN="../../build"

# The snpe modes decode a tensor file directly, so these cases need no
# sub-plugin and run before the tensorflow2-lite check below. The frame is
# 257 x 255, a pixel count that is not a multiple of four, and the capsfilter
# pins the frame size so that a width and height swap fails to link.
python3 generateTest.py
testResult $? 4-0 "generate the snpe tensor files" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=snpe_depth.raw blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=1:257:255 input-type=float32 ! tensor_transform mode=typecast option=float32 ! tensor_decoder mode=image_segment option1=snpe-depth ! video/x-raw,format=RGBA,width=257,height=255 ! filesink location=test_output.4" 4 0 0 $PERFORMANCE
callCompareTest snpe_depth.golden test_output.4 4-1 "snpe-depth golden frame" 0

# The same pipeline with an int32 tensor, the only difference from case 4
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=snpe_depth.raw blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=1:257:255 input-type=float32 ! tensor_transform mode=typecast option=int32 ! tensor_decoder mode=image_segment option1=snpe-depth ! video/x-raw,format=RGBA,width=257,height=255 ! fakesink" 4_n 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=snpe_deeplab.raw blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=257:255:1 input-type=float32 ! tensor_transform mode=typecast option=float32 ! tensor_decoder mode=image_segment option1=snpe-deeplab ! video/x-raw,format=RGBA,width=257,height=255 ! filesink location=test_output.5" 5 0 0 $PERFORMANCE
# 20 is the number of labels option2 defaults to
python3 checkResult.py snpe_deeplab.raw test_output.5 20
testResult $? 5-1 "snpe-deeplab colors follow the labels" 0 1

# The same pipeline with an int32 tensor, the only difference from case 5
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=snpe_deeplab.raw blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=257:255:1 input-type=float32 ! tensor_transform mode=typecast option=int32 ! tensor_decoder mode=image_segment option1=snpe-deeplab ! video/x-raw,format=RGBA,width=257,height=255 ! fakesink" 5_n 0 1

rm -f snpe_depth.raw snpe_depth.golden snpe_deeplab.raw test_output.4 test_output.5

if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep tensorflow2-lite.so)
        if [[ ! $check ]]; then
            echo "Cannot find tensorflow2-lite shared lib"
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
            check=$(ls ${value} | grep tensorflow2-lite.so)
            if [[ ! $check ]]; then
                echo "Cannot find tensorflow2-lite shared lib"
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
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num_buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=640,height=480 ! tee name=t t. ! queue ! mix. t. ! queue ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! tensor_filter framework=tensorflow2-lite model=${PATH_TO_MODEL} ! tensor_decoder mode=image_segment option1=tflite-deeplab ! mix. videomixer name=mix sink_0::alpha=0.7 sink_1::alpha=0.6 ! videoconvert ! fakesink" 0_n 0 1

# THIS WON'T FAIL, BUT NOT MUCH MEANINGFUL.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num_buffers=4 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=257,height=257 ! tee name=t t. ! queue ! mix. t. ! queue ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! tensor_filter framework=tensorflow2-lite model=${PATH_TO_MODEL} ! tensor_decoder mode=image_segment option1=tflite-deeplab ! mix. videomixer name=mix sink_0::alpha=0.7 sink_1::alpha=0.6 ! videoconvert ! fakesink" 0_p 0 0

# The golden is the decoder's output, taken off a tee before videomixer, so
# that the compare checks the image_segment decoder and not videomixer's alpha
# blending, whose rounding differs between GStreamer releases (1.24 vs 1.28).
# It is still an exact compare of an argmax over tensorflow-lite output, so a
# rebuild of the tensorflow2-lite package with another compiler can move
# boundary pixels. That is a golden to refresh, not a decoder to debug.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=257,height=257 ! tee name=t t. ! queue ! mix. t. ! queue ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! tensor_filter framework=tensorflow2-lite model=${PATH_TO_MODEL} ! tensor_decoder mode=image_segment option1=tflite-deeplab ! tee name=d d. ! queue ! filesink location=test_output.0 d. ! queue ! mix. videomixer name=mix sink_0::alpha=0.7 sink_1::alpha=0.6 ! fakesink" 1 0 0 $PERFORMANCE

callCompareTest test_golden.0 test_output.0 1 "test with videotestsrc" 0

# THIS WON'T FAIL, BUT NOT MUCH MEANINGFUL (TARGET MODEL IS DIFFERENT ONE)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
videotestsrc num_buffers=4 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=257,height=257 ! tee name=t \
    t. ! queue ! mix. \
    t. ! queue ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! \
        tensor_filter framework=tensorflow2-lite model=${PATH_TO_MODEL} latency=1 throughput=1 ! \
        tensor_decoder mode=image_segment option1=snpe-deeplab option2=180 ! mix. \
videomixer name=mix sink_0::alpha=0.7 sink_1::alpha=0.6 ! videoconvert ! fakesink" \
2 0 0 $PERFORMANCE

# THIS SHOULD EMIT ERROR
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
videotestsrc num_buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=257,height=257 ! tee name=t \
    t. ! queue ! mix. \
    t. ! queue ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,div:255.0 ! \
        tensor_filter framework=tensorflow2-lite model=${PATH_TO_MODEL} ! \
        tensor_decoder mode=image_segment option1=snpe-depth ! mix. \
videomixer name=mix sink_0::alpha=0.7 sink_1::alpha=0.6 ! videoconvert ! fakesink" \
3_n 0 1

rm test_output.*

report

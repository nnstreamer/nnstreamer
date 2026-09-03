#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author MyungJoo Ham <myungjoo.ham@samsung.com>
## @date Aug 20 2026
## @brief SSAT Test Cases for the LiteRT (CompiledModel API) tensor filter sub-plugin
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
        check=$(ls ${ini_path} | grep -E 'litert\.(so|dylib)')
        if [[ ! $check ]]; then
            echo "Cannot find litert shared lib"
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
            check=$(ls ${value} | grep -E 'litert\.(so|dylib)')
            if [[ ! $check ]]; then
                echo "Cannot find litert lib"
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

PATH_TO_MODEL="../test_models/models/mobilenet_v1_1.0_224_quant.tflite"
PATH_TO_LABEL="../test_models/labels/labels.txt"
PATH_TO_IMAGE="../test_models/data/orange.png"
PATH_TO_CLASS="class.out.log"
PATH_TO_CLASS1="class1.out.log"
PATH_TO_CLASS2="class2.out.log"
PATH_TO_DYNAMIC_MODEL="../test_models/models/dynamic_batch_add_one.tflite"

# Test 1: Positive. Golden classification result, same model and golden label
# as the tensorflow2-lite SSAT tests; cross-runtime divergence fails here.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_filter framework=litert model=${PATH_TO_MODEL} ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=${PATH_TO_CLASS}" 1 0 0 $PERFORMANCE
class=$(cat ${PATH_TO_CLASS})
[ "$class" = "orange" ]
testResult $? 1 "Golden test comparison" 0 1

# Test 2: Positive. Explicit cpu accelerator custom property.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_filter framework=litert model=${PATH_TO_MODEL} custom=Accelerators:cpu ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=${PATH_TO_CLASS}" 2 0 0 $PERFORMANCE
class=$(cat ${PATH_TO_CLASS})
[ "$class" = "orange" ]
testResult $? 2 "Golden test comparison with Accelerators:cpu" 0 1

# Test 3: Negative. Mismatched input dimensions must fail.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=42,height=42,framerate=0/1 ! tensor_converter ! tensor_filter framework=litert model=${PATH_TO_MODEL} ! fakesink" 3_n 0 1 $PERFORMANCE

# Test 4: Negative. Invalid model path must fail.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224 ! tensor_converter ! tensor_filter framework=litert model=invalid_model_path.tflite ! fakesink" 4_n 0 1 $PERFORMANCE

# Test 5: Negative. Unknown accelerator custom property must fail.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_filter framework=litert model=${PATH_TO_MODEL} custom=Accelerators:tpu ! fakesink" 5_n 0 1 $PERFORMANCE

# Test 6: Positive. Two litert tensor_filter instances share the process-wide
# LiteRtEnvironment; both branches, held open at the same time in the same
# pipeline, must still produce the golden classification result. Both
# branches write their own log, removed first, so no earlier case output
# can stand in for a branch that produced nothing.
rm -f ${PATH_TO_CLASS1} ${PATH_TO_CLASS2}
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=litert model=${PATH_TO_MODEL} ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=${PATH_TO_CLASS1} t. ! queue ! tensor_filter framework=litert model=${PATH_TO_MODEL} ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=${PATH_TO_CLASS2}" 6 0 0 $PERFORMANCE
class1=$(cat ${PATH_TO_CLASS1})
class2=$(cat ${PATH_TO_CLASS2})
[ "$class1" = "orange" ] && [ "$class2" = "orange" ]
testResult $? 6 "Golden test comparison with two concurrent litert instances" 0 1

# Test 7: Positive. invoke-dynamic through a real pipeline. The gtest cases
# drive the subplugin directly, which leaves the whole framework side of a
# dynamic invoke untested: allocate_in_invoke being forced on, the flexible
# output and its meta header, g_free as the destroy notify, and input_meta
# being refreshed per buffer. The caps spell the trailing dimensions out as
# 4:1:1:1, which is what an element upstream actually sends and what the
# model reports as 4:1, so this also runs the padded form end to end.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} audiotestsrc num-buffers=3 ! audio/x-raw,format=F32LE,rate=16000,channels=4 ! tensor_converter frames-per-tensor=1 ! other/tensors,format=static,num_tensors=1,dimensions=4:1:1:1,types=float32,framerate=16000/1 ! tensor_filter framework=litert model=${PATH_TO_DYNAMIC_MODEL} invoke-dynamic=true ! other/tensors,format=flexible ! fakesink" 7 0 0 $PERFORMANCE

# Test 8: Negative. invoke-dynamic demands a flexible output; a static one
# must be refused rather than silently reinterpreted.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} audiotestsrc num-buffers=3 ! audio/x-raw,format=F32LE,rate=16000,channels=4 ! tensor_converter frames-per-tensor=1 ! tensor_filter framework=litert model=${PATH_TO_DYNAMIC_MODEL} invoke-dynamic=true ! other/tensors,format=static ! fakesink" 8_n 0 1 $PERFORMANCE

report

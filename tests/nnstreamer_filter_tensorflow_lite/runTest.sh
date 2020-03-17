#!/usr/bin/env bash
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

PATH_TO_MODEL="../test_models/models/mobilenet_v1_1.0_224_quant.tflite"
PATH_TO_LABEL="../test_models/labels/labels.txt"
PATH_TO_IMAGE="../test_models/data/orange.png"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow-lite model=${PATH_TO_MODEL} ! filesink location=tensorfilter.out.log" 1 0 0 $PERFORMANCE
python checkLabel.py tensorfilter.out.log ${PATH_TO_LABEL} orange
testResult $? 1 "Golden test comparison" 0 1

# Fail test for invalid input properties
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow-lite model=${PATH_TO_MODEL} input=7:1 inputtype=float32 ! filesink location=tensorfilter.out.log" 2F_n 0 1 $PERFORMANCE

# Fail test for invalid output properties
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow-lite model=${PATH_TO_MODEL} output=1:7 outputtype=int8 ! filesink location=tensorfilter.out.log" 3F_n 0 1 $PERFORMANCE

PATH_TO_MULTI_TENSOR_OUTPUT_MODEL="../test_models/models/multi_person_mobilenet_v1_075_float.tflite"

# Simple tests for multi-tensor output model
# This should emit error because of invalid width and height size
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num_buffers=4 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=353,height=257 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter framework=tensorflow-lite model=${PATH_TO_MULTI_TENSOR_OUTPUT_MODEL} ! fakesink" 4_n 0 1 $PERFORMANCE

# This won't fail, but not much meaningful
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num_buffers=4 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=257,height=353 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter framework=tensorflow-lite model=${PATH_TO_MULTI_TENSOR_OUTPUT_MODEL} ! fakesink" 5 0 0 $PERFORMANCE

# Test the backend setting done with tensorflow-lite
# This also performs tests for generic backend configuration parsing
function run_pipeline() {
    gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow-lite model=${PATH_TO_MODEL} accelerator=$1 ! filesink location=tensorfilter.out.log 2>info
}

# Property reading test for nnapi
run_pipeline true:cpu,npu,gpu
cat info | grep "nnapi = 1, accl = cpu"
testResult $? 2-1 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline true:!cpu
cat info | grep "nnapi = 1, accl = auto"
testResult $? 2-2 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline true:!npu,gpu
cat info | grep "nnapi = 1, accl = gpu"
testResult $? 2-3 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline true:!npu,gpu,abcd
cat info | grep "nnapi = 1, accl = gpu"
testResult $? 2-4 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline true:!npu,!abcd,gpu
cat info | grep "nnapi = 1, accl = gpu"
testResult $? 2-5 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline true:auto
cat info | grep "nnapi = 1, accl = auto"
testResult $? 2-6 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline true:default,cpu
cat info | grep "nnapi = 1, accl = default"
testResult $? 2-7 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline true:!cpu,default
cat info | grep "nnapi = 1, accl = default"
testResult $? 2-8 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline true:!default
cat info | grep "nnapi = 1, accl = auto"
testResult $? 2-9 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline true:npu.srcn
cat info | grep "nnapi = 1, accl = npu"
testResult $? 2-10 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline false:abcd
cat info | grep "nnapi = 0, accl = none"
testResult $? 2-11 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline false
cat info | grep "nnapi = 0, accl = none"
testResult $? 2-12 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline true:
cat info | grep "nnapi = 1, accl = auto"
testResult $? 2-13 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline true
cat info | grep "nnapi = 1, accl = auto"
testResult $? 2-14 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline auto
cat info | grep "nnapi = 0, accl = none"
testResult $? 2-15 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline true:!npu,abcd,gpu
cat info | grep "nnapi = 1, accl = gpu"
testResult $? 2-16 "NNAPI activation test" 0 1

# Property reading test for nnapi
run_pipeline true:cpu.neon,cpu
cat info | grep "nnapi = 1, accl = cpu.neon"
testResult $? 2-17 "NNAPI activation test" 0 1

# Cleanup
rm info

report

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
    printf "${Blue}Independent Mode${NC}"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

# NNStreamer and plugins path for test
PATH_TO_PLUGIN="../../build"

if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep tensorflow1-lite.so)
        if [[ ! $check ]]; then
            echo "Cannot find tensorflow1-lite shared lib"
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
            check=$(ls ${value} | grep tensorflow1-lite.so)
            if [[ ! $check ]]; then
                echo "Cannot find tensorflow1-lite shared lib"
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

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow1-lite model=${PATH_TO_MODEL} ! filesink location=tensorfilter.out.log" 1 0 0 $PERFORMANCE
python3 checkLabel.py tensorfilter.out.log ${PATH_TO_LABEL} orange
testResult $? 1 "Golden test comparison" 0 1

# Fail test for invalid input properties
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow1-lite model=${PATH_TO_MODEL} input=7:1 inputtype=float32 ! filesink location=tensorfilter.out.log" 2F_n 0 1 $PERFORMANCE

# Fail test for invalid output properties
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow1-lite model=${PATH_TO_MODEL} output=1:7 outputtype=int8 ! filesink location=tensorfilter.out.log" 3F_n 0 1 $PERFORMANCE

PATH_TO_MULTI_TENSOR_OUTPUT_MODEL="../test_models/models/multi_person_mobilenet_v1_075_float.tflite"

# Simple tests for multi-tensor output model
# This should emit error because of invalid width and height size
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num_buffers=4 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=353,height=257 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter framework=tensorflow1-lite model=${PATH_TO_MULTI_TENSOR_OUTPUT_MODEL} ! fakesink" 4_n 0 1 $PERFORMANCE

# This won't fail, but not much meaningful
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num_buffers=4 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=257,height=353 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter framework=tensorflow1-lite model=${PATH_TO_MULTI_TENSOR_OUTPUT_MODEL} ! fakesink" 5 0 0 $PERFORMANCE

# Input and output combination test
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc pattern=13 num-buffers=1 ! videoconvert !  video/x-raw,width=640,height=480,framerate=30/1 ! tensor_converter ! tee name=t t. ! queue ! mux.sink_0 t. ! queue ! filesink location=combi.dummy.golden buffer-mode=unbuffered sync=false async=false filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! mux.sink_1 tensor_mux name=mux ! tensor_filter framework=tensorflow1-lite model=${PATH_TO_MODEL} input-combination=1 output-combination=i0,o0 ! tensor_demux name=demux demux.src_0 ! filesink location=tensorfilter.combi.in.log buffer-mode=unbuffered sync=false async=false demux.src_1 ! filesink location=tensorfilter.combi.out.log buffer-mode=unbuffered sync=false async=false" 7 0 0 $PERFORMANCE
callCompareTest combi.dummy.golden tensorfilter.combi.in.log 7_0 "Output Combination Golden Test 7-0" 1 0
python3 checkLabel.py tensorfilter.combi.out.log ${PATH_TO_LABEL} orange
testResult $? 1 "Golden test comparison" 0 1

# Test the backend setting done with tensorflow1-lite
# This also performs tests for generic backend configuration parsing
function run_pipeline() {
    gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow1-lite model=${PATH_TO_MODEL} accelerator=$1 ! filesink location=tensorfilter.out.log 2>info
}

arch=$(uname -m)
if [ "$arch" = "aarch64" ] || [ "$arch" = "armv7l" ]; then
  auto_accl="cpu.neon"
elif [ "$arch" = "x86_64" ]; then
  auto_accl="cpu.simd"
else
  auto_accl="cpu"
fi

# Property reading test for accelerator
run_pipeline true:cpu,npu,gpu
cat info | grep "accl = cpu$"
testResult $? 2-1 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline true:!cpu
cat info | grep "accl = ${auto_accl}$"
testResult $? 2-2 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline true:!npu,gpu
cat info | grep "accl = gpu$"
testResult $? 2-3 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline true:!npu,gpu,abcd
cat info | grep "accl = gpu$"
testResult $? 2-4 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline true:!npu,!abcd,gpu
cat info | grep "accl = gpu$"
testResult $? 2-5 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline true:auto
cat info | grep "accl = ${auto_accl}$"
testResult $? 2-6 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline true:default,gpu
cat info | grep "accl = cpu$"
testResult $? 2-7 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline true:!cpu,default
cat info | grep "accl = cpu$"
testResult $? 2-8 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline true:!default
cat info | grep "accl = ${auto_accl}$"
testResult $? 2-9 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline true:npu.srcn
cat info | grep "accl = npu$"
testResult $? 2-10 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline false:abcd
cat info | grep " accl = none$"
testResult $? 2-11 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline false
cat info | grep "accl = none$"
testResult $? 2-12 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline true:
cat info | grep "accl = ${auto_accl}$"
testResult $? 2-13 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline true
cat info | grep "accl = ${auto_accl}$"
testResult $? 2-14 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline auto
cat info | grep "accl = none$"
testResult $? 2-15 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline true:!npu,abcd,gpu
cat info | grep "accl = gpu$"
testResult $? 2-16 "accelerator set test" 0 1

# Property reading test for accelerator
run_pipeline true:${auto_accl},cpu
cat info | grep "accl = ${auto_accl}$"
testResult $? 2-17 "accelerator set test" 0 1

# Property reading test for accelerator before setting the framework (analogous test is 2-3)
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter accelerator=true:!npu,gpu framework=tensorflow1-lite model=${PATH_TO_MODEL} ! filesink location=tensorfilter.out.log 2>info
cat info | grep "accl = gpu$"
testResult $? 2-18 "accelerator set test" 0 1

# Dimension declaration test cases
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow1-lite model=${PATH_TO_MODEL} !  \"other/tensors,num_tensors=1,dimensions=1001:1:1\" ! filesink location=tensorfilter.out.log" 3 0 0 $PERFORMANCE
python3 checkLabel.py tensorfilter.out.log ${PATH_TO_LABEL} orange
testResult $? 3 "Golden test comparison" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow1-lite model=${PATH_TO_MODEL} !  \"other/tensors,num_tensors=1,dimensions=(string)1001:1\" ! filesink location=tensorfilter.out.log" 4 0 0 $PERFORMANCE
python3 checkLabel.py tensorfilter.out.log ${PATH_TO_LABEL} orange
testResult $? 4 "Golden test comparison" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow1-lite model=${PATH_TO_MODEL} ! \"other/tensors,num_tensors=1,dimensions=(string)1001\" ! filesink location=tensorfilter.out.log" 5 0 0 $PERFORMANCE
python3 checkLabel.py tensorfilter.out.log ${PATH_TO_LABEL} orange
testResult $? 5 "Golden test comparison" 0 1

# Cleanup
rm info *.log *.golden

report

#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Parichay Kapoor <pk.kapoor@samsung.com>
## @date May 7th 2019
## @brief SSAT Test Cases for NNStreamer pytorch plugin
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

# Test constant passthrough custom filter (1, 2)
PATH_TO_PLUGIN="../../build"
PATH_TO_MODEL="../test_models/models/pytorch_lenet5.pt"
PATH_TO_IMAGE="../test_models/data/9.png"

if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep pytorch.so)
        if [[ ! $check ]]; then
            echo "Cannot find pytorch shared lib"
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
            check=$(ls ${value} | grep pytorch.so)
            if [[ ! $check ]]; then
                echo "Cannot find pytorch shared lib"
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

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} input=1:28:28:1 inputtype=uint8 output=10:1:1:1 outputtype=uint8 ! filesink location=tensorfilter.out.log" 1 0 0 $PERFORMANCE
python checkLabel.py tensorfilter.out.log ${PATH_TO_IMAGE}
testResult $? 1 "Golden test comparison" 0 1

# Fail test for invalid input properties
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} input=7:1 inputtype=float32 ! filesink location=tensorfilter.out.log" 2F_n 0 1 $PERFORMANCE

# Fail test for invalid output properties
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} output=1:7 outputtype=int8 ! filesink location=tensorfilter.out.log" 3F_n 0 1 $PERFORMANCE

# Test the setting of accelerators
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} output=1:7 outputtype=int8 accelerator=true:gpu ! filesink location=tensorfilter.out.log 2>info
cat info | grep "gpu = 1, accl = gpu"
testResult $? 2-1 "GPU activation test" 0 1

gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} output=1:7 outputtype=int8 accelerator=true:cpu ! filesink location=tensorfilter.out.log 2>info
cat info | grep "gpu = 0, accl = cpu"
testResult $? 2-2 "GPU activation test" 0 1

gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} output=1:7 outputtype=int8 ! filesink location=tensorfilter.out.log 2>info
cat info | grep "gpu = 0, accl = default"
testResult $? 2-3 "GPU activation test" 0 1

# Cleanup
rm info

report

#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## Copyright 2022 NXP
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

# Check if architecture is supported
SUPPORTED_ARCHS="aarch64 armv7l x86_64"
ARCH=`uname -m`
echo ${SUPPORTED_ARCHS} | grep -w -q ${ARCH}
if [ $? -ne 0 ]; then
  echo "Architecture is not supported - ${ARCH}"
  report
  exit
fi

# NNStreamer and plugins path for test
PATH_TO_PLUGIN="../../build"

if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep tvm.so)
        if [[ ! $check ]]; then
            echo "Cannot find tvm shared lib"
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
            check=$(ls ${value} | grep tvm.so)
            if [[ ! $check ]]; then
                echo "Cannot find tvm shared lib"
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

PATH_TO_MODEL="../test_models/models/mobilenet_v1_0.75_224_quant_${ARCH}.so"
PATH_TO_LABEL="../test_models/labels/labels.txt"
PATH_TO_IMAGE="../test_models/data/orange.png"
PATH_TO_CLASS="class.out.log"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tvm model=${PATH_TO_MODEL} custom=device:CPU,num_input_tensors:1 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=${PATH_TO_CLASS}" 1 0 0 $PERFORMANCE

class=`cat ${PATH_TO_CLASS}`
[ "$class" = "orange" ]
testResult $? 1 "Golden test comparison" 0 1

# Negative test: wrong input type float32 instead of uint8
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,div:127.5,add:-1.0 ! tensor_filter framework=tvm model=${PATH_TO_MODEL} custom=device:CPU,num_input_tensors:1 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=${PATH_TO_CLASS} " 1_n 0 1 $PERFORMANCE

# Cleanup
rm *.log

report

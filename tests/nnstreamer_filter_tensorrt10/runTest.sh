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
        check=$(ls ${ini_path} | grep tensorrt.so)
        if [[ ! $check ]]; then
            echo "Cannot find TensorRT shared lib"
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
            check=$(ls ${value} | grep tensorrt.so)
            if [[ ! $check ]]; then
                echo "Cannot find TensorRT lib"
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

PATH_TO_MODEL="../test_models/models/yolov5nu_224.onnx"
PATH_TO_LABEL="../test_models/labels/coco.txt"
PATH_TO_IMAGE="../test_models/data/orange.png"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    filesrc location=${PATH_TO_IMAGE} ! \
    pngdec ! \
    videoscale ! \
    imagefreeze ! \
    videoconvert ! \
    video/x-raw,width=224,height=224,format=RGB,framerate=0/1 ! \
    tensor_converter ! \
    tensor_transform mode=transpose option=1:2:0:3 ! \
    tensor_transform mode=arithmetic option=typecast:float32,div:255 ! \
    tensor_filter framework=tensorrt10 model=${PATH_TO_MODEL} ! \
    tensor_transform mode=transpose option=1:0:2:3 ! \
    tensor_decoder mode=bounding_boxes option1=yolov8 option2=${PATH_TO_LABEL} option3=1 option4=224:224 option5=224:224 ! \
    multifilesink location=yolov5nu_result_%1d.log" \
    1 0 0 $PERFORMANCE

# Cleanup
rm yolov5nu_result_*.log*

report

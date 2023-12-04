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
        check=$(ls ${ini_path} | grep onnxruntime.so)
        if [[ ! $check ]]; then
            echo "Cannot find onnxruntime shared lib"
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
            check=$(ls ${value} | grep onnxruntime.so)
            if [[ ! $check ]]; then
                echo "Cannot find onnxruntime lib"
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

PATH_TO_MODEL="../test_models/models/mobilenet_v2_quant.onnx"
PATH_TO_LABEL="../test_models/labels/labels.txt"
PATH_TO_IMAGE="../test_models/data/orange.png"
PATH_TO_CLASS="class.out.log"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_transform mode=arithmetic option=typecast:float32,div:127.5,add:-1.0 ! tensor_filter framework=onnxruntime model=${PATH_TO_MODEL} ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=${PATH_TO_CLASS} " 1 0 0 $PERFORMANCE
class=`cat ${PATH_TO_CLASS}`
[ "$class" = "orange" ]
testResult $? 1 "Golden test comparison" 0 1

# Negative tests: wrong input dimensions 1:3:224:224 instead of 1:224:224:3
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose ! tensor_transform mode=arithmetic option=typecast:float32,div:127.5,add:-1.0 ! tensor_filter framework=onnxruntime model=${PATH_TO_MODEL} ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=${PATH_TO_CLASS} " 1_n 0 1 $PERFORMANCE

# Negative tests: wrong model data type
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_transform mode=arithmetic option=typecast:float16,div:127.5,add:-1.0 ! tensor_filter framework=onnxruntime model=${PATH_TO_MODEL} ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=${PATH_TO_CLASS} " 2_n 0 1 $PERFORMANCE

# Negative tests: wrong model input type
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_transform mode=arithmetic option=typecast:float32,div:127.5,add:-1.0 ! tensor_filter framework=onnxruntime model=${PATH_TO_MODEL} input=7:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=${PATH_TO_CLASS} " 3_n 0 1 $PERFORMANCE

# Negative tests: wrong model output type
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_transform mode=arithmetic option=typecast:float32,div:127.5,add:-1.0 ! tensor_filter framework=onnxruntime model=${PATH_TO_MODEL} output=1:7 outputtype=int8 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=${PATH_TO_CLASS} " 4_n 0 1 $PERFORMANCE

# Negative tests: invalid model path
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_transform mode=arithmetic option=typecast:float32,div:127.5,add:-1.0 ! tensor_filter framework=onnxruntime model=../${PATH_TO_MODEL} ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=${PATH_TO_CLASS} " 5_n 0 1 $PERFORMANCE

PATH_TO_MODEL="../test_models/models/mobilenet_v1_1.0_224_quant.invalid"
# Negative tests: invalid model file
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_transform mode=arithmetic option=typecast:float32,div:127.5,add:-1.0 ! tensor_filter framework=onnxruntime model=${PATH_TO_MODEL} ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=${PATH_TO_CLASS} " 6_n 0 1 $PERFORMANCE


# Cleanup
rm *.log

report

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
        check=$(ls ${ini_path} | grep tensorrt10.so)
        if [[ ! $check ]]; then
            echo "Cannot find TensorRT10 shared lib"
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
            check=$(ls ${value} | grep tensorrt10.so)
            if [[ ! $check ]]; then
                echo "Cannot find TensorRT10 shared lib"
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

## @brief Regression test for issue #4850: reading tensor_filter outputs across a
##        queue boundary (fakesink dump=true maps and reads the buffer contents in
##        the sink thread while the filter runs the next inference) segfaulted when
##        the subplugin handed CUDA managed memory downstream. Multiple buffers are
##        required to overlap downstream CPU reads with an active GPU inference.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    filesrc location=${PATH_TO_IMAGE} ! \
    pngdec ! \
    videoscale ! \
    imagefreeze num-buffers=8 ! \
    videoconvert ! \
    video/x-raw,width=224,height=224,format=RGB,framerate=30/1 ! \
    tensor_converter ! \
    tensor_transform mode=transpose option=1:2:0:3 ! \
    tensor_transform mode=arithmetic option=typecast:float32,div:255 ! \
    tensor_filter framework=tensorrt10 model=${PATH_TO_MODEL} ! \
    queue ! \
    fakesink dump=true sync=false" \
    2 0 0 $PERFORMANCE

# Negative test: unsupported model file extension.
# configure_instance() throws in loadModel() before the cuda stream and
# input buffers are allocated; exercises the _model_path cleanup path.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc num_buffers=1 ! \
    videoconvert ! \
    video/x-raw,width=224,height=224,format=RGB,framerate=0/1 ! \
    tensor_converter ! \
    tensor_filter framework=tensorrt10 model=${PATH_TO_LABEL} ! \
    fakesink" \
    3_n 0 1 $PERFORMANCE

# Negative test: invalid custom property.
# configure_instance() throws before loadModel(); the partially
# configured instance must be cleaned up gracefully.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    videotestsrc num_buffers=1 ! \
    videoconvert ! \
    video/x-raw,width=224,height=224,format=RGB,framerate=0/1 ! \
    tensor_converter ! \
    tensor_filter framework=tensorrt10 model=${PATH_TO_MODEL} custom=InvalidProp:1 ! \
    fakesink" \
    4_n 0 1 $PERFORMANCE

report

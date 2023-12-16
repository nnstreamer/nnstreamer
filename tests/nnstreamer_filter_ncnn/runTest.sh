#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Kijun Shin <sharaelong@snu.ac.kr>
## @date Dec 16 2023
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
  
gstTest "your gst-launch-1.0 Arguments" "test case ID" "set 1 if this is not critical" "set 1 if this passes if gstLaunch fails" "set 1 to enable PERFORMANCE test" "set a positive value (seconds) to enable timeout mode"

PATH_TO_PARAM="../test_models/models/ncnn_models/squeezenet_v1.1.param"
PATH_TO_BIN="../test_models/models/ncnn_models/squeezenet_v1.1.param_v1.1.bin"
PATH_TO_LABEL="../test_models/labels/squeezenet_labels.txt"
PATH_TO_IMAGE="../test_models/data/orange.png"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} input=227:227:3 inputtype=float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 1 0 0 $PERFORMANCE
cat ncnn.out.log | grep "potpie"
testResult $? 1 "Golden test comparison" 0 1

# Fail test for invalid input dimension
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} input=227:227:3:1:1 inputtype=float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 2 0 1 $PERFORMANCE

# Fail test for invalid number of model files
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN},${PATH_TO_PARAM} input=227:227:3 inputtype=float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 3 0 1 $PERFORMANCE

# Fail test for invalid input matrices
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} input=227:227:3 inputtype=float32,float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 4 0 1 $PERFORMANCE

# Fail test for invalid output matrices
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} input=227:227:3 inputtype=float32 output=1000:1 outputtype=float32,float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 5 0 1 $PERFORMANCE

# Fail test for invalid argument
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} custom=use_yolo_decoder=auto input=227:227:3 inputtype=float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 6 0 1 $PERFORMANCE

# Fail test for invalid argument
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} custom=use_vulkan=true input=227:227:3 inputtype=float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 7 0 1 $PERFORMANCE


function run_pipeline() {
    gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} custom=use_vulkan=true input=227:227:3 inputtype=float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log 2>info
}

# Property reading test for accelerator
# run_pipeline true:GPU
# cat info | grep "vulkan"
# testResult $? 8 "vulkan accelerator test" 0 1

# Cleanup
rm info

report

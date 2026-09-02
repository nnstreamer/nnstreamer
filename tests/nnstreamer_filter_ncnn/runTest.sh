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

if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep ncnn.so)
        if [[ ! $check ]]; then
            echo "Cannot find ncnn shared lib"
            report
            exit
        fi
    else
        echo "Cannot find ${ini_path}"
        report
        exit
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
            check=$(ls ${value} | grep ncnn.so)
            if [[ ! $check ]]; then
                echo "Cannot find ncnn shared lib"
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

PATH_TO_PARAM="../test_models/models/ncnn_models/squeezenet_v1.1.param"
PATH_TO_BIN="../test_models/models/ncnn_models/squeezenet_v1.1.bin"
PATH_TO_LABEL="../test_models/labels/squeezenet_labels.txt"
PATH_TO_IMAGE="../test_models/data/orange.png"
# A single Input layer, so the network hands its input straight back and the
# shape of what the yolo decoder reads is whatever the pad caps carry.
PATH_TO_PASSTHROUGH_PARAM="../test_models/models/ncnn_models/passthrough.param"
PATH_TO_PASSTHROUGH_BIN="../test_models/models/ncnn_models/passthrough.bin"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze num-buffers=2 ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} input=227:227:3 inputtype=float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 1 0 0 $PERFORMANCE
# Both frames must be labeled; a stale or uninitialized output tensor would
# leave the first one unlabeled.
test "$(grep -ao orange ncnn.out.log | wc -l)" = "2"
testResult $? 1 "Golden test comparison" 0 1

# Fail test for invalid input dimension; ncnn matrices hold up to 4 axes.
# Trailing 1s do not count, so an over-ranked dimension needs a real 5th axis.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} input=227:227:3:2:2 inputtype=float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 2 0 1 $PERFORMANCE

# Fail test for invalid number of model files
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN},${PATH_TO_PARAM} input=227:227:3 inputtype=float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 3 0 1 $PERFORMANCE

# Fail test for invalid input matrices
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} input=227:227:3 inputtype=float32,float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 4 0 1 $PERFORMANCE

# Fail test for invalid output matrices
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} input=227:227:3 inputtype=float32 output=1000:1 outputtype=float32,float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 5 0 1 $PERFORMANCE

# Fail test for invalid argument
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} custom=use_yolo_decoder:fail input=227:227:3 inputtype=float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 6 0 1 $PERFORMANCE

# Fail test for invalid argument
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} custom=nakluv_esu=true input=227:227:3 inputtype=float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 7 0 1 $PERFORMANCE


# The dimension and type properties are optional (issue #4642). The input is
# taken from the pad caps and the output is inferred from the model, so each of
# the four combinations below has to produce the very same labels as case 1.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze num-buffers=2 ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 9 0 0 $PERFORMANCE
test "$(grep -ao orange ncnn.out.log | wc -l)" = "2"
testResult $? 9 "Golden test comparison without dimension and type properties" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze num-buffers=2 ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} input=227:227:3 inputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 10 0 0 $PERFORMANCE
test "$(grep -ao orange ncnn.out.log | wc -l)" = "2"
testResult $? 10 "Golden test comparison with the output inferred from the model" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze num-buffers=2 ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 11 0 0 $PERFORMANCE
test "$(grep -ao orange ncnn.out.log | wc -l)" = "2"
testResult $? 11 "Golden test comparison with the input taken from the pad caps" 0 1

# nnstreamer pads a dimension with trailing 1s, which an ncnn matrix does not carry
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze num-buffers=2 ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} input=227:227:3:1:1 inputtype=float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 12 0 0 $PERFORMANCE
test "$(grep -ao orange ncnn.out.log | wc -l)" = "2"
testResult $? 12 "Golden test comparison with a trailing-1 padded input dimension" 0 1

# A single buffer has to be labeled as well; the output tensor of the very first
# invoke used to be left uninitialized.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze num-buffers=1 ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} input=227:227:3 inputtype=float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 13 0 0 $PERFORMANCE
test "$(grep -ao orange ncnn.out.log | wc -l)" = "1"
testResult $? 13 "Golden test comparison of the very first frame" 0 1

# Fail test for a type that ncnn cannot handle
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} input=227:227:3 inputtype=uint8 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 14_n 0 1 $PERFORMANCE

# Fail test for the yolo decoder, which cannot infer its output shape
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} custom=use_yolo_decoder:true input=227:227:3 inputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log" 15_n 0 1 $PERFORMANCE

# Fail test for a detection row narrower than the 6 values the yolo decoder reads
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=GRAY8,width=3,height=2,framerate=30/1 ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tensor_filter framework=ncnn model=${PATH_TO_PASSTHROUGH_PARAM},${PATH_TO_PASSTHROUGH_BIN} custom=use_yolo_decoder:true output=10:5:1 outputtype=float32 ! fakesink" 16_n 0 1 $PERFORMANCE

# Fail test for a yolo output row with no room for the 5 box fields. The source
# is shaped so that the model fills every row, which is when the last one would
# be written past the end of the output tensor.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=8,height=2,framerate=30/1 ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PASSTHROUGH_PARAM},${PATH_TO_PASSTHROUGH_BIN} custom=use_yolo_decoder:true output=4:2:1 outputtype=float32 ! fakesink" 17_n 0 1 $PERFORMANCE

function run_pipeline() {
    gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! videoscale ! video/x-raw,width=227,height=227,format=BGR,framerate=0/1 ! tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5 ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_filter framework=ncnn model=${PATH_TO_PARAM},${PATH_TO_BIN} accelerator=$1 input=227:227:3 inputtype=float32 output=1000:1 outputtype=float32 ! tensor_decoder mode=image_labeling option1=${PATH_TO_LABEL} ! filesink location=ncnn.out.log 2>info
}

# Property reading test for accelerator
run_pipeline true:gpu
cat info | grep "accl = gpu$"
testResult $? 8 "vulkan accelerator test" 0 1

# Cleanup
rm info *.log

report

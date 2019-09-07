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
SYSTEM=`uname -m`

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow-lite model=${PATH_TO_MODEL} ! filesink location=tensorfilter.out.log" 1 0 0 $PERFORMANCE
python checkLabel.py tensorfilter.out.log ${PATH_TO_LABEL} orange
testResult $? 1 "Golden test comparison" 0 1

# Fail test for invalid input properties
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow-lite model=${PATH_TO_MODEL} input=7:1 inputtype=float32 ! filesink location=tensorfilter.out.log" 2F_n 0 1 $PERFORMANCE

# Fail test for invalid output properties
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow-lite model=${PATH_TO_MODEL} output=1:7 outputtype=int8 ! filesink location=tensorfilter.out.log" 3F_n 0 1 $PERFORMANCE

# Property reading test for nnapi
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow-lite model=${PATH_TO_MODEL} nnapi=true:cpu ! filesink location=tensorfilter.out.log" 2-1 1 0 $PERFORMANCE 2> info
# This test is not critical because NNAPI is not available in some systems (it's available with Tizen-ARM only!)
if [ "$SYSTEM" == "armv7l" ] || [ "$SYSTEM" == "aarch64" ]; then
	cat info | grep "true : cpu"
	testResult $? 2-1a "NNAPI activaion test" 1 1
fi

# Property reading test for nnapi
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow-lite model=${PATH_TO_MODEL} nnapi=true ! filesink location=tensorfilter.out.log" 2-2 1 0 $PERFORMANCE 2> info
# This test is not critical because NNAPI is not available in some systems (it's available with Tizen-ARM only!)
if [ "$SYSTEM" == "armv7l" ] || [ "$SYSTEM" == "aarch64" ]; then
	cat info | grep "true : cpu"
	testResult $? 2-2a "NNAPI activation test" 1 1
fi

# Property reading test for nnapi
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow-lite model=${PATH_TO_MODEL} nnapi=true:gpu ! filesink location=tensorfilter.out.log" 2-3 1 0 $PERFORMANCE 2> info
# This test is not critical because NNAPI is not available in some systems (it's available with Tizen-ARM only!)
if [ "$SYSTEM" == "armv7l" ] || [ "$SYSTEM" == "aarch64" ]; then
	cat info | grep "true : gpu"
	testResult $? 2-3a "NNAPI activation test" 1 1
	# This test is not critical because NNAPI is not available in some systems (it's available with Tizen-ARM only!)
fi

report

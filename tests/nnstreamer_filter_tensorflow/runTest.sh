#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author HyoungJoo Ahn <hello.ahn@samsung.com>
## @date Dec 17 2018
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

if [ -z ${SO_EXT} ]; then
    SO_EXT="so"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

# NNStreamer and plugins path for test
PATH_TO_PLUGIN="../../build"

if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep tensorflow.${SO_EXT})
        if [[ ! $check ]]; then
            echo "Cannot find tensorflow shared lib"
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
            check=$(ls ${value} | grep tensorflow.${SO_EXT})
            if [[ ! $check ]]; then
                echo "Cannot find tensorflow shared lib"
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

# Test with mnist model
PATH_TO_MODEL="../test_models/models/mnist.pb"
PATH_TO_DATA="../test_models/data/9.raw"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_DATA} ! application/octet-stream ! tensor_converter input-dim=784:1 input-type=uint8 ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter framework=tensorflow model=${PATH_TO_MODEL} input=784:1 inputtype=float32 inputname=input output=10:1 outputtype=float32 outputname=softmax ! filesink location=tensorfilter.out.1.log " 1 0 0 $PERFORMANCE
python3 checkLabel.py tensorfilter.out.1.log 9
testResult $? 1 "Golden test comparison" 0 1

# Input and output comnination test
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc pattern=13 num-buffers=1 ! videoconvert !  video/x-raw,width=640,height=480,framerate=30/1 ! tensor_converter ! tee name=t t. ! queue ! filesink location=combi.dummy.golden buffer-mode=unbuffered sync=false async=false t. ! queue ! mux.sink_0 filesrc location=${PATH_TO_DATA} ! application/octet-stream ! tensor_converter input-dim=784:1 input-type=uint8 ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! mux.sink_1 tensor_mux name=mux ! tensor_filter framework=tensorflow model=${PATH_TO_MODEL} input=784:1 inputtype=float32 inputname=input output=10:1 outputtype=float32 outputname=softmax input-combination=1 output-combination=i0,o0 ! tensor_demux name=demux demux.src_0 ! queue ! filesink location=tensorfilter.combi.in.log buffer-mode=unbuffered sync=false async=false demux.src_1 ! queue ! filesink location=tensorfilter.out.1.log buffer-mode=unbuffered sync=false async=false" 2 0 0 $PERFORMANCE
callCompareTest combi.dummy.golden tensorfilter.combi.in.log 2_0 "Output Combination Golden Test 2-0" 1 0
python3 checkLabel.py tensorfilter.out.1.log 9
testResult $? 1 "Golden test comparison" 0 1

# Test with speech command model (.wav file, answer is 'yes', this model has a input type DT_STRING.)
PATH_TO_MODEL="../test_models/models/conv_actions_frozen.pb"
PATH_TO_DATA="../test_models/data/yes.wav"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_DATA} blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=1:16022 input-type=int16 ! tensor_filter framework=tensorflow model=${PATH_TO_MODEL} input=1:16022 inputtype=int16 inputname=wav_data output=12:1 outputtype=float32 outputname=labels_softmax ! filesink location=tensorfilter.out.3.log " 3 0 0 $PERFORMANCE
python3 checkLabel.py tensorfilter.out.3.log 2
testResult $? 2 "Golden test comparison" 0 1

report

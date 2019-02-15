#!/usr/bin/env bash
##
## @file runTest.sh
## @author HyoungJoo Ahn <hello.ahn@samsung.com>
## @date Dec 17 2018
## @brief SSAT Test Cases for NNStreamer
##

if [[ "$SSATAPILOADED" != "1" ]];then
	SILENT=0
	INDEPENDENT=1
	search="ssat-api.sh"
	source $search
	printf "${Blue}Independent Mode${NC}
"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1
if [[ -z "${TEST_TENSORFLOW}" ]]; then
    report
fi

# Test with mnist model
PATH_TO_PLUGIN="../../build"
PATH_TO_MODEL="../test_models/models/mnist.pb"
PATH_TO_DATA="data/9.raw"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_DATA} ! application/octet-stream ! tensor_converter input-dim=784:1 input-type=uint8 ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! tensor_filter framework=tensorflow model=${PATH_TO_MODEL} input=784:1:1:1 inputtype=float32 inputname=input output=10:1:1:1 outputtype=float32 outputname=softmax ! filesink location=tensorfilter.out.1.log " 1 0 0 $PERFORMANCE
python checkLabel.py tensorfilter.out.1.log ${PATH_TO_DATA}
testResult $? 1 "Golden test comparison" 0 1

# Test with speech command model (.wav file, answer is 'yes', this model has a input type DT_STRING.)
PATH_TO_MODEL="../test_models/models/conv_actions_frozen.pb"
PATH_TO_DATA="data/2.wav"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_DATA} blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=1:16022 input-type=int16 ! tensor_filter framework=tensorflow model=${PATH_TO_MODEL} input=1:16022 inputtype=int16 inputname=wav_data output=12 outputtype=float32 outputname=labels_softmax ! filesink location=tensorfilter.out.2.log " 2 0 0 $PERFORMANCE
python checkLabel.py tensorfilter.out.2.log ${PATH_TO_DATA}
testResult $? 2 "Golden test comparison" 0 1

report

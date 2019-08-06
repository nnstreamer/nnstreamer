#!/usr/bin/env bash
##
## @file runTest.sh
## @author HyoungJoo Ahn <hello.ahn@samsung.com>
## @date Jun 17 2019
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
        check=$(ls ${ini_path} | grep caffe2.so)
        if [[ ! $check ]]; then
            echo "Cannot find caffe2 shared lib"
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
            check=$(ls ${value} | grep caffe2.so)
            if [[ ! $check ]]; then
                echo "Cannot find caffe2 shared lib"
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

# Test with Classifier model
PATH_TO_INIT_MODEL="../test_models/models/caffe2_init_net.pb"
PATH_TO_PRED_MODEL="../test_models/models/caffe2_predict_net.pb"
PATH_TO_DATA="./data/5"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_DATA} blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=32:32:3:1 input-type=float32 ! tensor_filter framework=caffe2 model=\"${PATH_TO_INIT_MODEL},${PATH_TO_PRED_MODEL}\" inputname=data input=32:32:3:1 inputtype=float32 output=10:1:1:1 outputtype=float32 outputname=softmax ! filesink location=tensorfilter.out.log" 1 0 0 $PERFORMANCE
python checkLabel.py tensorfilter.out.log 5
testResult $? 1 "Golden test comparison" 0 1
report

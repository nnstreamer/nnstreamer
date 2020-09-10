#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Jijoong Moon <jijoong.moon@samsung.com>
## @date Sept 10 2020
## @brief SSAT Test Cases for NNTrainer tensor filter
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
        check=$(ls ${ini_path} | grep nntrainer.${SO_EXT})
        if [[ ! $check ]]; then
            echo "Cannot find nntrainer shared lib"
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
            check=$(ls ${value} | grep nntrainer.${SO_EXT})
            if [[ ! $check ]]; then
                echo "Cannot find nntrainer shared lib"
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
PATH_TO_CONFIG="../test_models/models/mnist.ini"
PATH_TO_DATA="../test_models/data/2.png"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_DATA} ! pngdec ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tensor_filter framework=nntrainer model=${PATH_TO_CONFIG} input=1:28:28:1 inputtype=float32 output=1:10:1:1 outputtype=float32 ! filesink location=nntrainer.out.1.log" 1 0 0 $PERFORMANCE
python checkLabel.py nntrainer.out.1.log 2
testResult $? 1 "Golden test comparison" 0 1

PATH_TO_CONFIG="../test_models/models/mnist.ini"
PATH_TO_DATA="../test_models/data/0.png"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_DATA} ! pngdec ! tensor_converter ! tensor_transform mode=typecast option=float32 ! tensor_filter framework=nntrainer model=${PATH_TO_CONFIG} input=1:28:28:1 inputtype=float32 output=1:10:1:1 outputtype=float32 ! filesink location=nntrainer.out.2.log" 1 0 0 $PERFORMANCE
python checkLabel.py nntrainer.out.2.log 0
testResult $? 2 "Golden test comparison" 0 1

report

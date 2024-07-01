#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Bram Veldhoen
## @date Jun 2024
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
        check=$(ls ${ini_path} | grep dali.so)
        if [[ ! $check ]]; then
            echo "Cannot find dali shared lib"
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
            check=$(ls ${value} | grep dali.so)
            if [[ ! $check ]]; then
                echo "Cannot find dali lib"
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

# Create the dali pipeline
./create_dali_pipeline ./ 320 320 3

PATH_TO_PIPELINE="dali_pipeline_N_3_320_320.bin"
PATH_TO_IMAGE="../test_models/data/orange.png"
PATH_TO_LOG="dali_result_%1d.log"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    filesrc location=${PATH_TO_IMAGE} ! \
    pngdec ! \
    videoscale ! \
    imagefreeze ! \
    videoconvert ! \
    capsfilter caps=video/x-raw,width=1000,height=1000,format=RGB,framerate=0/1 ! \
    tensor_converter ! \
    capsfilter caps=other/tensors,num_tensors=1,types=uint8,dimensions=3:1000:1000:1,format=static ! \
    tensor_filter framework=dali model=${PATH_TO_PIPELINE} inputname=input0 input=3:1000:1000:1 inputtype=uint8 output=320:320:3:1 outputtype=float32 ! \
    capsfilter caps=other/tensors,num_tensors=1,types=float32,dimensions=320:320:3:1,format=static ! \
    filesink location=${PATH_TO_LOG} " \
    1 0 0 $PERFORMANCE

# Cleanup
rm dali_result_*.log*

report

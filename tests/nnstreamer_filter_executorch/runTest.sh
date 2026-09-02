#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Yongjoo Ahn <yongjoo1.ahn@samsung.com>
## @date July 8th 2024
## @brief SSAT Test Cases for NNStreamer executorch plugin
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

PATH_TO_PLUGIN="../../build"
if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep executorch.so)
        if [[ ! $check ]]; then
            echo "Cannot find executorch shared lib"
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
            check=$(ls ${value} | grep executorch.so)
            if [[ ! $check ]]; then
                echo "Cannot find executorch shared lib"
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

if [ "$SKIPGEN" == "YES" ]; then
    echo "Test Case Generation Skipped"
    sopath=$2
else
    echo "Test Case Generation Started"
    python3 ../nnstreamer_converter/generateTest.py
    sopath=$1
fi

# Test high rank input output tensors
PATH_TO_MODEL="../test_models/models/sample_3x4_two_input_two_output.pte"

## wrong input type : (expected) float32 vs uint8
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=2 ! videoscale ! videoconvert ! video/x-raw,width=4,height=3,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! other/tensors,num_tensors=1,dimensions=4:3:1:1,types=uint8,format=static,framerate=0/1 ! tee name=t t. ! queue ! mux.sink_0 t. ! queue ! mux.sink_1  tensor_mux name=mux sync_mode=nosync ! queue ! tensor_filter framework=executorch model=${PATH_TO_MODEL} ! tensor_sink" 1_n 0 1 $PERFORMANCE

## wrong input dimension : (expected) 4:3.4:3 vs 3:4:1:1.3:4:1:1
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=2 ! videoscale ! videoconvert ! video/x-raw,width=4,height=3,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=2:1:0:3 ! other/tensors,num_tensors=1,dimensions=3:4:1:1,types=uint8,format=static,framerate=0/1 ! tensor_transform mode=typecast option=float32 ! tee name=t t. ! queue ! mux.sink_0 t. ! queue ! mux.sink_1  tensor_mux name=mux sync_mode=nosync ! queue ! tensor_filter framework=executorch model=${PATH_TO_MODEL} ! tensor_sink" 2_n 0 1 $PERFORMANCE

## correct input/output info
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=2 ! videoscale ! videoconvert ! video/x-raw,width=4,height=3,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! other/tensors,num_tensors=1,dimensions=4:3:1:1,types=uint8,format=static,framerate=0/1 ! tensor_transform mode=typecast option=float32 ! tee name=t t. ! queue ! mux.sink_0 t. ! queue ! mux.sink_1  tensor_mux name=mux sync_mode=nosync ! queue ! tensor_filter framework=executorch model=${PATH_TO_MODEL} ! filesink location=tensorfilter.out.log" 3 0 0 $PERFORMANCE

# Test multiple input output tensors
PATH_TO_MODEL="../test_models/models/sample_4x4x4x4x4_two_input_one_output.pte"

## wrong input type : (expected) float32 vs uint8
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=\"test_00.dat\" blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=4:4:4:4:4 input-type=uint8 ! tee name=t t. ! queue ! mux.sink_0 t. ! queue ! mux.sink_1  tensor_mux name=mux sync_mode=nosync ! queue ! tensor_filter framework=executorch model=${PATH_TO_MODEL} ! filesink location=tensorfilter.out.log" 4_n 0 1 $PERFORMANCE

## correct input/output info
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=\"test_00.dat\" blocksize=-1 ! application/octet-stream ! tensor_converter input-dim=4:4:4:4:4 input-type=float32 ! tee name=t t. ! queue ! mux.sink_0 t. ! queue ! mux.sink_1  tensor_mux name=mux sync_mode=nosync ! queue ! tensor_filter framework=executorch model=${PATH_TO_MODEL} ! filesink location=tensorfilter.out.log" 5 0 0 $PERFORMANCE
callCompareTest test_00.dat.golden tensorfilter.out.log 6 "Compare 5" 1 0

# Cleanup
rm *.log *.golden *.dat

report

## The .pte fixtures under ../test_models/models are rebuilt by
## generateModel.py in this directory. They are regenerated rather than
## hand-edited: ExecuTorch 1.4 rejects the serialized form the 2024 exporter
## produced with Error::InvalidProgram.

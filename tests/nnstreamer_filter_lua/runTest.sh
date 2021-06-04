#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Yongjoo Ahn <yongjoo1.ahn@samsung.com>
## @date Jun 3 2021
## @brief SSAT Test Cases for NNStreamer tensor_filter::lua
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

PATH_TO_PLUGIN="../../build"
# Check lua libraies are built
if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep lua.so)
        if [[ ! $check ]]; then
            echo "Cannot find lua shared lib"
            report
            exit
        fi
    else
        echo "Cannot find ${ini_path}"
        report exit
    fi
else
    echo "No build directory"
    report
    exit
fi

PATH_TO_PASSTHROUGH_MODEL="../test_models/models/passthrough.lua"
PATH_TO_SCALER_MODEL="../test_models/models/scaler.lua"

# Passthrough test
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=lua model=\"${PATH_TO_PASSTHROUGH_MODEL}\" ! filesink location=\"testcase1.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase1.direct.log\" sync=true" 1 0 0 $PERFORMANCE
callCompareTest testcase1.direct.log testcase1.passthrough.log 1 "Compare 1" 0 0

# Passthrough -> Scaler test
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=lua model=\"${PATH_TO_PASSTHROUGH_MODEL}\" ! tensor_filter framework=lua model=\"${PATH_TO_SCALER_MODEL}\" ! filesink location=\"testcase2.scaled.log\" sync=true t. ! queue ! filesink location=\"testcase2.direct.log\" sync=true" 2 0 0 $PERFORMANCE
python3 ../nnstreamer_filter_python3/checkScaledTensor.py testcase2.direct.log 640 480 testcase2.scaled.log 320 240 3
testResult $? 2 "Golden test comparison" 0 1

# Passthrough test with given string script
PASSTHROUGH_SCRIPT="inputTensorInfo = {dim = {3, 299, 299, 1}, type = 'uint8_t' } outputTensorInfo = {dim = {3, 299, 299, 1}, type = 'uint8_t' } function nnstreamer_invoke() input = input_tensor() output = output_tensor() for i=1,3*299*299 do output[i] = input[i] end end"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=299,height=299,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=lua model=\"${PASSTHROUGH_SCRIPT}\" ! filesink location=\"testcase3.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase3.direct.log\" sync=true" 3 0 0 $PERFORMANCE
callCompareTest testcase3.direct.log testcase3.passthrough.log 1 "Compare 2" 0 0


rm *.log

report

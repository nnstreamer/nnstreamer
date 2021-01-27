#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Dongju Chae <dongju.chae@samsung.com>
## @date Apr 3 2019
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

PATH_TO_PLUGIN="../../build"
# Check python libraies are built
if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep python3.so)
        if [[ ! $check ]]; then
            echo "Cannot find python shared lib"
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

FRAMEWORK="python3"
# This symlink is necessary only for testcases; when installed, symlinks will be made
pushd ../../build/ext/nnstreamer/tensor_filter
TEST_PYTHONPATH=${FRAMEWORK}_pymodule
mkdir -p ${TEST_PYTHONPATH}
pushd ${TEST_PYTHONPATH}
# Covert to an absolute path from the relative path
export PYTHONPATH=$(pwd)
if [[ ! -f ./nnstreamer_python.so ]]; then
  ln -s ../nnstreamer_${FRAMEWORK}.so nnstreamer_python.so
fi
popd
popd

# Passthrough test
PATH_TO_SCRIPT="../test_models/models/passthrough.py"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"${FRAMEWORK}\" model=\"${PATH_TO_SCRIPT}\" input=\"3:280:40:1\" inputtype=\"uint8\" output=\"3:280:40:1\" outputtype=\"uint8\" ! filesink location=\"testcase1.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase1.direct.log\" sync=true" 1 0 0 $PERFORMANCE
callCompareTest testcase1.direct.log testcase1.passthrough.log 1 "Compare 1" 0 0

# Scaler test
# 1) 640x480 --> 320x240
PATH_TO_SCRIPT="../test_models/models/scaler.py"
ARGUMENTS="320x240"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"${FRAMEWORK}\" model=\"${PATH_TO_SCRIPT}\" custom=\"${ARGUMENTS}\" ! filesink location=\"testcase2.scaled.log\" sync=true t. ! queue ! filesink location=\"testcase2.direct.log\" sync=true" 2 0 0 $PERFORMANCE
python3 checkScaledTensor.py testcase2.direct.log 640 480 testcase2.scaled.log 320 240 3
testResult $? 2 "Golden test comparison" 0 1

# 2) 640x480 --> 1280x960
ARGUMENTS="1280x960"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"${FRAMEWORK}\" model=\"${PATH_TO_SCRIPT}\" custom=\"${ARGUMENTS}\" ! filesink location=\"testcase3.scaled.log\" sync=true t. ! queue ! filesink location=\"testcase3.direct.log\" sync=true" 3 0 0 $PERFORMANCE
python3 checkScaledTensor.py testcase3.direct.log 640 480 testcase3.scaled.log 1280 960 3
testResult $? 3 "Golden test comparison" 0 1

rm *.log

report

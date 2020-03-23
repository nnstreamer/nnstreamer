#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author MyungJoo Ham <myungjoo.ham@gmail.com>
## @author Jijoong Moon <jijoong.moon@gmail.com>
## @date Nov 01 2018
## @brief This is a template file for SSAT test cases. You may designate your own license.
##
if [[ "$SSATAPILOADED" != "1" ]]; then
    SILENT=0
    INDEPENDENT=1
    search="ssat-api.sh"
    source $search
    printf "${Blue}Independent Mode${NC}
"
fi
testInit $1 # You may replace this with Test Group Name


PATH_TO_PLUGIN="../../build"
if [ ! -d "${PATH_TO_PLUGIN}" ]; then
    CUSTOMLIB_DIR=${CUSTOMLIB_DIR:="/usr/lib/nnstreamer/customfilters"}
fi

if [[ -z "${CUSTOMLIB_DIR}" ]]; then
    RNN_DIR="../../build/nnstreamer_example/custom_example_RNN"
else
    RNN_DIR="${CUSTOMLIB_DIR}"
fi

if [ -z ${SO_EXT} ]; then
    SO_EXT="so"
fi

# Generate video_4x4xBGRx.xraw
python generateTestCase.py

gstTest "--gst-plugin-path=../../build tensor_mux name=mux ! tensor_filter framework=custom model=${RNN_DIR}/libdummyRNN.${SO_EXT} ! tee name=t ! queue ! tensor_reposink slot-index=0 silent=false filesrc location=\"video_4x4xBGRx.xraw\" ! application/octet-stream ! tensor_converter input-dim=4:4:4:1 input-type=uint8 ! mux.sink_0 tensor_reposrc slot-index=0 silent=false caps=\"other/tensor,dimension=(string)4:4:4:1,type=(string)uint8,framerate=(fraction)0/1\" ! mux.sink_1 t. ! queue ! multifilesink location=\"out_%1d.log\"" 1 0 0 $PERFORMANCE

callCompareTest rnn.golden out_9.log 1-1 "Compare 1-1" 1 0

report

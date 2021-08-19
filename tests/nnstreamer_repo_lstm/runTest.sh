#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Jijoong Moon <jijoong.moon@samsung.com>
## @date Nov 28 2018
## @brief This is Testcase for the dummy lstm
##
##
##
##                                  repository
##                              +-----------------+
## +----------------------------|     slot 1      |<-----------------------------+
## |                            +-----------------+                              |
## |  +-------------------------|     slot 0      |<---------------------------+ |
## |  |                         +-----------------+                            | |
## |  |                                                                        | |
## |  |                  +-------+    +--------+    +-------+                  | |
## |  +-->repo_src:0 --->|       |--->|        |--->|       |---->repo_sink:0 -+ |
## +----->repo_src:1 --->|  MUX  |    | Filter |    | DEMUX |---->repo_sink:1 ---+
##        filesrc(new)-->|       |    |        |    |       |  |
##                       +-------+    +--------+    +-------+  -->out_%1d.log
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
    LSTM_DIR="../../build/tests/nnstreamer_example"
else
    LSTM_DIR="${CUSTOMLIB_DIR}"
fi

if [ -z ${SO_EXT} ]; then
    SO_EXT="so"
fi

# Generate video_4x4xBGRx.xraw & golden
python3 generateTestCase.py

gstTest "--gst-plugin-path=../../build \
tensor_mux name=mux sync-mode=nosync ! \
tensor_filter framework=custom model=${LSTM_DIR}/libdummyLSTM.${SO_EXT} ! \
tensor_demux name=demux \
    demux.src_0 ! queue ! tensor_reposink slot-index=0 silent=false \
    demux.src_1 ! queue ! tee name=t \
        t. ! queue ! tensor_reposink slot-index=1 silent=false \
        t. ! queue ! multifilesink location=\"out_%1d.log\" \
    tensor_reposrc slot-index=0 silent=false caps=\"other/tensor,dimension=(string)4:4:4:1,type=(string)float32,framerate=(fraction)0/1\" ! mux.sink_0 \
    tensor_reposrc slot-index=1 silent=false caps=\"other/tensor,dimension=(string)4:4:4:1,type=(string)float32,framerate=(fraction)0/1\" ! mux.sink_1 \
    filesrc location=\"video_4x4xBGRx.xraw\" ! application/octet-stream ! tensor_converter input-dim=4:4:4:1 input-type=float32 ! mux.sink_2" \
1 0 0 $PERFORMANCE

callCompareTest lstm.golden out_9.log 1-1 "Compare 1-1" 1 0

rm *.log *.xraw *.golden

report

#!/usr/bin/env bash
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

if [[ -z "${CUSTOMLIB_DIR// /}" ]]; then
    LSTM_DIR="../../build/nnstreamer_example/custom_example_LSTM"
else
    LSTM_DIR="${CUSTOMLIB_DIR}"
fi

# Generate video_4x4xBGRx.xraw & golden
python generateTestCase.py

gstTest "--gst-plugin-path=../../build tensor_mux name=mux ! tensor_filter framework=custom model=${LSTM_DIR}/libdummyLSTM.so ! tensor_demux name=demux ! queue ! tensor_reposink slot-index=0 silent=false demux.src_1 ! queue ! tee name=t ! queue ! tensor_reposink slot-index=1 silent=false tensor_reposrc slot-index=0 silent=false caps=\"other/tensor,dimension=(string)4:4:4:1,type=(string)float32,framerate=(fraction)0/1\" ! mux.sink_0 tensor_reposrc slot-index=1 silent=false caps=\"other/tensor,dimension=(string)4:4:4:1,type=(string)float32,framerate=(fraction)0/1\" ! mux.sink_1 filesrc location=\"video_4x4xBGRx.xraw\" ! application/octet-stream ! tensor_converter input-dim=4:4:4:1 input-type=float32 ! mux.sink_2 t. ! queue ! multifilesink location=\"out_%1d.log\"" 1 0 0 $PERFORMANCE

callCompareTest lstm.golden out_9.log 1-1 "Compare 1-1" 1 0

report

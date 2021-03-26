#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author MyungJoo Ham <myungjoo.ham@gmail.com>
## @date Nov 01 2018
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

if [ "$SKIPGEN" == "YES" ]; then
    echo "Test Case Generation Skipped"
    sopath=$2
else
    echo "Test Case Generation Started"
    python ../nnstreamer_converter/generateGoldenTestResult.py 8
    sopath=$1
fi
convertBMP2PNG

# Test gst availability. (0)
gstTest "videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! filesink location=\"testcase.apitest.log\" sync=true" 0 0 0 $PERFORMANCE

# Test with small stream (1, 2)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=dimchg option=0:2 ! filesink location=\"testcase01.dimchg02.log\" sync=true t. ! queue ! filesink location=\"testcase01.direct.log\" sync=true" 1 0 0 $PERFORMANCE
# dim1 = 3 (RGB)
# rbs = 16 x 16 x 3
# es = 1
python3 checkResult.py dimchg0:b testcase01.direct.log testcase01.dimchg02.log 3 768 1
testResult $? 1 "Golden test comparison" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t ! queue ! tensor_transform mode=dimchg option=0:2 ! filesink location=\"testcase02.dimchg02.log\" sync=true t. ! queue ! filesink location=\"testcase02.direct.log\" sync=true" 2 0 0 $PERFORMANCE
# dim1 = 3 (RGB)
# rbs = 16 x 16 x 3
# es = 1
python3 checkResult.py dimchg0:b testcase02.direct.log testcase02.dimchg02.log 4 1024 1
testResult $? 2 "Golden test comparison" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
        multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! tee name=t \
        t. ! queue ! mux.sink_0 \
        t. ! queue ! mux.sink_1 \
        t. ! queue ! mux.sink_2 \
        tensor_mux name=mux ! tensor_transform mode=dimchg option=0:2  ! tensor_demux name=demux \
        demux.src_0 ! queue ! filesink location=\"testcase03_0.dimchg02.log\" sync=true \
        demux.src_1 ! queue ! filesink location=\"testcase03_1.dimchg02.log\" sync=true \
        demux.src_2 ! queue ! filesink location=\"testcase03_2.dimchg02.log\" sync=true \
        t. ! queue ! filesink location=\"testcase03.direct.log\" sync=true" 3 0 0 $PERFORMANCE
python3 checkResult.py dimchg0:b testcase03.direct.log testcase03_0.dimchg02.log 4 1024 1
testResult $? 2 "Golden test comparison 3-0" 0 1
python3 checkResult.py dimchg0:b testcase03.direct.log testcase03_1.dimchg02.log 4 1024 1
testResult $? 2 "Golden test comparison 3-1" 0 1
python3 checkResult.py dimchg0:b testcase03.direct.log testcase03_2.dimchg02.log 4 1024 1
testResult $? 2 "Golden test comparison 3-2" 0 1

report

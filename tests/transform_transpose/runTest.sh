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
    printf "${Blue}Independent Mode${NC}"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

PATH_TO_PLUGIN="../../build"

if [ "$SKIPGEN" == "YES" ]; then
    echo "Test Case Generation Skipped"
    sopath=$2
else
    echo "Test Case Generation Started"
    python3 generateTest.py
    sopath=$1
fi

# Test gst availability. (0)
gstTest "videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! filesink location=\"testcase.apitest.log\" sync=true" 0 0 0 $PERFORMANCE

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"test01_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=100:50:3:1 input-type=float32 ! tensor_transform mode=transpose option=2:0:1:3 ! multifilesink location=\"./result01_%02d.log\" sync=true" 1 0 0 $PERFORMANCE

callCompareTest test01_00.dat.golden result01_00.log 1 "Compare 1" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"test02_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=200:100:3:1 input-type=float32 ! tensor_transform mode=transpose option=2:0:1:3 ! multifilesink location=\"./result02_%02d.log\" sync=true" 2 0 0 $PERFORMANCE

callCompareTest test02_00.dat.golden result02_00.log 2 "Compare 2" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"test03_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=200:100:3:1 input-type=float32 ! tensor_transform mode=transpose option=1:0:2:3 ! multifilesink location=\"./result03_%02d.log\" sync=true" 3 0 0 $PERFORMANCE

callCompareTest test03_00.dat.golden result03_00.log 3 "Compare 3" 1 0

# Test tensors stream
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
        multifilesrc location=\"test03_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=200:100:3:1 input-type=float32 ! tee name=t 
        t. ! queue ! mux.sink_0 \
        t. ! queue ! mux.sink_1 \
        t. ! queue ! mux.sink_2 \
        tensor_mux name=mux ! tensor_transform mode=transpose option=1:0:2:3 ! tensor_demux name=demux \
        demux.src_0 ! queue ! multifilesink location=\"./result04_0_%02d.log\" sync=true \
        demux.src_1 ! queue ! multifilesink location=\"./result04_1_%02d.log\" sync=true \
        demux.src_2 ! queue ! multifilesink location=\"./result04_2_%02d.log\" sync=true" 4 0 0 $PERFORMANCE

callCompareTest test03_00.dat.golden result04_0_00.log 3 "Compare 3" 1 0
callCompareTest test03_00.dat.golden result04_1_00.log 3 "Compare 3" 1 0
callCompareTest test03_00.dat.golden result04_2_00.log 3 "Compare 3" 1 0

# Dimension declaration test cases
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"test01_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=100:50:3:1 input-type=float32 ! tensor_transform mode=transpose option=2:0:1:3 ! other/tensor,dimension=3:100:50 ! multifilesink location=\"./result05_%02d.log\" sync=true" 5 0 0 $PERFORMANCE

callCompareTest test01_00.dat.golden result05_00.log 5 "Compare 5" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"test01_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=100:50:3:1 input-type=float32 ! tensor_transform mode=transpose option=2:0:1:3 ! other/tensors,num_tensors=1,dimensions=3:100:50 ! multifilesink location=\"./result06_%02d.log\" sync=true" 6 0 0 $PERFORMANCE

callCompareTest test01_00.dat.golden result06_00.log 6 "Compare 6" 1 0

rm *.log *.bmp *.png *.golden *.raw *.dat

report

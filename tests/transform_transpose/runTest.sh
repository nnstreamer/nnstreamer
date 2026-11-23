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
    python3 generateTest.py || (echo "Failed to run test preparation script (generateTest.py). Test not available." && report && exit)
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

# Check that transposing a multi-tensor buffer preserves each tensor's position.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
        filesrc blocksize=-1 location=test_mux_0.dat ! application/octet-stream ! tensor_converter input-dim=200:100:3:1 input-type=float32 ! queue ! mux.sink_0 \
        filesrc blocksize=-1 location=test_mux_1.dat ! application/octet-stream ! tensor_converter input-dim=200:100:3:1 input-type=float32 ! queue ! mux.sink_1 \
        filesrc blocksize=-1 location=test_mux_2.dat ! application/octet-stream ! tensor_converter input-dim=200:100:3:1 input-type=float32 ! queue ! mux.sink_2 \
        tensor_mux name=mux ! tensor_transform mode=transpose option=1:0:2:3 ! tensor_demux name=demux \
        demux.src_0 ! queue ! multifilesink location=\"./result04_0_%02d.log\" sync=true \
        demux.src_1 ! queue ! multifilesink location=\"./result04_1_%02d.log\" sync=true \
        demux.src_2 ! queue ! multifilesink location=\"./result04_2_%02d.log\" sync=true" 4 0 0 $PERFORMANCE

callCompareTest test_mux_0.dat.golden result04_0_00.log 4-0 "Compare 4-0" 1 0
callCompareTest test_mux_1.dat.golden result04_1_00.log 4-1 "Compare 4-1" 1 0
callCompareTest test_mux_2.dat.golden result04_2_00.log 4-2 "Compare 4-2" 1 0

# Dimension declaration test cases
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"test01_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=100:50:3:1 input-type=float32 ! tensor_transform mode=transpose option=2:0:1:3 ! other/tensor,dimension=3:100:50 ! multifilesink location=\"./result05_%02d.log\" sync=true" 5 0 0 $PERFORMANCE

callCompareTest test01_00.dat.golden result05_00.log 5 "Compare 5" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"test01_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=100:50:3:1 input-type=float32 ! tensor_transform mode=transpose option=2:0:1:3 ! other/tensors,format=static,num_tensors=1,dimensions=3:100:50 ! multifilesink location=\"./result06_%02d.log\" sync=true" 6 0 0 $PERFORMANCE

callCompareTest test01_00.dat.golden result06_00.log 6 "Compare 6" 1 0

# The option takes four indices only, even for a tensor of a higher rank
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"test01_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=10:50:3:1:10 input-type=float32 ! tensor_transform mode=transpose option=2:0:1:3:4 ! multifilesink location=\"./result07_%02d.log\" sync=true" 7_n 0 1 $PERFORMANCE

# Dimensions 4 and above keep their place and are transposed as well
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"test04_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=100:50:3:2:3 input-type=float32 ! tensor_transform mode=transpose option=2:0:1:3 ! other/tensor,dimension=3:100:50:2:3 ! multifilesink location=\"./result08_%02d.log\" sync=true" 8 0 0 $PERFORMANCE

callCompareTest test04_00.dat.golden result08_00.log 8 "Compare 8" 1 0

# A rank-2 tensor: dimensions 0 and 1 swap and the output stays rank 2.
# The dimension is typed as a string; gst-launch reads a bare 5:4 as a flagset.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"test05_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=4:5 input-type=float32 ! tensor_transform mode=transpose option=1:0:2:3 ! \"other/tensor,dimension=(string)5:4\" ! multifilesink location=\"./result09_%02d.log\" sync=true" 9 0 0 $PERFORMANCE

callCompareTest test05_00.dat.golden result09_00.log 9 "Compare 9" 1 0

# A rank-2 tensor cannot take its absent dimension 2 into dimension 0 or 1:
# the output would have a dimension of 0 and its caps are refused.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"test05_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=4:5 input-type=float32 ! tensor_transform mode=transpose option=2:0:1:3 ! multifilesink location=\"./result10_%02d.log\" sync=true" 10_n 0 1 $PERFORMANCE

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"test05_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=4:5 input-type=float32 ! tensor_transform mode=transpose option=0:2:1:3 ! multifilesink location=\"./result11_%02d.log\" sync=true" 11_n 0 1 $PERFORMANCE

# A rank-1 tensor: swapping its absent dimensions 1 and 2 still runs the
# transpose loop, which must copy every element.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"test06_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=4 input-type=float32 ! tensor_transform mode=transpose option=0:2:1:3 ! \"other/tensor,dimension=(string)4\" ! multifilesink location=\"./result12_%02d.log\" sync=true" 12 0 0 $PERFORMANCE

callCompareTest test06_00.dat.golden result12_00.log 12 "Compare 12" 1 0

rm *.log *.bmp *.png *.golden *.raw *.dat

report

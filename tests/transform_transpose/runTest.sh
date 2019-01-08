#!/usr/bin/env bash
##
## @file runTest.sh
## @author MyungJoo Ham <myungjoo.ham@gmail.com>
## @date Nov 01 2018
## @brief SSAT Test Cases for NNStreamer
##
if [[ "$SSATAPILOADED" != "1" ]]
then
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

if [ "$SKIPGEN" == "YES" ]
then
  echo "Test Case Generation Skipped"
  sopath=$2
else
  echo "Test Case Generation Started"
  python generateTest.py
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

report

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
else
  echo "Test Case Generation Started"
  python generateGoldenTestResult.py
fi
convertBMP2PNG

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=testcase01_RGB_640x480.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! testtensors silent=TRUE ! tensorscheck silent=TRUE ! filesink location=\"testcase01_RGB_640x480.nonip.log\" sync=true" 1 0 0 $PERFORMANCE

callCompareTest testcase01_RGB_640x480.golden testcase01_RGB_640x480.nonip.log 1 "Compare 1" 1 0

report

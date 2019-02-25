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
  python generateGoldenTestResult.py
  python ../nnstreamer_converter/generateGoldenTestResult.py 8
  sopath=$1
fi
convertBMP2PNG

function do_test {
    param="${4}_gen_golden_data"
    gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=testcase01_${1}_${2}x${3}.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=${1},width=${2},height=${3},framerate=0/1 ! filesink location=\"testcase01_${1}_${2}x${3}.golden.raw\" sync=true" $param 0 0 $PERFORMANCE

    gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=testcase01_${1}_${2}x${3}.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=${1},width=${2},height=${3},framerate=0/1 ! tensor_converter ! tensor_decoder mode=direct_video ! filesink location=\"testcase01_${1}_${2}x${3}.log\" sync=true" ${4} 0 0 $PERFORMANCE

    callCompareTest testcase01_${1}_${2}x${3}.golden.raw testcase01_${1}_${2}x${3}.log ${4} "Golden Test ${4}" 1 0
}

do_test RGB 640 480 1
do_test RGB 642 480 1-1
do_test BGRx 640 480 1-2
do_test BGRx 642 480 1-3

# Test with a stream of 10 small PNG frames
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_decoder mode=direct_video ! filesink location=\"testcase2.dec.log\" sync=true t. ! queue ! filesink location=\"testcase2.con.log\" sync=true" 2 0 0 $PERFORMANCE
callCompareTest testcase2.con.log testcase2.dec.log 2 "Compare for case 2" 0 0

# Expand the unit test coverage
## Set Property woth option*
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png,framerate=\(fraction\)30/1\" ! pngdec ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_decoder mode=direct_video option1=\"nothing\" option2=\"else\" option3=\"matters\" option4=\"whattheheck=is\" option5=\"goingon=idontknow\" option6=\"whydowehave\" option7=\"somany\" option8=\"options\" option9=\"whydontyouguess\" ! filesink location=\"testcase3.dec.log\" sync=true t. ! queue ! filesink location=\"testcase3.con.log\" sync=true" 3 0 0 $PERFORMANCE
callCompareTest testcase3.con.log testcase3.dec.log 3 "Compare for case 3" 0 0

## Trigger "decoder not configured"
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc ! tensor_converter ! tensor_decoder mode=direct_video ! video/mpeg ! fakesink" 4 0 1 $PERFORMANCE

report

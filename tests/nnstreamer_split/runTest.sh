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

if [ "$SKIPGEN" == "YES" ]
then
  echo "Test Case Generation Skipped"
  sopath=$2
else
  echo "Test Case Generation Started"
  python ../nnstreamer_converter/generateGoldenTestResult.py 11
  sopath=$1
fi
convertBMP2PNG

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw, format = RGB, width=100, height=100, framerate=0/1 ! tensor_converter ! tensor_split name=split tensorseg=3:100:100 split. ! queue ! filesink location=split00.log" 1 0 0 $PERFORMANCE

callCompareTest testcase_0_0.golden split00.log 1 "Compare 1" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw, format = RGB, width=100, height=100, framerate=0/1 ! tensor_converter ! tensor_split name=split tensorseg=1:100:100,2:100:100 split. ! queue ! filesink location=split01_0.log split. ! queue ! filesink location=split01_1.log" 2 0 0 $PERFORMANCE

callCompareTest testcase_1_0.golden split01_0.log 2_0 "Compare 2-0" 1 0
callCompareTest testcase_1_1.golden split01_1.log 2_1 "Compare 2-1" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw, format = RGB, width=100, height=100, framerate=0/1 ! tensor_converter ! tensor_split name=split tensorseg=1:100:100,1:100:100,1:100:100 split. ! queue ! filesink location=split02_0.log split. ! queue ! filesink location=split02_1.log split. ! queue ! filesink location=split02_2.log" 3 0 0 $PERFORMANCE

callCompareTest testcase_2_0.golden split02_0.log 3_0 "Compare 3-0" 1 0
callCompareTest testcase_2_1.golden split02_1.log 3_1 "Compare 3-1" 1 0
callCompareTest testcase_2_2.golden split02_2.log 3_2 "Compare 3-2" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! tensor_split name=split tensorseg=3:16:16 split. ! queue ! filesink location=split03.log" 4 0 0 $PERFORMANCE

callCompareTest testcase_stream.golden split03.log 4 "Compare 4" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! tensor_split name=split tensorseg=1:16:16,2:16:16 split. ! queue ! filesink location=split04_0.log split. ! queue ! filesink location=split04_1.log" 5 0 0 $PERFORMANCE

callCompareTest testcase_stream_1_0.golden split04_0.log 5_0 "Compare 5-0" 1 0
callCompareTest testcase_stream_1_1.golden split04_1.log 5_1 "Compare 5-1" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! tensor_split name=split tensorseg=1:16:16,1:16:16,1:16:16 split. ! queue ! filesink location=split05_0.log split. ! queue ! filesink location=split05_1.log split. ! queue ! filesink location=split05_2.log" 6 0 0 $PERFORMANCE

callCompareTest testcase_stream_2_0.golden split05_0.log 6_0 "Compare 6-0" 1 0
callCompareTest testcase_stream_2_1.golden split05_1.log 6_1 "Compare 6-1" 1 0
callCompareTest testcase_stream_2_2.golden split05_2.log 6_2 "Compare 6-2" 1 0

report

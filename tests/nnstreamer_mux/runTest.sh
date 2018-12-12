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
  python ../nnstreamer_converter/generateGoldenTestResult.py 9
  sopath=$1
fi
convertBMP2PNG

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! filesink location=testcase01_RGB_100x100.log filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0" 1 0 0 $PERFORMANCE

callCompareTest testcase01_RGB_100x100.golden testcase01_RGB_100x100.log 1 "Compare 1" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! filesink location=testcase02_RGB_100x100.log filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0 filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_1" 2 0 0 $PERFORMANCE

callCompareTest testcase02_RGB_100x100.golden testcase02_RGB_100x100.log 2 "Compare 2" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! filesink location=testcase03_RGB_100x100.log filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0 filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_1 filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_2" 3 0 0 $PERFORMANCE

callCompareTest testcase03_RGB_100x100.golden testcase03_RGB_100x100.log 3 "Compare 3" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! filesink location=testcase01.log multifilesrc location=\"testsequence01_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0" 4 0 0 $PERFORMANCE

callCompareTest testcase01.golden testcase01.log 4 "Compare 4" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! filesink location=testcase02.log multifilesrc location=\"testsequence02_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence02_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_1" 5 0 0 $PERFORMANCE

callCompareTest testcase02.golden testcase02.log 5 "Compare 5" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! filesink location=testcase03.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_1 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_2" 6 0 0 $PERFORMANCE

callCompareTest testcase03.golden testcase03.log 6 "Compare 6" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux ! filesink location=testcase04.log multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_1 multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_2 multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_3" 7 0 0 $PERFORMANCE

callCompareTest testcase03.golden testcase03.log 7 "Compare 7" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux sync_mode=slowest ! multifilesink location=testsynch00_%1d.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)10/1\" ! pngdec ! tensor_converter ! mux.sink_1" 8 0 0 $PERFORMANCE

callCompareTest testsynch00_0.golden testsynch00_0.log 8-1 "Compare 8-1" 1 0
callCompareTest testsynch00_1.golden testsynch00_1.log 8-2 "Compare 8-2" 1 0
callCompareTest testsynch00_2.golden testsynch00_2.log 8-3 "Compare 8-3" 1 0
callCompareTest testsynch00_3.golden testsynch00_3.log 8-4 "Compare 8-4" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux sync_mode=slowest ! multifilesink location=testsynch01_%1d.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)20/1\" ! pngdec ! tensor_converter ! mux.sink_1" 9 0 0 $PERFORMANCE

callCompareTest testsynch01_0.golden testsynch01_0.log 9-1 "Compare 9-1" 1 0
callCompareTest testsynch01_1.golden testsynch01_1.log 9-2 "Compare 9-2" 1 0
callCompareTest testsynch01_2.golden testsynch01_2.log 9-3 "Compare 9-3" 1 0
callCompareTest testsynch01_3.golden testsynch01_3.log 9-4 "Compare 9-4" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux sync_mode=slowest ! multifilesink location=testsynch02_%1d.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)20/1\" ! pngdec ! tensor_converter ! mux.sink_1 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)10/1\" ! pngdec ! tensor_converter ! mux.sink_2" 10 0 0 $PERFORMANCE

callCompareTest testsynch02_0.golden testsynch02_0.log 10-1 "Compare 10-1" 1 0
callCompareTest testsynch02_1.golden testsynch02_1.log 10-2 "Compare 10-2" 1 0
callCompareTest testsynch02_2.golden testsynch02_2.log 10-3 "Compare 10-3" 1 0
callCompareTest testsynch02_3.golden testsynch02_3.log 10-4 "Compare 10-4" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux silent=true sync_mode=basepad sync_option=0:33333333 ! multifilesink location=testsynch03_%1d.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)10/1\" ! pngdec ! tensor_converter ! mux.sink_1" 11 0 0 $PERFORMANCE

callCompareTest testsynch03_0.golden testsynch03_0.log 11-1 "Compare 11-1" 1 0
callCompareTest testsynch03_1.golden testsynch03_1.log 11-2 "Compare 11-2" 1 0
callCompareTest testsynch03_2.golden testsynch03_2.log 11-3 "Compare 11-3" 1 0
callCompareTest testsynch03_3.golden testsynch03_3.log 11-4 "Compare 11-4" 1 0
callCompareTest testsynch03_4.golden testsynch03_4.log 11-5 "Compare 11-5" 1 0
callCompareTest testsynch03_5.golden testsynch03_5.log 11-6 "Compare 11-6" 1 0
callCompareTest testsynch03_6.golden testsynch03_6.log 11-7 "Compare 11-7" 1 0
callCompareTest testsynch03_7.golden testsynch03_7.log 11-8 "Compare 11-8" 1 0
callCompareTest testsynch03_8.golden testsynch03_8.log 11-9 "Compare 11-9" 1 0
callCompareTest testsynch03_9.golden testsynch03_9.log 11-10 "Compare 11-10" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux silent=true sync_mode=basepad sync_option=0:33333333 ! multifilesink location=testsynch04_%1d.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)20/1\" ! pngdec ! tensor_converter ! mux.sink_1" 12 0 0 $PERFORMANCE

callCompareTest testsynch04_0.golden testsynch04_0.log 12-1 "Compare 12-1" 1 0
callCompareTest testsynch04_1.golden testsynch04_1.log 12-2 "Compare 12-2" 1 0
callCompareTest testsynch04_2.golden testsynch04_2.log 12-3 "Compare 12-3" 1 0
callCompareTest testsynch04_3.golden testsynch04_3.log 12-4 "Compare 12-4" 1 0
callCompareTest testsynch04_4.golden testsynch04_4.log 12-5 "Compare 12-5" 1 0
callCompareTest testsynch04_5.golden testsynch04_5.log 12-6 "Compare 12-6" 1 0
callCompareTest testsynch04_6.golden testsynch04_6.log 12-7 "Compare 12-7" 1 0
callCompareTest testsynch04_7.golden testsynch04_7.log 12-8 "Compare 12-8" 1 0
callCompareTest testsynch04_8.golden testsynch04_8.log 12-9 "Compare 12-9" 1 0
callCompareTest testsynch04_9.golden testsynch04_9.log 12-10 "Compare 12-10" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN}  tensor_mux name=mux silent=true sync_mode=basepad sync_option=0:33333333 ! multifilesink location=testsynch05_%1d.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)20/1\" ! pngdec ! tensor_converter ! mux.sink_1 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)10/1\" ! pngdec ! tensor_converter ! mux.sink_2" 13 0 0 $PERFORMANCE

callCompareTest testsynch05_0.golden testsynch05_0.log 13-1 "Compare 13-1" 1 0
callCompareTest testsynch05_1.golden testsynch05_1.log 13-2 "Compare 13-2" 1 0
callCompareTest testsynch05_2.golden testsynch05_2.log 13-3 "Compare 13-3" 1 0
callCompareTest testsynch05_3.golden testsynch05_3.log 13-4 "Compare 13-4" 1 0
callCompareTest testsynch05_4.golden testsynch05_4.log 13-5 "Compare 13-5" 1 0
callCompareTest testsynch05_5.golden testsynch05_5.log 13-6 "Compare 13-6" 1 0
callCompareTest testsynch05_6.golden testsynch05_6.log 13-7 "Compare 13-7" 1 0
callCompareTest testsynch05_7.golden testsynch05_7.log 13-8 "Compare 13-8" 1 0
callCompareTest testsynch05_8.golden testsynch05_8.log 13-9 "Compare 13-9" 1 0
callCompareTest testsynch05_9.golden testsynch05_9.log 13-10 "Compare 13-10" 1 0

report

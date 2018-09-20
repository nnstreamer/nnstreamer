#!/usr/bin/env bash
source ../testAPI.sh

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

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_mux:5 tensor_mux name=mux ! filesink location=testcase01_RGB_100x100.log filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0" 1

compareAllSizeLimit testcase01_RGB_100x100.golden testcase01_RGB_100x100.log 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_mux:5 tensor_mux name=mux ! filesink location=testcase02_RGB_100x100.log filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0 filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_1" 2

compareAllSizeLimit testcase02_RGB_100x100.golden testcase02_RGB_100x100.log 2

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_mux:5 tensor_mux name=mux ! filesink location=testcase03_RGB_100x100.log filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0 filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_1 filesrc location=testcase02_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_2" 3

compareAllSizeLimit testcase03_RGB_100x100.golden testcase03_RGB_100x100.log 3

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_mux:5 tensor_mux name=mux ! filesink location=testcase01.log multifilesrc location=\"testsequence01_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0" 4

compareAllSizeLimit testcase01.golden testcase01.log 4

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_mux:5 tensor_mux name=mux ! filesink location=testcase02.log multifilesrc location=\"testsequence02_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence02_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_1" 5

compareAllSizeLimit testcase02.golden testcase02.log 5

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_mux:5 tensor_mux name=mux ! filesink location=testcase03.log multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_1 multifilesrc location=\"testsequence03_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_2" 6

compareAllSizeLimit testcase03.golden testcase03.log 6

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} --gst-debug=tensor_mux:5 tensor_mux name=mux ! filesink location=testcase04.log multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_1 multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_2 multifilesrc location=\"testsequence04_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_3" 7

compareAllSizeLimit testcase03.golden testcase03.log 7

report

#!/usr/bin/env bash
source ../testAPI.sh

if [ "$SKIPGEN" == "YES" ]
then
  echo "Test Case Generation Skipped"
  sopath=$2
else
  echo "Test Case Generation Started"
  python ../nnstreamer_converter/generateGoldenTestResult.py 10
  sopath=$1
fi
convertBMP2PNG

gstTest "--gst-plugin-path=../../build/gst --gst-debug=tensordemux:5 tensormux name=mux ! tensordemux name=demux filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0 demux.src_0 !queue! filesink location=demux00.log" 1

compareAllSizeLimit testcase.golden demux00.log 1

gstTest "--gst-plugin-path=../../build/gst --gst-debug=tensordemux:5 tensormux name=mux ! tensordemux name=demux filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_1 demux.src_0 ! queue ! filesink location=demux02_0.log demux.src_1 ! queue ! filesink location=demux02_1.log" 2

compareAllSizeLimit testcase.golden demux02_0.log 2_0
compareAllSizeLimit testcase.golden demux02_1.log 2_1

gstTest "--gst-plugin-path=../../build/gst --gst-debug=tensordemux:5 tensormux name=mux ! tensordemux name=demux filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_0 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_1 filesrc location=testcase_RGB_100x100.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=100,height=100,framerate=0/1  ! tensor_converter ! mux.sink_2 demux.src_0 ! queue ! filesink location=demux03_0.log demux.src_1 ! queue ! filesink location=demux03_1.log demux.src_2 ! queue ! filesink location=demux03_2.log" 3

compareAllSizeLimit testcase.golden demux03_0.log 3_0
compareAllSizeLimit testcase.golden demux03_1.log 3_1
compareAllSizeLimit testcase.golden demux03_2.log 3_2


gstTest "--gst-plugin-path=../../build/gst --gst-debug=tensordemux:5 tensormux name=mux ! tensordemux name=demux multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 demux.src_0 ! queue ! filesink location=demux04.log" 4

compareAllSizeLimit testcase_stream.golden demux04.log 4

gstTest "--gst-plugin-path=../../build/gst --gst-debug=tensordemux:5 tensormux name=mux ! tensordemux name=demux multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_1 demux.src_0 ! queue ! filesink location=demux05_0.log demux.src_1 ! queue ! filesink location=demux05_1.log" 5

compareAllSizeLimit testcase_stream.golden demux05_0.log 5_0
compareAllSizeLimit testcase_stream.golden demux05_1.log 5_1

gstTest "--gst-plugin-path=../../build/gst --gst-debug=tensordemux:5 tensormux name=mux ! tensordemux name=demux multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_1 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_2 demux.src_0 ! queue ! filesink location=demux06_0.log demux.src_1 ! queue ! filesink location=demux06_1.log demux.src_2 ! queue ! filesink location=demux06_2.log" 6

compareAllSizeLimit testcase_stream.golden demux06_0.log 6_0
compareAllSizeLimit testcase_stream.golden demux06_1.log 6_1
compareAllSizeLimit testcase_stream.golden demux06_2.log 6_2

gstTest "--gst-plugin-path=../../build/gst --gst-debug=tensordemux:5 tensormux name=mux ! tensordemux name=demux multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_0 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_1 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_2 multifilesrc location=\"testsequence_%1d.png\" index=0 caps=\"image/png, framerate=(fraction)30/1\" ! pngdec ! tensor_converter ! mux.sink_3 demux.src_0 ! queue ! filesink location=demux07_0.log demux.src_1 ! queue ! filesink location=demux07_1.log demux.src_2 ! queue ! filesink location=demux07_2.log demux.src_3 ! queue ! filesink location=demux07_3.log" 7

compareAllSizeLimit testcase_stream.golden demux07_0.log 7_0
compareAllSizeLimit testcase_stream.golden demux07_1.log 7_1
compareAllSizeLimit testcase_stream.golden demux07_2.log 7_2
compareAllSizeLimit testcase_stream.golden demux07_3.log 7_3

report

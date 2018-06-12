#!/usr/bin/env bash
source ../testAPI.sh

gstTest "videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! filesink location=\"testcase02.apitest.log\" sync=true" 0

PATH_TO_MODEL="../../build/nnstreamer_example/custom_example_passthrough/libnnstreamer_customfilter_passthrough.so"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter silent=TRUE ! tee name=t ! queue ! tensor_filter debug=FALSE framework=\"custom\" model=\"${PATH_TO_MODEL}\" input=\"3:280:40\" inputtype=\"uint8\" output=\"3:280:40\" outputtype=\"uint8\" ! filesink location=\"testcase01.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase01.direct.log\" sync=true" 1

compareAll testcase01.direct.log testcase01.passthrough.log 1

report

#!/usr/bin/env bash
source ../testAPI.sh

gstTest "videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! filesink location=\"testcase02.apitest.log\" sync=true" 0

PATH_TO_MODEL="../../build/nnstreamer_example/custom_example_passthrough/libnnstreamer_customfilter_passthrough.so"
PATH_TO_MODEL_V="../../build/nnstreamer_example/custom_example_passthrough/libnnstreamer_customfilter_passthrough_variable.so"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL}\" input=\"3:280:40\" inputtype=\"uint8\" output=\"3:280:40\" outputtype=\"uint8\" ! filesink location=\"testcase01.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase01.direct.log\" sync=true" 1

compareAll testcase01.direct.log testcase01.passthrough.log 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL}\" ! filesink location=\"testcase02.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase02.direct.log\" sync=true" 2

compareAll testcase02.direct.log testcase02.passthrough.log 2


gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL_V}\" input=\"3:640:480\" inputtype=\"uint8\" output=\"3:640:480\" outputtype=\"uint8\" ! filesink location=\"testcase03.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase03.direct.log\" sync=true" 3

compareAll testcase03.direct.log testcase03.passthrough.log 3

## @TODO there is a known bug that breaks case 4.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=640,height=480,framerate=0/1 ! videoconvert ! video/x-raw, format=RGB ! tensor_converter ! tee name=t ! queue ! tensor_filter framework=\"custom\" model=\"${PATH_TO_MODEL_V}\" ! filesink location=\"testcase04.passthrough.log\" sync=true t. ! queue ! filesink location=\"testcase04.direct.log\" sync=true" 4

compareAll testcase04.direct.log testcase04.passthrough.log 4

report

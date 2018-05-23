#!/usr/bin/env bash
if [ $# -eq 0 ]
then
  PATH_TO_PLUGIN="$PWD/../../build"
else
  PATH_TO_PLUGIN="$1"
fi

failed=0

# Generate one frame only (num-buffers=1)

gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=1 ! video/x-raw,format=RGB,width=280,height=40,framerate=0/1 ! tee name=t ! queue ! videoconvert ! video/x-raw, format=BGRx ! tensor_converter ! filesink location="test.bgrx.log" sync=true t. ! queue ! filesink location="test.rgb.log" sync=true || failed=1

# Check the results (test.rgb.log, test.bgrx.log)
cmp -n `stat --printf="%s" testcase01.bgrx.golden` test.bgrx.log testcase01.bgrx.golden || failed=1
cmp -n `stat --printf="%s" testcase01.rgb.golden` test.rgb.log testcase01.rgb.golden  ||  failed=1

if [ "$failed" -eq "0" ]
then
  echo Testcase 01: SUCCESS
  exit 0
else
  echo Testcase 01: FAILED
  exit -1
fi

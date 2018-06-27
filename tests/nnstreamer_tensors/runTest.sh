#!/usr/bin/env bash
source ../testAPI.sh

if [ "$SKIPGEN" == "YES" ]
then
  echo "Test Case Generation Skipped"
  sopath=$2
else
  echo "Test Case Generation Started"
  python generateGoldenTestResult.py
  sopath=$1
fi

gstTest "--gst-plugin-path=../../build/tests/nnstreamer_tensors/tensors_test filesrc location=testcase01_RGB_642x480.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,width=642,height=480,framerate=0/1 ! testtensors silent=TRUE ! tensorscheck ! filesink location=\"testcase01_RGB_642x480.nonip.log\" sync=true"

report

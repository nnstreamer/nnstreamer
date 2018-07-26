#!/usr/bin/env bash
source ../testAPI.sh

# Test constant passthrough custom filter (1, 2)
PATH_TO_MODEL="models/mobilenet_v1_1.0_224_quant.tflite"
PATH_TO_IMAGE="img/orange.png"
PATH_TO_LABEL="labels/labels.txt"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=\"${PATH_TO_IMAGE}\" ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw, format=RGB, framerate=0/1 ! tensor_converter ! tensor_filter framework=\"tensorflow-lite\" model=\"${PATH_TO_MODEL}\" ! filesink location=\"tensorfilter.out.log\" " 1
python checkLabel.py tensorfilter.out.log ${PATH_TO_LABEL} ${PATH_TO_IMAGE}
casereport 1 $? "Golden test comparison"

report

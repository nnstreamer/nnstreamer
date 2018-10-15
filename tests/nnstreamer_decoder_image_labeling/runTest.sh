#!/usr/bin/env bash
source ../testAPI.sh

# Test constant passthrough decoder (1, 2)
PATH_TO_MODEL="../test_models/models/mobilenet_v1_1.0_224_quant.tflite"
PATH_TO_LABEL="../test_models/labels/labels.txt"
PATH_TO_IMAGE="img/orange.png"
PATH_TO_FILE="tensordecoder.txt"

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=\"${PATH_TO_IMAGE}\" ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw, format=RGB, framerate=0/1 ! tensor_converter ! tensor_filter framework=\"tensorflow-lite\" model=\"${PATH_TO_MODEL}\" ! tensor_decoder output-type=2 mode=image_labeling mode-option-1=\"${PATH_TO_LABEL}\" ! filesink location=\"${PATH_TO_FILE}\""
label=$(cat "${PATH_TO_FILE}")
if [ $label = "orange" ]
then
casereport 1 $? "Golden test comparison"
report
fi

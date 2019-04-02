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

# Test constant passthrough decoder (1, 2)
PATH_TO_PLUGIN="../../build"
CASESTART=0
CASEEND=1

# tflite case: 4:1:1917:1/f32, 91:1917:1:1/f32 --> 4:160:120

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_mux name=mux ! tensor_decoder mode=bounding_boxes option1=tflite-ssd option2=coco_labels_list.txt option3=box_priors.txt option4=160:120 option5=300:300 ! videoconvert ! video/x-raw,format=BGRx ! multifilesink location=testtflitessd_output.%d  multifilesrc name=fs1 location=testtflitessd_tensors.0.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=4:1:1917 input-type=float32 ! mux.sink_0  multifilesrc name=fs2 location=testtflitessd_tensors.1.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=91:1917 input-type=float32 ! mux.sink_1  " 0 0 0 $PERFORMANCE

callCompareTest testtflitessd_golden.0 testtflitessd_output.0 0-1 "TFLITESSD Decode 1" 0
callCompareTest testtflitessd_golden.1 testtflitessd_output.1 0-2 "TFLITESSD Decode 2" 0

# tf case: 1:1:1:1, 100:1:1:1, 100:1:1:1, 4:100:1:1 --> 4:160:120

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_mux name=mux ! tensor_decoder mode=bounding_boxes option1=tf-ssd option2=coco_labels_list.txt option4=160:120 option5=640:480 ! videoconvert ! video/x-raw,format=BGRx ! multifilesink location=testtfssd_output.%d  multifilesrc name=fs1 location=testtfssd_tensors.0.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=1:1 input-type=float32 ! mux.sink_0  multifilesrc name=fs2 location=testtfssd_tensors.1.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=100:1 input-type=float32 ! mux.sink_1  multifilesrc name=fs3 location=testtfssd_tensors.2.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=100:1 input-type=float32 ! mux.sink_2  multifilesrc name=fs4 location=testtfssd_tensors.3.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=4:100 input-type=float32 ! mux.sink_3 " 1 0 0 $PERFORMANCE

callCompareTest testtfssd_golden.0 testtfssd_output.0 0-1 "TFSSD Decode 1" 0
callCompareTest testtfssd_golden.1 testtfssd_output.1 0-2 "TFSSD Decode 2" 0

report

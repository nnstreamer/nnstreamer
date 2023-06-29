#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author MyungJoo Ham <myungjoo.ham@gmail.com>
## @date Nov 01 2018
## @brief SSAT Test Cases for NNStreamer
##
if [[ "$SSATAPILOADED" != "1" ]]; then
    SILENT=0
    INDEPENDENT=1
    search="ssat-api.sh"
    source $search
    printf "${Blue}Independent Mode${NC}"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

# Test constant passthrough decoder (1, 2)
PATH_TO_PLUGIN="../../build"
CASESTART=0
CASEEND=1

# mobilenet-ssd & tflite-ssd(deprecated) case: 4:1:1917:1/f32, 91:1917:1/f32 --> 4:160:120:1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_mux name=mux ! tensor_decoder mode=bounding_boxes option1=mobilenet-ssd option2=coco_labels_list.txt option3=box_priors.txt option4=160:120 option5=300:300 ! videoconvert ! video/x-raw,format=BGRx ! multifilesink location=mobilenetssd_output.%d  multifilesrc name=fs1 location=mobilenetssd_tensors.0.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=4:1:1917:1 input-type=float32 ! mux.sink_0  multifilesrc name=fs2 location=mobilenetssd_tensors.1.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=91:1917:1 input-type=float32 ! mux.sink_1  " 0 0 0 $PERFORMANCE

callCompareTest mobilenetssd_golden.0 mobilenetssd_output.0 0-1 "mobilenet-ssd Decode 1" 0
callCompareTest mobilenetssd_golden.1 mobilenetssd_output.1 0-2 "mobilenet-ssd Decode 2" 0
rm mobilenetssd_output.*

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_mux name=mux ! tensor_decoder option1=mobilenet-ssd config-file=config_file.0 ! videoconvert ! video/x-raw,format=BGRx ! multifilesink location=mobilenetssd_output.%d  multifilesrc name=fs1 location=mobilenetssd_tensors.0.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=4:1:1917:1 input-type=float32 ! mux.sink_0  multifilesrc name=fs2 location=mobilenetssd_tensors.1.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=91:1917:1 input-type=float32 ! mux.sink_1  " 0 0 0 $PERFORMANCE

callCompareTest mobilenetssd_golden.0 mobilenetssd_output.0 0-1 "mobilenet-ssd Decode 1 (with config_file.0)" 0
callCompareTest mobilenetssd_golden.1 mobilenetssd_output.1 0-2 "mobilenet-ssd Decode 2 (with config_file.0)" 0
rm mobilenetssd_output.*

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_mux name=mux ! tensor_decoder mode=bounding_boxes option1=tflite-ssd option2=coco_labels_list.txt option3=box_priors.txt option4=160:120 option5=300:300 ! videoconvert ! video/x-raw,format=BGRx ! multifilesink location=tflitessd_output.%d  multifilesrc name=fs1 location=mobilenetssd_tensors.0.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=4:1:1917:1 input-type=float32 ! mux.sink_0  multifilesrc name=fs2 location=mobilenetssd_tensors.1.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=91:1917:1 input-type=float32 ! mux.sink_1  " 0 0 0 $PERFORMANCE

callCompareTest mobilenetssd_golden.0 tflitessd_output.0 0-1 "tflite-ssd(deprecated) Decode 1" 0
callCompareTest mobilenetssd_golden.1 tflitessd_output.1 0-2 "tflite-ssd(deprecated) Decode 2" 0
rm tflitessd_output.*

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_mux name=mux ! tensor_decoder option1=tflite-ssd config-file=config_file.0 ! videoconvert ! video/x-raw,format=BGRx ! multifilesink location=tflitessd_output.%d  multifilesrc name=fs1 location=mobilenetssd_tensors.0.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=4:1:1917:1 input-type=float32 ! mux.sink_0  multifilesrc name=fs2 location=mobilenetssd_tensors.1.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=91:1917:1 input-type=float32 ! mux.sink_1  " 0 0 0 $PERFORMANCE

callCompareTest mobilenetssd_golden.0 tflitessd_output.0 0-1 "tflite-ssd(deprecated) Decode 1 (with config_file.0)" 0
callCompareTest mobilenetssd_golden.1 tflitessd_output.1 0-2 "tflite-ssd(deprecated) Decode 2 (with config_file.0)" 0
rm tflitessd_output.*

# mobilenet-ssd-post-process & tf-ssd(deprecated) case: 1, 100:1, 100:1, 4:100:1 --> 4:160:120:1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_mux name=mux ! tensor_decoder mode=bounding_boxes option1=mobilenet-ssd-postprocess option2=coco_labels_list.txt option4=160:120 option5=640:480 ! videoconvert ! video/x-raw,format=BGRx ! multifilesink location=mobilenetssd_postprocess_output.%d  multifilesrc name=fs1 location=mobilenetssd_postprocess_tensors.0.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=1 input-type=float32 ! mux.sink_0  multifilesrc name=fs2 location=mobilenetssd_postprocess_tensors.1.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=100:1 input-type=float32 ! mux.sink_1  multifilesrc name=fs3 location=mobilenetssd_postprocess_tensors.2.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=100:1 input-type=float32 ! mux.sink_2  multifilesrc name=fs4 location=mobilenetssd_postprocess_tensors.3.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=4:100:1 input-type=float32 ! mux.sink_3 " 1 0 0 $PERFORMANCE

callCompareTest mobilenetssd_postprocess_golden.0 mobilenetssd_postprocess_output.0 0-1 "mobilenet-ssd-postprocess Decode 1" 0
callCompareTest mobilenetssd_postprocess_golden.1 mobilenetssd_postprocess_output.1 0-2 "mobilenet-ssd-postprocess Decode 2" 0
rm mobilenetssd_postprocess_output.*

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_mux name=mux ! tensor_decoder option1=mobilenet-ssd-postprocess config-file=config_file.1 ! videoconvert ! video/x-raw,format=BGRx ! multifilesink location=mobilenetssd_postprocess_output.%d  multifilesrc name=fs1 location=mobilenetssd_postprocess_tensors.0.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=1 input-type=float32 ! mux.sink_0  multifilesrc name=fs2 location=mobilenetssd_postprocess_tensors.1.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=100:1 input-type=float32 ! mux.sink_1  multifilesrc name=fs3 location=mobilenetssd_postprocess_tensors.2.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=100:1 input-type=float32 ! mux.sink_2  multifilesrc name=fs4 location=mobilenetssd_postprocess_tensors.3.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=4:100:1 input-type=float32 ! mux.sink_3 " 1 0 0 $PERFORMANCE

callCompareTest mobilenetssd_postprocess_golden.0 mobilenetssd_postprocess_output.0 0-1 "mobilenet-ssd-postprocess Decode 1 (with config_file.1)" 0
callCompareTest mobilenetssd_postprocess_golden.1 mobilenetssd_postprocess_output.1 0-2 "mobilenet-ssd-postprocess Decode 2 (with config_file.1)" 0
rm mobilenetssd_postprocess_output.*

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_mux name=mux ! tensor_decoder mode=bounding_boxes option1=tf-ssd option2=coco_labels_list.txt option4=160:120 option5=640:480 ! videoconvert ! video/x-raw,format=BGRx ! multifilesink location=tfssd_postprocess_output.%d  multifilesrc name=fs1 location=mobilenetssd_postprocess_tensors.0.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=1 input-type=float32 ! mux.sink_0  multifilesrc name=fs2 location=mobilenetssd_postprocess_tensors.1.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=100:1 input-type=float32 ! mux.sink_1  multifilesrc name=fs3 location=mobilenetssd_postprocess_tensors.2.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=100:1 input-type=float32 ! mux.sink_2  multifilesrc name=fs4 location=mobilenetssd_postprocess_tensors.3.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=4:100:1 input-type=float32 ! mux.sink_3 " 1 0 0 $PERFORMANCE

callCompareTest mobilenetssd_postprocess_golden.0 tfssd_postprocess_output.0 0-1 "tf-ssd(deprecated) Decode 1" 0
callCompareTest mobilenetssd_postprocess_golden.1 tfssd_postprocess_output.1 0-2 "tf-ssd(deprecated) Decode 2" 0
rm tfssd_postprocess_output.*

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_mux name=mux ! tensor_decoder option1=tf-ssd config-file=config_file.1 ! videoconvert ! video/x-raw,format=BGRx ! multifilesink location=tfssd_postprocess_output.%d  multifilesrc name=fs1 location=mobilenetssd_postprocess_tensors.0.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=1 input-type=float32 ! mux.sink_0  multifilesrc name=fs2 location=mobilenetssd_postprocess_tensors.1.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=100:1 input-type=float32 ! mux.sink_1  multifilesrc name=fs3 location=mobilenetssd_postprocess_tensors.2.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=100:1 input-type=float32 ! mux.sink_2  multifilesrc name=fs4 location=mobilenetssd_postprocess_tensors.3.%d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=4:100:1 input-type=float32 ! mux.sink_3 " 1 0 0 $PERFORMANCE

callCompareTest mobilenetssd_postprocess_golden.0 tfssd_postprocess_output.0 0-1 "tf-ssd(deprecated) Decode 1 (with config_file.1)" 0
callCompareTest mobilenetssd_postprocess_golden.1 tfssd_postprocess_output.1 0-2 "tf-ssd(deprecated) Decode 2 (with config_file.1)" 0
rm tfssd_postprocess_output.*

# palm detection decoder test
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_mux name=mux ! tensor_decoder mode=bounding_boxes option1=mp-palm-detection option3=0.5:4:1.0:1.0:0.5:0.5:8:16:16:16 option4=160:120 option5=300:300 ! videoconvert !  video/x-raw,format=RGBA ! multifilesink location=palm_detection_result_%1d.log \
    multifilesrc location=palm_detection_input_0.%1d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=18:2016:1:1 input-type=float32 ! mux.sink_0 \
    multifilesrc location=palm_detection_input_1.%1d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=1:2016:1:1 input-type=float32 ! mux.sink_1" 5 0 0 $PERFORMANCE

callCompareTest palm_detection_result_golden.0 palm_detection_result_0.log 5-0 "palm detection Decode 0" 0
callCompareTest palm_detection_result_golden.1 palm_detection_result_1.log 5-1 "palm detection Decode 1" 0
rm palm_detection_result_*.log

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_mux name=mux ! tensor_decoder config-file=config_file.2 ! videoconvert !  video/x-raw,format=RGBA ! multifilesink location=palm_detection_result_%1d.log \
    multifilesrc location=palm_detection_input_0.%1d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=18:2016:1:1 input-type=float32 ! mux.sink_0 \
    multifilesrc location=palm_detection_input_1.%1d start-index=$CASESTART stop-index=$CASEEND caps=application/octet-stream ! tensor_converter input-dim=1:2016:1:1 input-type=float32 ! mux.sink_1" 5 0 0 $PERFORMANCE

callCompareTest palm_detection_result_golden.0 palm_detection_result_0.log 5-0 "palm detection Decode 0 (with config_file.2)" 0
callCompareTest palm_detection_result_golden.1 palm_detection_result_1.log 5-1 "palm detection Decode 1 (with config_file.2)" 0
rm palm_detection_result_*.log

report

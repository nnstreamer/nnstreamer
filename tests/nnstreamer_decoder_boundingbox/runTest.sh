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

# yolov5 decoder test
## wrong tensor dimension
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=yolov5_decoder_input.raw start-index=0 stop-index=0 caps=application/octet-stream ! tensor_converter input-dim=85:10647:1 input-type=float32 ! tensor_decoder mode=bounding_boxes option1=yolov5 option2=coco-80.txt option3=0:0.25:0.45 option4=320:320 option5=320:320 option6=0 option7=1 ! fakesink" "6 yolov5 decoder_n" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=yolov5_decoder_input.raw start-index=0 stop-index=0 caps=application/octet-stream ! tensor_converter input-dim=85:6300:1 input-type=float32 ! tensor_decoder mode=bounding_boxes option1=yolov5 option2=coco-80.txt option3=0:0.25:0.45 option4=320:320 option5=320:320 option6=0 option7=1 ! videoconvert ! video/x-raw,format=RGBA ! multifilesink location=yolov5_result_%1d.log" "6 yolov5 decoder" 0 0

callCompareTest yolov5_result_golden.raw yolov5_result_0.log "6 diff" "yolov5 golden" 0

# test track mode
## wrong tensor dimension
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=yolov5_decoder_input.raw start-index=0 stop-index=0 caps=application/octet-stream ! tensor_converter input-dim=85:10647:1 input-type=float32 ! tensor_decoder mode=bounding_boxes option1=yolov5 option2=coco-80.txt option3=0:0.25:0.45 option4=320:320 option5=320:320 option6=1 option7=1 ! fakesink" "7 yolov5 decoder_n" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=yolov5_decoder_input.raw start-index=0 stop-index=2 caps=application/octet-stream ! tensor_converter input-dim=85:6300:1 input-type=float32 ! tensor_decoder mode=bounding_boxes option1=yolov5 option2=coco-80.txt option3=0:0.25:0.45 option4=320:320 option5=320:320 option6=1 option7=1 ! videoconvert ! video/x-raw,format=RGBA ! multifilesink location=yolov5_track_result_%1d.log" "7 yolov5 decoder with track mode" 0 0

callCompareTest yolov5_track_result_golden.raw yolov5_track_result_0.log "7 diff" "yolov5 with track mode golden" 0
callCompareTest yolov5_track_result_golden.raw yolov5_track_result_1.log "7 diff" "yolov5 with track mode golden" 0
callCompareTest yolov5_track_result_golden.raw yolov5_track_result_2.log "7 diff" "yolov5 with track mode golden" 0

# yolov8 decoder test
## wrong tensor dimension
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=yolov8_decoder_input.raw caps=application/octet-stream start-index=0 stop-index=0 ! tensor_converter input-dim=84:8400:1 input-type=float32 ! tensor_decoder mode=bounding_boxes option1=yolov8 option2=coco-80.txt option3=0:0.25:0.45 option4=320:320 option5=320:320 option6=0 option7=1 ! fakesink" "8 yolov8 decoder_n" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=yolov8_decoder_input.raw caps=application/octet-stream start-index=0 stop-index=0 ! tensor_converter input-dim=84:2100:1 input-type=float32 ! tensor_decoder mode=bounding_boxes option1=yolov8 option2=coco-80.txt option3=0:0.25:0.45 option4=320:320 option5=320:320 option6=0 option7=1 ! videoconvert ! video/x-raw,format=RGBA ! multifilesink location=yolov8_result_%1d.log" "8 yolov8 decoder" 0 0

callCompareTest yolov8_result_golden.raw yolov8_result_0.log "8 diff" "yolov8 golden" 0

# yolov10 decoder test
## wrong tensor dimension
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=yolov10_decoder_input.raw caps=application/octet-stream start-index=0 stop-index=0 ! tensor_converter input-dim=4:300:1 input-type=float32 ! tensor_decoder mode=bounding_boxes option1=yolov10 option2=coco-80.txt option3=0:0.25:0.45 option4=320:320 option5=320:320 option6=0 option7=1 ! fakesink" "9 yolov10 decoder dim_n" 0 1

## wrong tensor type
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=yolov10_decoder_input.raw caps=application/octet-stream start-index=0 stop-index=0 ! tensor_converter input-dim=6:300:1 input-type=float32 ! tensor_transform mode=typecast option=int32 ! tensor_decoder mode=bounding_boxes option1=yolov10 option2=coco-80.txt option3=0:0.25:0.45 option4=320:320 option5=320:320 ! fakesink" "9 yolov10 decoder type_n" 0 1

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=yolov10_decoder_input.raw caps=application/octet-stream start-index=0 stop-index=0 ! tensor_converter input-dim=6:300:1 input-type=float32 ! tensor_decoder mode=bounding_boxes option1=yolov10 option2=coco-80.txt option3=0:0.25 option4=320:320 option5=320:320 option6=0 option7=1 ! videoconvert ! video/x-raw,format=RGBA ! multifilesink location=yolov10_result_%1d.log" "9 yolov10 decoder" 0 0

callCompareTest yolov10_result_golden.raw yolov10_result_0.log "9 diff" "yolov10 golden" 0

# yolov8-obb decoder test

## create label file
echo "plane
ship
storage tank
baseball diamond
tennis court
basketball court
ground track field
harbor
bridge
large vehicle
small vehicle
helicopter
roundabout
soccer ball field
swimming pool" > dota8-obb-label.txt

## wrong tensor dimension
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=yolov8_obb_decoder_input.raw caps=application/octet-stream num-buffers=1 ! tensor_converter input-dim=22:8400:1 input-type=float32 ! tensor_decoder mode=bounding_boxes option1=yolov8-obb option2=dota8-obb-label.txt option4=640:640 option5=640:640 ! fakesink" "10 yolov8-obb inputdim_n" 0 1
## wrong tensor type
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=yolov8_obb_decoder_input.raw caps=application/octet-stream num-buffers=1 ! tensor_converter input-dim=20:8400:1 input-type=uint8 ! tensor_decoder mode=bounding_boxes option1=yolov8-obb option2=dota8-obb-label.txt option4=640:640 option5=640:640 ! fakesink" "10 yolov8-obb inputtype_n" 0 1

## golden test
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=yolov8_obb_decoder_input.raw caps=application/octet-stream num-buffers=1 ! tensor_converter input-dim=20:8400:1 input-type=float32 ! tensor_decoder mode=bounding_boxes option1=yolov8-obb option2=dota8-obb-label.txt option3=0:0.25:0.45 option4=640:640 option5=640:640 ! video/x-raw,format=RGBA ! filesink location=yolo11n-obb_result.log" "10 yolov8-obb" 0 0

callCompareTest yolov8_obb_decoder_result_golden.raw yolo11n-obb_result.log "10 yolov8-obb diff" "diff with golden" 0

rm dota8-obb-label.txt

# negative case for box properties

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=10 ! video/x-raw,format=RGB,width=224,height=224,framerate=0/1 ! videoconvert ! tensor_converter ! tensor_decoder mode=bounding_boxes option1=wrong_mode_name ! fakesink " 11_n 0 1

rm yolov*.log

report

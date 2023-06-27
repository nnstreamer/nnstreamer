#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Harsh Jain <hjain24in@gmail.com>
## @date 18 JUNE, 2023
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

PATH_TO_PLUGIN="../../build"
PATH_TO_IMAGE="../test_models/data/orange.png"
PATH_TO_LABELS="../nnstreamer_decoder_boundingbox/coco_labels_list.txt"
PATH_TO_BOX_PRIORS="../nnstreamer_decoder_boundingbox/box_priors.txt"
PATH_TO_MODEL="../test_models/models/ssd_mobilenet_v2_coco.tflite"
CASESTART=0
CASEEND=1


gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
    filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoconvert ! videoscale ! video/x-raw,width=300,height=300,format=RGB,framerate=0/1 ! tensor_converter ! crop.raw \
    filesrc  location=mobilenet_ssd_tensor.0 blocksize=-1 ! application/octet-stream ! tensor_converter name=el1 input-dim=4:1:1917:1 input-type=float32 ! mux.sink_0 \
    filesrc  location=mobilenet_ssd_tensor.1 blocksize=-1 ! application/octet-stream ! tensor_converter name=el2 input-dim=91:1917:1 input-type=float32 ! mux.sink_1 \
    tensor_mux name=mux ! other/tensors,format=static ! tensor_decoder mode=tensor_region option1=1 option2=${PATH_TO_LABELS} option3=${PATH_TO_BOX_PRIORS} ! crop.info\
    tensor_crop name=crop ! other/tensors,format=flexible ! tensor_converter ! tensor_decoder mode=direct_video ! videoconvert ! video/x-raw,format=RGBx  !  filesink location=tensor_region_output_orange.txt   " 0 0 0 $PERFORMANCE

callCompareTest tensor_region_orange.txt tensor_region_output_orange.txt 0 "mobilenet-ssd Decode 1" 0
rm tensor_region_output_*
report

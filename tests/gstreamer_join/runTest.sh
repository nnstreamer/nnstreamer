#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Gichan Jang <gichan2.jang@samsung.com>
## @date Nov 11 2020
## @brief SSAT Test Cases for gstreamer element join
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

# Generate golden test results
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc name=vsrc num-buffers=1 pattern=13 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=160,height=120 ! filesink location=smpte.golden
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc name=vsrc num-buffers=1 pattern=15 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=160,height=120 ! filesink location=gamut.golden
# tensor stream test when GSTCAP is specified
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
        videotestsrc num-buffers=4 pattern=12 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! mux.sink_0 \
        videotestsrc pattern=13 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! mux.sink_1 \
        videotestsrc pattern=15 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! mux.sink_2 \
        tensor_mux name=mux ! tensor_if name=tif compared_value=TENSOR_AVERAGE_VALUE compared-value-option=0 supplied-value=100 operator=LT then=TENSORPICK then-option=1 else=TENSORPICK else-option=2 \
            tif.src_0 ! queue ! join.sink_0 \
            tif.src_1 ! queue ! join.sink_1 \
            join name=join ! multifilesink location=testJoin1_%1d.log sync=false async=false" 1 0 0 $PERFORMANCE
callCompareTest smpte.golden testJoin1_0.log 1-1 "Compare 1-1" 1 0
callCompareTest gamut.golden testJoin1_1.log 1-2 "Compare 1-2" 1 0
callCompareTest smpte.golden testJoin1_2.log 1-3 "Compare 1-3" 1 0
callCompareTest gamut.golden testJoin1_3.log 1-4 "Compare 1-4" 1 0

# Generate golden test results
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc name=vsrc num-buffers=1 pattern=13 ! tensor_converter ! filesink location=smpte.default.golden
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc name=vsrc num-buffers=1 pattern=15 ! tensor_converter ! filesink location=gamut.default.golden
# tensor stream test when GSTCAP is not specified (default CAP)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
        videotestsrc num-buffers=4 pattern=12 ! tensor_converter ! mux.sink_0 \
        videotestsrc pattern=13 ! tensor_converter ! mux.sink_1 \
        videotestsrc pattern=15 ! tensor_converter ! mux.sink_2 \
        tensor_mux name=mux ! tensor_if name=tif compared_value=TENSOR_AVERAGE_VALUE compared-value-option=0 supplied-value=100 operator=LT then=TENSORPICK then-option=1 else=TENSORPICK else-option=2 \
            tif.src_0 ! queue ! join.sink_0 \
            tif.src_1 ! queue ! join.sink_1 \
            join name=join ! multifilesink location=testJoin2_%1d.log sync=false async=false" 2 0 0 $PERFORMANCE
callCompareTest smpte.default.golden testJoin2_0.log 2-1 "Compare 2-1" 1 0
callCompareTest gamut.default.golden testJoin2_1.log 2-2 "Compare 2-2" 1 0
callCompareTest smpte.default.golden testJoin2_2.log 2-3 "Compare 2-3" 1 0
callCompareTest gamut.default.golden testJoin2_3.log 2-4 "Compare 2-4" 1 0

# Fail test: sink pads caps are not matched (dimension)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} \
        videotestsrc num-buffers=4 pattern=12 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! mux.sink_0 \
        videotestsrc pattern=13 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=160,height=120 ! tensor_converter ! mux.sink_1 \
        videotestsrc pattern=15 ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=480,height=240 ! tensor_converter ! mux.sink_2 \
        tensor_mux name=mux ! tensor_if name=tif compared_value=TENSOR_AVERAGE_VALUE compared-value-option=0 supplied-value=100 operator=LT then=TENSORPICK then-option=1 else=TENSORPICK else-option=2 \
            tif.src_0 ! queue ! join.sink_0 \
            tif.src_1 ! queue ! join.sink_1 \
            join name=join ! fakesink sync=true" 3_n 0 1 $PERFORMANCE

rm *.golden
rm *.log

report

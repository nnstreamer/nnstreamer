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

PATH_TO_PLUGIN="../../build"

if [ "$SKIPGEN" == "YES" ]; then
    echo "Test Case Generation Skipped"
    sopath=$2
else
    echo "Test Case Generation Started"
    python3 ../nnstreamer_converter/generateGoldenTestResult.py 12
    sopath=$1
fi
convertBMP2PNG

# Fail Test : Negotiation Error (dimension)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=testsequence_%1d.png index=0 caps=\"image/png,framerate=(fraction)3/1\" ! pngdec ! tensor_converter ! queue ! tensor_reposink silent=false slot-index=0 tensor_reposrc silent=false slot-index=0 caps=\"other/tensor,dimension=(string)3:100:100:1,type=(string)uint8,framerate=(fraction)3/1\" ! multifilesink location=testsequence01_%1d.log" 0F_n 0 1 $PERFORMANCE

# Fail Test : Negotiation Error (type)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=testsequence_%1d.png index=0 caps=\"image/png,framerate=(fraction)3/1\" ! pngdec ! tensor_converter ! queue ! tensor_reposink silent=false slot-index=0 tensor_reposrc silent=false slot-index=0 caps=\"other/tensor,dimension=(string)3:16:16:1,type=(string)float32,framerate=(fraction)3/1\" ! multifilesink location=testsequence01_%1d.log" 1F_n 0 1 $PERFORMANCE

# Fail Test : Negotiation Error (MimeType)
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=testsequence_%1d.png index=0 caps=\"image/png,framerate=(fraction)3/1\" ! pngdec ! tensor_converter ! queue ! tensor_reposink silent=false slot-index=0 tensor_reposrc silent=false slot-index=0 caps=\"other/tensors,num_tensors=1,framerate=(fraction)3/1,types=(string)uint8,dimensions=(string)3:16:16:1\" ! multifilesink location=testsequence01_%1d.log" 2F_n 0 1 $PERFORMANCE

# The first gst buffer at tensor_reposrc is dummy.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=testsequence_%1d.png index=0 caps=\"image/png,framerate=(fraction)3/1\" ! pngdec ! tensor_converter ! queue ! tensor_reposink silent=false slot-index=0 tensor_reposrc silent=false slot-index=0 caps=\"other/tensor,dimension=(string)3:16:16:1,type=(string)uint8,framerate=(fraction)3/1\" ! multifilesink location=testsequence01_%1d.log" 1 0 0 $PERFORMANCE

callCompareTest testsequence_1.golden testsequence01_1.log 1-1 "Compare 1-1" 1 0
callCompareTest testsequence_2.golden testsequence01_2.log 1-2 "Compare 1-2" 1 0
callCompareTest testsequence_3.golden testsequence01_3.log 1-3 "Compare 1-3" 1 0
callCompareTest testsequence_4.golden testsequence01_4.log 1-4 "Compare 1-4" 1 0
callCompareTest testsequence_5.golden testsequence01_5.log 1-5 "Compare 1-5" 1 0
callCompareTest testsequence_6.golden testsequence01_6.log 1-6 "Compare 1-6" 1 0
callCompareTest testsequence_7.golden testsequence01_7.log 1-7 "Compare 1-7" 1 0
callCompareTest testsequence_8.golden testsequence01_8.log 1-8 "Compare 1-8" 1 0
callCompareTest testsequence_9.golden testsequence01_9.log 1-9 "Compare 1-9" 1 0
callCompareTest testsequence_10.golden testsequence01_10.log 1-10 "Compare 1-10" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=testsequence_%1d.png index=0 caps=\"image/png,framerate=(fraction)3/1\" ! pngdec ! tensor_converter ! queue ! tensor_reposink silent=false slot-index=0 tensor_reposrc silent=false slot-index=0 caps=\"other/tensor,dimension=(string)3:16:16:1,type=(string)uint8,framerate=(fraction)3/1\" ! multifilesink location=testsequence02_0_%1d.log multifilesrc location=testsequence_%1d.png index=0 caps=\"image/png,framerate=(fraction)3/1\" ! pngdec ! tensor_converter ! queue ! tensor_reposink silent=false slot-index=1 tensor_reposrc silent=false slot-index=1 caps=\"other/tensor,dimension=(string)3:16:16:1,type=(string)uint8,framerate=(fraction)3/1\" ! multifilesink location=testsequence02_1_%1d.log" 2 0 0 $PERFORMANCE

callCompareTest testsequence_1.golden testsequence02_0_1.log 2-1 "Compare 2-1" 1 0
callCompareTest testsequence_2.golden testsequence02_0_2.log 2-2 "Compare 2-2" 1 0
callCompareTest testsequence_3.golden testsequence02_0_3.log 2-3 "Compare 2-3" 1 0
callCompareTest testsequence_4.golden testsequence02_0_4.log 2-4 "Compare 2-4" 1 0
callCompareTest testsequence_5.golden testsequence02_0_5.log 2-5 "Compare 2-5" 1 0
callCompareTest testsequence_6.golden testsequence02_0_6.log 2-6 "Compare 2-6" 1 0
callCompareTest testsequence_7.golden testsequence02_0_7.log 2-7 "Compare 2-7" 1 0
callCompareTest testsequence_8.golden testsequence02_0_8.log 2-8 "Compare 2-8" 1 0
callCompareTest testsequence_9.golden testsequence02_0_9.log 2-9 "Compare 2-9" 1 0
callCompareTest testsequence_10.golden testsequence02_0_10.log 2-10 "Compare 2-10" 1 0

callCompareTest testsequence_1.golden testsequence02_1_1.log 2-11 "Compare 2-11" 1 0
callCompareTest testsequence_2.golden testsequence02_1_2.log 2-12 "Compare 2-12" 1 0
callCompareTest testsequence_3.golden testsequence02_1_3.log 2-13 "Compare 2-13" 1 0
callCompareTest testsequence_4.golden testsequence02_1_4.log 2-14 "Compare 2-14" 1 0
callCompareTest testsequence_5.golden testsequence02_1_5.log 2-15 "Compare 2-15" 1 0
callCompareTest testsequence_6.golden testsequence02_1_6.log 2-16 "Compare 2-16" 1 0
callCompareTest testsequence_7.golden testsequence02_1_7.log 2-17 "Compare 2-17" 1 0
callCompareTest testsequence_8.golden testsequence02_1_8.log 2-18 "Compare 2-18" 1 0
callCompareTest testsequence_9.golden testsequence02_1_9.log 2-19 "Compare 2-19" 1 0
callCompareTest testsequence_10.golden testsequence02_1_10.log 2-20 "Compare 2-20" 1 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=testsequence_%1d.png index=0 caps=\"image/png,framerate=(fraction)3/1\" ! pngdec ! tensor_converter ! queue ! tensor_reposink silent=false slot-index=0 tensor_reposrc silent=false slot-index=0 caps=\"other/tensor,dimension=(string)3:16:16:1,type=(string)uint8,framerate=(fraction)3/1\" ! multifilesink location=testsequence03_0_%1d.log multifilesrc location=testsequence_%1d.png index=0 caps=\"image/png,framerate=(fraction)3/1\" ! pngdec ! tensor_converter ! queue ! tensor_reposink silent=false slot-index=1 tensor_reposrc silent=false slot-index=1 caps=\"other/tensor,dimension=(string)3:16:16:1,type=(string)uint8,framerate=(fraction)3/1\" ! multifilesink location=testsequence03_1_%1d.log multifilesrc location=testsequence_%1d.png index=0 caps=\"image/png,framerate=(fraction)3/1\" ! pngdec ! tensor_converter ! queue ! tensor_reposink silent=false slot-index=2 tensor_reposrc silent=false slot-index=2 caps=\"other/tensor,dimension=(string)3:16:16:1,type=(string)uint8,framerate=(fraction)3/1\" ! multifilesink location=testsequence03_2_%1d.log" 3 0 0 $PERFORMANCE

callCompareTest testsequence_1.golden testsequence03_0_1.log 3-1 "Compare 3-1" 1 0
callCompareTest testsequence_2.golden testsequence03_0_2.log 3-2 "Compare 3-2" 1 0
callCompareTest testsequence_3.golden testsequence03_0_3.log 3-3 "Compare 3-3" 1 0
callCompareTest testsequence_4.golden testsequence03_0_4.log 3-4 "Compare 3-4" 1 0
callCompareTest testsequence_5.golden testsequence03_0_5.log 3-5 "Compare 3-5" 1 0
callCompareTest testsequence_6.golden testsequence03_0_6.log 3-6 "Compare 3-6" 1 0
callCompareTest testsequence_7.golden testsequence03_0_7.log 3-7 "Compare 3-7" 1 0
callCompareTest testsequence_8.golden testsequence03_0_8.log 3-8 "Compare 3-8" 1 0
callCompareTest testsequence_9.golden testsequence03_0_9.log 3-9 "Compare 3-9" 1 0
callCompareTest testsequence_10.golden testsequence03_0_10.log 3-10 "Compare 3-10" 1 0

callCompareTest testsequence_1.golden testsequence03_1_1.log 3-11 "Compare 3-11" 1 0
callCompareTest testsequence_2.golden testsequence03_1_2.log 3-12 "Compare 3-12" 1 0
callCompareTest testsequence_3.golden testsequence03_1_3.log 3-13 "Compare 3-13" 1 0
callCompareTest testsequence_4.golden testsequence03_1_4.log 3-14 "Compare 3-14" 1 0
callCompareTest testsequence_5.golden testsequence03_1_5.log 3-15 "Compare 3-15" 1 0
callCompareTest testsequence_6.golden testsequence03_1_6.log 3-16 "Compare 3-16" 1 0
callCompareTest testsequence_7.golden testsequence03_1_7.log 3-17 "Compare 3-17" 1 0
callCompareTest testsequence_8.golden testsequence03_1_8.log 3-18 "Compare 3-18" 1 0
callCompareTest testsequence_9.golden testsequence03_1_9.log 3-19 "Compare 3-19" 1 0
callCompareTest testsequence_10.golden testsequence03_1_10.log 3-20 "Compare 3-20" 1 0

callCompareTest testsequence_1.golden testsequence03_2_1.log 3-21 "Compare 3-21" 1 0
callCompareTest testsequence_2.golden testsequence03_2_2.log 3-22 "Compare 3-22" 1 0
callCompareTest testsequence_3.golden testsequence03_2_3.log 3-23 "Compare 3-23" 1 0
callCompareTest testsequence_4.golden testsequence03_2_4.log 3-24 "Compare 3-24" 1 0
callCompareTest testsequence_5.golden testsequence03_2_5.log 3-25 "Compare 3-25" 1 0
callCompareTest testsequence_6.golden testsequence03_2_6.log 3-26 "Compare 3-26" 1 0
callCompareTest testsequence_7.golden testsequence03_2_7.log 3-27 "Compare 3-27" 1 0
callCompareTest testsequence_8.golden testsequence03_2_8.log 3-28 "Compare 3-28" 1 0
callCompareTest testsequence_9.golden testsequence03_2_9.log 3-29 "Compare 3-29" 1 0
callCompareTest testsequence_10.golden testsequence03_2_10.log 3-30 "Compare 3-30" 1 0

# Dimension declaration test case
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} multifilesrc location=testsequence_%1d.png index=0 caps=\"image/png,framerate=(fraction)3/1\" ! pngdec ! tensor_converter ! queue ! tensor_reposink silent=false slot-index=0 tensor_reposrc silent=false slot-index=0 caps=\"other/tensor,dimension=(string)3:16:16,type=(string)uint8,framerate=(fraction)3/1\" ! multifilesink location=testsequence04_%1d.log" 4 0 0 $PERFORMANCE

callCompareTest testsequence_1.golden testsequence04_1.log 4-1 "Compare 4-1" 1 0
callCompareTest testsequence_2.golden testsequence04_2.log 4-2 "Compare 4-2" 1 0
callCompareTest testsequence_3.golden testsequence04_3.log 4-3 "Compare 4-3" 1 0
callCompareTest testsequence_4.golden testsequence04_4.log 4-4 "Compare 4-4" 1 0
callCompareTest testsequence_5.golden testsequence04_5.log 4-5 "Compare 4-5" 1 0
callCompareTest testsequence_6.golden testsequence04_6.log 4-6 "Compare 4-6" 1 0
callCompareTest testsequence_7.golden testsequence04_7.log 4-7 "Compare 4-7" 1 0
callCompareTest testsequence_8.golden testsequence04_8.log 4-8 "Compare 4-8" 1 0
callCompareTest testsequence_9.golden testsequence04_9.log 4-9 "Compare 4-9" 1 0
callCompareTest testsequence_10.golden testsequence04_10.log 4-10 "Compare 4-10" 1 0

rm *.log *.bmp *.png *.golden *.raw *.dat

report

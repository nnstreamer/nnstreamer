#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author MyungJoo Ham <myungjoo.ham@samsung.com>
## @date Nov 12 2026
## @brief SSAT Test Cases for tensor_if
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
    python3 generateTest.py || (echo "Failed to run test preparation script (generateTest.py). Test not available." && report && exit)
    sopath=$1
fi

# Six frames of two uint8 tensors, 2:2 and 3. See generateTest.py for the values.
SRC="multifilesrc location=\"input_%02d.dat\" caps=\"application/octet-stream\" ! tensor_converter input-dim=2:2,3 input-type=uint8,uint8"

# A_VALUE, PASSTHROUGH and SKIP
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:0,0 operator=GE supplied-value=10 then=PASSTHROUGH else=SKIP tif.src_0 ! filesink location=\"result01.log\"" 1 0 0 $PERFORMANCE
callCompareTest passthrough_ge.golden result01.log 1 "A_VALUE GE, PASSTHROUGH" 0 0

# The other comparison operators
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:0,0 operator=GT supplied-value=10 then=PASSTHROUGH else=SKIP tif.src_0 ! filesink location=\"result02.log\"" 2 0 0 $PERFORMANCE
callCompareTest passthrough_gt.golden result02.log 2 "A_VALUE GT, PASSTHROUGH" 0 0
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:0,0 operator=LE supplied-value=10 then=PASSTHROUGH else=SKIP tif.src_0 ! filesink location=\"result03.log\"" 3 0 0 $PERFORMANCE
callCompareTest passthrough_le.golden result03.log 3 "A_VALUE LE, PASSTHROUGH" 0 0
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:0,0 operator=EQ supplied-value=10 then=PASSTHROUGH else=SKIP tif.src_0 ! filesink location=\"result04.log\"" 4 0 0 $PERFORMANCE
callCompareTest passthrough_eq.golden result04.log 4 "A_VALUE EQ, PASSTHROUGH" 0 0
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:0,0 operator=NE supplied-value=10 then=PASSTHROUGH else=SKIP tif.src_0 ! filesink location=\"result05.log\"" 5 0 0 $PERFORMANCE
callCompareTest passthrough_ne.golden result05.log 5 "A_VALUE NE, PASSTHROUGH" 0 0

# Blanks around the tokens are accepted
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=\" 0 , 0\" operator=GE supplied-value=\" 10 \" then=PASSTHROUGH else=SKIP tif.src_0 ! filesink location=\"result06.log\"" 6 0 0 $PERFORMANCE
callCompareTest passthrough_ge.golden result06.log 6 "A_VALUE GE with blanks, PASSTHROUGH" 0 0

# A_VALUE of a non-first element, with the omitted dimensions filled with 0
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=1:1,0 operator=LT supplied-value=10 then=PASSTHROUGH else=SKIP tif.src_0 ! filesink location=\"result07.log\"" 7 0 0 $PERFORMANCE
callCompareTest passthrough_lt.golden result07.log 7 "A_VALUE LT of [1][1], PASSTHROUGH" 0 0

# TENSOR_AVERAGE_VALUE with an inclusive and an exclusive range, TENSORPICK
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=TENSOR_AVERAGE_VALUE compared-value-option=0 operator=RANGE_INCLUSIVE supplied-value=10,20 then=TENSORPICK then-option=1 else=SKIP tif.src_0 ! other/tensors,num_tensors=1 ! filesink location=\"result08.log\"" 8 0 0 $PERFORMANCE
callCompareTest pick_range_inclusive.golden result08.log 8 "TENSOR_AVERAGE_VALUE RANGE_INCLUSIVE, TENSORPICK" 0 0

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=TENSOR_AVERAGE_VALUE compared-value-option=0 operator=RANGE_EXCLUSIVE supplied-value=10,20 then=TENSORPICK then-option=1 else=SKIP tif.src_0 ! other/tensors,num_tensors=1 ! filesink location=\"result09.log\"" 9 0 0 $PERFORMANCE
callCompareTest pick_range_exclusive.golden result09.log 9 "TENSOR_AVERAGE_VALUE RANGE_EXCLUSIVE, TENSORPICK" 0 0

# TENSORPICK on both branches, reordering the tensors of the then branch
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:0,0 operator=GE supplied-value=10 then=TENSORPICK then-option=\" 1 , 0\" else=TENSORPICK else-option=0 tif.src_0 ! queue ! other/tensors,num_tensors=2 ! filesink location=\"result10_0.log\" tif.src_1 ! queue ! other/tensors,num_tensors=1 ! filesink location=\"result10_1.log\"" 10 0 0 $PERFORMANCE
callCompareTest pick_then.golden result10_0.log 10_0 "TENSORPICK 1,0 on the then branch" 0 0
callCompareTest pick_else.golden result10_1.log 10_1 "TENSORPICK 0 on the else branch" 0 0

# A supplied value that holds every frame on the then branch; the else branch would fail on tensor 2
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:0,0 operator=RANGE_INCLUSIVE supplied-value=\"0, 255\" then=PASSTHROUGH else=TENSORPICK else-option=2 tif.src_0 ! filesink location=\"result11.log\"" 11 0 0 $PERFORMANCE
callCompareTest passthrough_all.golden result11.log 11 "RANGE_INCLUSIVE 0,255, PASSTHROUGH of every frame" 0 0

# A refused compared-value-option clears the option, and every buffer fails.
# Parsing a token up to its first non-digit would read "0:0:0:x,0" as "0:0:0:0,0" and "0x1" as "0".
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:x,0 operator=GE supplied-value=10 then=PASSTHROUGH else=SKIP tif.src_0 ! filesink location=\"result12_n.log\"" 12_n 0 1 $PERFORMANCE
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:0,-1 operator=GE supplied-value=10 then=PASSTHROUGH else=SKIP tif.src_0 ! filesink location=\"result13_n.log\"" 13_n 0 1 $PERFORMANCE
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=TENSOR_AVERAGE_VALUE compared-value-option=0x1 operator=RANGE_INCLUSIVE supplied-value=10,20 then=TENSORPICK then-option=1 else=SKIP tif.src_0 ! filesink location=\"result14_n.log\"" 14_n 0 1 $PERFORMANCE

# Cases 15_n to 18_n are case 10 with one malformed option and the same capsfilters.
# A refused then-option or else-option leaves the option empty, so TENSORPICK picks no tensor
# and the capsfilter refuses the branch. Parsing a token up to its first non-digit
# would read "0,abc", "1,0x1" and "1, " as two tensors and "abc" as one, passing the capsfilter.
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:0,0 operator=GE supplied-value=10 then=TENSORPICK then-option=0,abc else=TENSORPICK else-option=0 tif.src_0 ! queue ! other/tensors,num_tensors=2 ! filesink location=\"result15_0_n.log\" tif.src_1 ! queue ! other/tensors,num_tensors=1 ! filesink location=\"result15_1_n.log\"" 15_n 0 1 $PERFORMANCE
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:0,0 operator=GE supplied-value=10 then=TENSORPICK then-option=1,0x1 else=TENSORPICK else-option=0 tif.src_0 ! queue ! other/tensors,num_tensors=2 ! filesink location=\"result16_0_n.log\" tif.src_1 ! queue ! other/tensors,num_tensors=1 ! filesink location=\"result16_1_n.log\"" 16_n 0 1 $PERFORMANCE
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:0,0 operator=GE supplied-value=10 then=TENSORPICK then-option=\"1, \" else=TENSORPICK else-option=0 tif.src_0 ! queue ! other/tensors,num_tensors=2 ! filesink location=\"result17_0_n.log\" tif.src_1 ! queue ! other/tensors,num_tensors=1 ! filesink location=\"result17_1_n.log\"" 17_n 0 1 $PERFORMANCE
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:0,0 operator=GE supplied-value=10 then=TENSORPICK then-option=1,0 else=TENSORPICK else-option=abc tif.src_0 ! queue ! other/tensors,num_tensors=2 ! filesink location=\"result18_0_n.log\" tif.src_1 ! queue ! other/tensors,num_tensors=1 ! filesink location=\"result18_1_n.log\"" 18_n 0 1 $PERFORMANCE

# Case 11 with a malformed supplied-value. A refused supplied-value leaves it at 0,0, so every frame
# goes to the else branch, which fails.
# Parsing a token up to its first non-digit would read "0,255x" as "0,255".
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} ${SRC} ! tensor_if name=tif compared-value=A_VALUE compared-value-option=0:0:0:0,0 operator=RANGE_INCLUSIVE supplied-value=0,255x then=PASSTHROUGH else=TENSORPICK else-option=2 tif.src_0 ! filesink location=\"result19_n.log\"" 19_n 0 1 $PERFORMANCE

report

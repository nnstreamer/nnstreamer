#!/usr/bin/env bash
##
## SPDX-License-Identifier: LGPL-2.1-only
##
## @file runTest.sh
## @author Parichay Kapoor <pk.kapoor@samsung.com>
## @date May 7th 2019
## @brief SSAT Test Cases for NNStreamer pytorch plugin
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

# Test constant passthrough custom filter (1, 2)
PATH_TO_PLUGIN="../../build"
PATH_TO_MODEL="../test_models/models/pytorch_lenet5.pt"
PATH_TO_IMAGE="../test_models/data/9.png"

if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer/tensor_filter"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep pytorch.so)
        if [[ ! $check ]]; then
            echo "Cannot find pytorch shared lib"
            report
            exit
        fi
    else
        echo "Cannot find ${ini_path}"
    fi
else
    ini_file="/etc/nnstreamer.ini"
    if [[ -f ${ini_file} ]]; then
        path=$(grep "^filters" ${ini_file})
        key=${path%=*}
        value=${path##*=}

        if [[ $key != "filters" ]]; then
            echo "String Error"
            report
            exit
        fi

        if [[ -d ${value} ]]; then
            check=$(ls ${value} | grep pytorch.so)
            if [[ ! $check ]]; then
                echo "Cannot find pytorch shared lib"
                report
                exit
            fi
        else
            echo "Cannot file ${value}"
            report
            exit
        fi
    else
        echo "Cannot identify nnstreamer.ini"
        report
        exit
    fi
fi

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} input=1:28:28:1 inputtype=uint8 output=10:1:1:1 outputtype=uint8 ! filesink location=tensorfilter.out.log" 1 0 0 $PERFORMANCE
python3 checkLabel.py tensorfilter.out.log ${PATH_TO_IMAGE}
testResult $? 1 "Golden test comparison" 0 1

# Fail test for invalid input properties
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} input=7:1 inputtype=float32 ! filesink location=tensorfilter.out.log" 2F_n 0 1 $PERFORMANCE

# Fail test for invalid output properties
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} output=1:7 outputtype=int8 ! filesink location=tensorfilter.out.log" 3F_n 0 1 $PERFORMANCE

# Input and output combination test
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc pattern=13 num-buffers=1 ! videoconvert !  video/x-raw,width=640,height=480,framerate=30/1 ! tensor_converter ! tee name=t t. ! queue ! mux.sink_0 t. ! queue ! filesink location=combi.dummy.golden buffer-mode=unbuffered sync=false async=false  filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! mux.sink_1 tensor_mux name=mux ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} input=1:28:28:1 inputtype=uint8 output=10:1:1:1 outputtype=uint8 input-combination=1 output-combination=i0,o0 ! tensor_demux name=demux demux.src_0 ! filesink location=tensorfilter.combi.in.log buffer-mode=unbuffered sync=false async=false demux.src_1 ! filesink location=tensorfilter.combi.out.log buffer-mode=unbuffered sync=false async=false" 4 0 0 $PERFORMANCE
callCompareTest combi.dummy.golden tensorfilter.combi.in.log 4_0 "Output Combination Golden Test 4-0" 1 0
python3 checkLabel.py tensorfilter.out.log ${PATH_TO_IMAGE}
testResult $? 1 "Golden test comparison" 0 1

# Test the setting of accelerators
gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} output=1:7 outputtype=int8 accelerator=true:gpu ! filesink location=tensorfilter.out.log 2>info
cat info | grep "gpu = 1, accl = gpu"
testResult $? 5-1 "GPU activation test" 0 1

gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} output=1:7 outputtype=int8 accelerator=true:cpu ! filesink location=tensorfilter.out.log 2>info
cat info | grep "gpu = 0, accl = cpu"
testResult $? 5-2 "GPU activation test" 0 1

gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} filesrc location=${PATH_TO_IMAGE} ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} output=1:7 outputtype=int8 ! filesink location=tensorfilter.out.log 2>info
cat info | grep "gpu = 0, accl = cpu"
testResult $? 5-3 "GPU activation test" 0 1

PATH_TO_MODEL="../test_models/models/sample_3x4_two_input_two_output.pt"
# This model is made with below simple python script:
#
# import torch
# class MyCell(torch.nn.Module):
#     def __init__(self):
#         super(MyCell, self).__init__()
#     def forward(self, x, y):
#         new_x = x + 1.0
#         new_y = y + 2.0
#         return new_x, new_y
# my_cell = MyCell()
# x, y = torch.rand(3, 4), torch.rand(3, 4)
# traced_cell = torch.jit.trace(my_cell, (x, y))
# traced_cell(x, y)
# traced_cell.save('sample_3x4_two_input_two_output.pt')

# Test multiple input output tensors

## wrong input dimension
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=2 ! videoscale ! videoconvert ! video/x-raw,width=3,height=4,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=2:1:0:3 ! other/tensors,num_tensors=1,dimensions=4:3:1:1,types=uint8,format=static,framerate=0/1 ! tensor_transform mode=typecast option=float32 ! tee name=t t. ! queue ! mux.sink_0 t. ! queue ! mux.sink_1  tensor_mux name=mux sync_mode=nosync ! queue ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} input=4:3:1:1.4:3:2:1 inputtype=float32.float32 output=4:3:1:1.4:3:1:1 outputtype=float32.float32 ! tensor_sink" 6_n 0 1 $PERFORMANCE

## wrong output dimension
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=2 ! videoscale ! videoconvert ! video/x-raw,width=3,height=4,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=2:1:0:3 ! other/tensors,num_tensors=1,dimensions=4:3:1:1,types=uint8,format=static,framerate=0/1 ! tensor_transform mode=typecast option=float32 ! tee name=t t. ! queue ! mux.sink_0 t. ! queue ! mux.sink_1  tensor_mux name=mux sync_mode=nosync ! queue ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} input=4:3:1:1.4:3:1:1 inputtype=float32.float32 output=4:3:1:1.4:3:2:1 outputtype=float32.uint8 ! tensor_sink" 7_n 0 1 $PERFORMANCE

## correct input/output info
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=2 ! videoscale ! videoconvert ! video/x-raw,width=3,height=4,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=2:1:0:3 ! other/tensors,num_tensors=1,dimensions=4:3:1:1,types=uint8,format=static,framerate=0/1 ! tensor_transform mode=typecast option=float32 ! tee name=t t. ! queue ! mux.sink_0 t. ! queue ! mux.sink_1  tensor_mux name=mux sync_mode=nosync ! queue ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} input=4:3:1:1.4:3:1:1 inputtype=float32.float32 output=4:3:1:1.4:3:1:1 outputtype=float32.float32 ! filesink location=tensorfilter.out.log" 8 0 0 $PERFORMANCE

## transform after filter specifying format as static
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=2 ! videoscale ! videoconvert ! video/x-raw,width=3,height=4,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=2:1:0:3 ! other/tensors,num_tensors=1,dimensions=4:3:1:1,types=uint8,format=static,framerate=0/1 ! tensor_transform mode=typecast option=float32 ! tee name=t t. ! queue ! mux.sink_0 t. ! queue ! mux.sink_1  tensor_mux name=mux sync_mode=nosync ! queue ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} input=4:3:1:1.4:3:1:1 inputtype=float32.float32 output=4:3:1:1.4:3:1:1 outputtype=float32.float32 ! other/tensors,format=static ! tensor_transform mode=typecast option=uint8 ! filesink location=tensorfilter.out.log" 9 0 0 $PERFORMANCE

## Dimension declaration test cases
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=2 ! videoscale ! videoconvert ! video/x-raw,width=3,height=4,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=2:1:0:3 ! other/tensors,num_tensors=1,dimensions=4:3:1:1,types=uint8,format=static,framerate=0/1 ! tensor_transform mode=typecast option=float32 ! tee name=t t. ! queue ! mux.sink_0 t. ! queue ! mux.sink_1  tensor_mux name=mux sync_mode=nosync ! queue ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} input=4:3:1:1.4:3:1:1 inputtype=float32.float32 output=4:3:1:1.4:3:1:1 outputtype=float32.float32 ! other/tensors,num_tensors=2,dimensions=4:3:1.4:3:1 ! filesink location=tensorfilter.out.log" 10-1 0 0 $PERFORMANCE

gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=2 ! videoscale ! videoconvert ! video/x-raw,width=3,height=4,format=GRAY8,framerate=0/1 ! tensor_converter ! tensor_transform mode=transpose option=2:1:0:3 ! other/tensors,num_tensors=1,dimensions=4:3:1:1,types=uint8,format=static,framerate=0/1 ! tensor_transform mode=typecast option=float32 ! tee name=t t. ! queue ! mux.sink_0 t. ! queue ! mux.sink_1  tensor_mux name=mux sync_mode=nosync ! queue ! tensor_filter framework=pytorch model=${PATH_TO_MODEL} input=4:3:1:1.4:3:1:1 inputtype=float32.float32 output=4:3:1:1.4:3:1:1 outputtype=float32.float32 ! other/tensors,num_tensors=2,dimensions=4:3.4:3 ! filesink location=tensorfilter.out.log" 10-2 0 0 $PERFORMANCE

# Cleanup
rm info *.log *.golden

report

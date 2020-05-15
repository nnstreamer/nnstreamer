#!/usr/bin/env bash

# @Author: Geunsik Lim <geunsik.lim@samsung.com>
# @brief: The simple script is to build and run the VIM3/Vivante fiflter 
# This script has been evaluated on the Ubuntu 18.04 (arm64) + VIM3/Vivante board.
# @note Please install the below package to enable nnstreamer on a Tizen board
#   zypper clean; zypper refresh
#   zypper install meson ninja cmake
#   zypper install gst-plugins-base-devel
#   zypper install gst-plugins-good
#   zypper install gst-plugins-good-devel
#   zypper install gstreamer-devel
#   zypper install nnstreamer nnstreamer-devel
#   zypper install libjpeg libjpeg-devel
#   zypper install gst-libav
#
# @note You must install the below packages to use 'zypper' command.
#   augeas-libs, bzip2, libsolv, libsolv-tools, libzypp, pacrunner-libproxy, zypper
# 
#################### Configuration setting ################################
# Specify the test items that you want to execute for the .so of Vivante model.
BUILD=1
RUN_TEST=1
RUN_TEST_NUM=1
CORRECTNESS=0

# Specify higher number (0..5) to display more log messages for debugging.
export VSI_NN_LOG_LEVEL=5


#################### Build ################################################
if [[ $BUILD == 1 ]]; then
    echo -e "Compling source .........."
    rm -rf ./build 
    meson  -Denable-vivante=true  build 
    ninja -C build  
    
    if [[ $? != 0 ]]; then
        echo -e "Ooops. The compilation task is failed. Please fix the source code."
        exit 1;
    fi
    
    filter_dir="/usr/lib/nnstreamer/filters/"
    echo -e "Copying a Vivante tensor filter to $filter_dir .........."
    echo -e "Location: $filter_dir " 
    cp ./build/ext/nnstreamer/tensor_filter/vivante/libnnstreamer_filter_vivante.so  $filter_dir
    
    # gst-inspect-1.0 tensor_filter
else
    echo -e "Skipping a task for compiling source code .........."
fi



#################### Run Test (Aging Test) ##############################

# CMD='gst-launch-1.0 videotestsrc ! videoconvert ! videoscale !   tensor_converter ! fakesink'

# To test the Tensorflow-Lite model
# CMD='gst-launch-1.0 filesrc location=/mnt/sda2/nnstreamer/tests/test_models/data/orange.png ! pngdec ! videoscale ! imagefreeze ! videoconvert ! video/x-raw,format=RGB,framerate=0/1 ! tensor_converter ! tensor_filter framework=tensorflow-lite model=/mnt/sda2/nnstreamer/tests/test_models/models/mobilenet_v1_1.0_224_quant.tflite  ! filesink location=tensorfilter.out.log '


# To test the inceptionv3/Vivante model
# tensor_transform mode=typecast option=float32 ! tensor_transform mode=arithmetic option=add:-128,div:128 !
#CMD='gst-launch-1.0 filesrc location=/usr/share/vivante/res/goldfish_299x299.jpg ! jpegdec ! videoconvert ! video/x-raw,format=RGB,width=299,height=299 ! tensor_converter ! tensor_filter framework=vivante model="/usr/share/vivante/inceptionv3/inception_v3.nb,/usr/share/vivante/inceptionv3/libinceptionv3_ori.so" ! filesink location=vivante.out.bin'


# To test the yolov3/Vivante model
# tensor_transform mode=typecast option=float32 ! tensor_transform mode=arithmetic option=add:0,div:256 !
CMD='gst-launch-1.0 filesrc location=/usr/share/vivante/res/sample_car_bicyle_dog_416x416.jpg ! jpegdec ! videoconvert ! video/x-raw,format=BGR,width=416,height=416 ! tensor_converter ! tensor_transform mode=transpose option=1:2:0:3 ! tensor_transform mode=typecast option=int8 ! tensor_filter framework=vivante model="/usr/share/vivante/yolov3/yolov3.nb,/usr/share/vivante/yolov3/libyolov3_ori.so" ! filesink location=vivante.out.bin'


sync
if [[ $RUN_TEST == 1 ]]; then
    count=0
    while [[ true ]]; do
        ((count++))
        echo -e "\e[34m ============ Aging Test STEP: $count / $RUN_TEST_NUM ================= \e[0m"
        start_time=$( date +%s.%N )
        $CMD
        if [[ $? != 0 ]]; then
            echo -e "Oooops. The exectuion is failed. Pleasse fix a bug."
            exit 1
        fi
        elapsed_time=$( date +%s.%N --date="$start_time seconds ago" )
        echo -e "\e[91m Elapsed Time: $elapsed_time \e[0m"
        sleep 1
        if [[ $count -ge $RUN_TEST_NUM ]]; then
            echo -e "\n\e[91m Congrats!!! The aging test ($RUN_TEST_NUM times) is successfully completed. \e[0m "
            exit 0
        fi
        done
else
    echo -e "Skipping a aging test .........."
fi


#################### Checking a correctness of an output tensor ##############################
# You can find checkLabel.py and gen24BMP.py file at the nnstreamer github repository.
# python checkLabel.py vivante.out.dat $model_dir/imagenet_slim_labels.txt dog

if [[ $CORRECTNESS == 1 ]]; then
    ls -al ./*.dat
    echo -e "Please run the 'diff' command in order to check a difference"
    echo -e "between the 'output*.dat' file and the 'nnstreamer*.dat' file."
    echo -e "For example, $ diff -urN ./output*.dat ./nnstreamer*.dat."
else
    echo -e "Skipping a correctness check .........."
fi



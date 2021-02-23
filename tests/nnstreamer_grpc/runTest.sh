#!/usr/bin/env bash
# SPDX-License-Identifier: LGPL-2.1-only
#
# @file runTest.sh
# @author Dongju Chae <dongju.chae@samsung.com>
# @date Nov 04 2020
# @brief SSAT Test Cases for NNStreamer
#

if [[ "$SSATAPILOADED" != "1" ]]; then
    SILENT=0
    INDEPENDENT=1
    search="ssat-api.sh"
    source $search
    printf "${Blue}Independent Mode${NC}"
fi

# This is compatible with SSAT (https://github.com/myungjoo/SSAT)
testInit $1

# Skip test on qemu env.
if [[ "$SKIP_QEMU_ARM64_INCOMPATIBLE_TESTS" == "1" ]]; then
  echo "Skip ssat tests on qemu/arm64 env."
  report
  exit
fi

# Check gRPC availability
PATH_TO_PLUGIN=${NNSTREAMER_BUILD_ROOT_PATH}
if [[ -d $PATH_TO_PLUGIN ]]; then
    ini_path="${PATH_TO_PLUGIN}/ext/nnstreamer"
    if [[ -d ${ini_path} ]]; then
        check=$(ls ${ini_path} | grep nnstreamer-grpc.so)
        if [[ ! $check ]]; then
            echo "Cannot find nnstreamer-grpc shared lib"
            report
            exit
        fi
    else
        echo "Cannot find ${ini_path}"
    fi
fi

# Check IDL availability
PATH_TO_PLUGIN_EXTRA=${PATH_TO_PLUGIN}/ext/nnstreamer/extra
TEST_PROTOBUF=1
TEST_FLATBUF=1
if [[ -d $PATH_TO_PLUGIN_EXTRA ]]; then
  ini_path="${PATH_TO_PLUGIN_EXTRA}"
  if [[ -d ${ini_path} ]]; then
    check=$(ls ${ini_path} | grep nnstreamer_grpc_protobuf.so)
    if [[ ! $check ]]; then
      echo "Cannot find nnstreamer_grpc_protobuf shared lib"
      TEST_PROTOBUF=0
    fi
    check=$(ls ${ini_path} | grep nnstreamer_grpc_flatbuf.so)
    if [[ ! $check ]]; then
      echo "Cannot find nnstreamer_grpc_flatbuf shared lib"
      TEST_FLATBUF=0
    fi
  else
    echo "Cannot find ${ini_path}"
  fi
fi

export LD_LIBRARY_PATH=${PATH_TO_PLUGIN_EXTRA}:${LD_LIBRARY_PATH}

NUM_BUFFERS=10

# Dump original frames, passthrough, other/tensor
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter ! multifilesink location=original1_%1d.log" 0 0 0 $PERFORMANCE

# Dump original frames, passthrough, other/tensors
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter frames-per-tensor=2 ! multifilesink location=original2_%1d.log" 1 0 0 $PERFORMANCE

if [[ $TEST_PROTOBUF -eq 1 ]]; then
  IDL="protobuf"
  PORT=`python3 get_available_port.py`
  # tensor_sink (client) --> tensor_src (server), other/tensor, protobuf
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=${NUM_BUFFERS} idl=${IDL} ! 'other/tensor,dimension=(string)3:640:480,type=(string)uint8,framerate=(fraction)5/1' ! multifilesink location=result_%1d.log" 2 0 0 $PERFORMANCE &
  sleep 1
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter ! tensor_sink_grpc port=${PORT} idl=${IDL}" 2 0 0 $PERFORMANCE

  for i in `seq 0 $((NUM_BUFFERS-1))`
  do
    callCompareTest original1_${i}.log result_${i}.log 2-${i} "gRPC/Protobuf Compare 2-${i}" 0 0
  done

  rm result_*.log

  PORT=`python3 get_available_port.py`
  # tensor_sink (server) --> tensor_src (client), other/tensor, protobuf
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter ! tensor_sink_grpc port=${PORT} server=true idl=${IDL}" 3 0 0 $PERFORMANCE &
  sleep 1
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=${NUM_BUFFERS} server=false idl=${IDL} ! 'other/tensor,dimension=(string)3:640:480,type=(string)uint8,framerate=(fraction)5/1' ! multifilesink location=result_%1d.log" 3 0 0 $PERFORMANCE

  for i in `seq 0 $((NUM_BUFFERS-1))`
  do
    callCompareTest original1_${i}.log result_${i}.log 3-${i} "gRPC/Protobuf Compare 3-${i}" 0 0
  done

  rm result_*.log

  PORT=`python3 get_available_port.py`
  # tensor_sink (client) --> tensor_src (server), other/tensors, protobuf
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=$((NUM_BUFFERS/2)) idl=${IDL} ! 'other/tensors,num_tensors=2,dimensions=(string)3:640:480.3:640:480,types=(string)uint8.uint8,framerate=(fraction)5/1' ! multifilesink location=result_%1d.log" 4 0 0 $PERFORMANCE &
  sleep 1
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter frames-per-tensor=2 ! tensor_sink_grpc port=${PORT} idl=${IDL}" 4 0 0 $PERFORMANCE

  for i in `seq 0 $((NUM_BUFFERS/2-1))`
  do
    callCompareTest original2_${i}.log result_${i}.log 4-${i} "gRPC/Protobuf Compare 4-${i}" 0 0
  done

  rm result_*.log

  PORT=`python3 get_available_port.py`
  # tensor_sink (server) --> tensor_src (client), other/tensors, protobuf
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter frames-per-tensor=2 ! tensor_sink_grpc port=${PORT} server=true idl=${IDL}" 5 0 0 $PERFORMANCE &
  sleep 1
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=$((NUM_BUFFERS/2)) server=false idl=${IDL} ! 'other/tensors,num_tensors=2,dimensions=(string)3:640:480.3:640:480,types=(string)uint8.uint8,framerate=(fraction)5/1' ! multifilesink location=result_%1d.log" 5 0 0 $PERFORMANCE

  for i in `seq 0 $((NUM_BUFFERS/2-1))`
  do
    callCompareTest original2_${i}.log result_${i}.log 5-${i} "gRPC/Protobuf Compare 5-${i}" 0 0
  done

  rm result_*.log
fi

if [[ $TEST_FLATBUF -eq 1 ]]; then
  IDL="flatbuf"
  PORT=`python3 get_available_port.py`
  # tensor_sink (client) --> tensor_src (server), other/tensor, flatbuf
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=${NUM_BUFFERS} idl=${IDL} ! 'other/tensor,dimension=(string)3:640:480,type=(string)uint8,framerate=(fraction)5/1' ! multifilesink location=result_%1d.log" 6 0 0 $PERFORMANCE &
  sleep 1
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter ! tensor_sink_grpc port=${PORT} idl=${IDL}" 6 0 0 $PERFORMANCE

  for i in `seq 0 $((NUM_BUFFERS-1))`
  do
    callCompareTest original1_${i}.log result_${i}.log 6-${i} "gRPC/Flatbuf Compare 6-${i}" 0 0
  done

  rm result_*.log

  PORT=`python3 get_available_port.py`
  # tensor_sink (server) --> tensor_src (client), other/tensor, flatbuf
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter ! tensor_sink_grpc port=${PORT} server=true idl=${IDL}" 7 0 0 $PERFORMANCE &
  sleep 1
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=${NUM_BUFFERS} server=false idl=${IDL} ! 'other/tensor,dimension=(string)3:640:480,type=(string)uint8,framerate=(fraction)5/1' ! multifilesink location=result_%1d.log" 7 0 0 $PERFORMANCE

  for i in `seq 0 $((NUM_BUFFERS-1))`
  do
    callCompareTest original1_${i}.log result_${i}.log 7-${i} "gRPC/Flatbuf Compare 7-${i}" 0 0
  done

  rm result_*.log

  PORT=`python3 get_available_port.py`
  # tensor_sink (client) --> tensor_src (server), other/tensors, flatbuf
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=$((NUM_BUFFERS/2)) idl=${IDL} ! 'other/tensors,num_tensors=2,dimensions=(string)3:640:480.3:640:480,types=(string)uint8.uint8,framerate=(fraction)5/1' ! multifilesink location=result_%1d.log" 8 0 0 $PERFORMANCE &
  sleep 1
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter frames-per-tensor=2 ! tensor_sink_grpc port=${PORT} idl=${IDL}" 8 0 0 $PERFORMANCE

  for i in `seq 0 $((NUM_BUFFERS/2-1))`
  do
    callCompareTest original2_${i}.log result_${i}.log 8-${i} "gRPC/Flatbuf Compare 8-${i}" 0 0
  done

  rm result_*.log

  PORT=`python3 get_available_port.py`
  # tensor_sink (server) --> tensor_src (client), other/tensors, flatbuf
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter frames-per-tensor=2 ! tensor_sink_grpc port=${PORT} server=true idl=${IDL}" 9 0 0 $PERFORMANCE &
  sleep 1
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=$((NUM_BUFFERS/2)) server=false idl=${IDL} ! 'other/tensors,num_tensors=2,dimensions=(string)3:640:480.3:640:480,types=(string)uint8.uint8,framerate=(fraction)5/1' ! multifilesink location=result_%1d.log" 9 0 0 $PERFORMANCE

  for i in `seq 0 $((NUM_BUFFERS/2-1))`
  do
    callCompareTest original2_${i}.log result_${i}.log 9-${i} "gRPC/Flatbuf Compare 9-${i}" 0 0
  done

  rm result_*.log
fi

rm original*.log

report

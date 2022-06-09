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
if [[ "$SKIP_QEMU_ARM_INCOMPATIBLE_TESTS" == "1" ]]; then
  echo "Skip ssat tests on qemu/arm and qemu/arm64 environments."
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

TIMEOUT_SEC=10
## Check if a pipeline is launched and playing
function waitformarker {
  for i in $(seq 1 ${TIMEOUT_SEC})
  do
    if [ -f 'marker.log' ]; then
      markersize=$(stat -c%s marker.log)
      if [ $markersize -ge 48 ]; then
        testResult 1 $1 "$2" 0 0
        return 0
      fi
    fi
    sleep 1
  done
  testResult 0 $1 "$2" 0 0
  exit
}
WAIT4MARKER=" videotestsrc num-buffers=1 ! video/x-raw,format=RGB,height=4,width=4 ! filesink location=marker.log "

## Initial pipelines to generate reference outputs
# passthrough, other/tensor
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter ! multifilesink location=original1_%1d.log" Initial-1 0 0 $PERFORMANCE
# passthrough, other/tensors
gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter frames-per-tensor=2 ! multifilesink location=original2_%1d.log" Initial-2 0 0 $PERFORMANCE

INDEX=1
BLOCKING_LIST=("TRUE" "FALSE")
IDL_LIST=()
if [[ $TEST_PROTOBUF -eq 1 ]]; then
  IDL_LIST[${#IDL_LIST[@]}]="Protobuf"
fi
if [[ $TEST_FLATBUF -eq 1 ]]; then
  IDL_LIST[${#IDL_LIST[@]}]="Flatbuf"
fi

## Test gRPC multi-pipelines with different IDL and blocking modes.
for IDL in "${IDL_LIST[@]}"; do
for BLOCKING in "${BLOCKING_LIST[@]}"; do
  PORT=`python3 ../get_available_port.py`
  BLOCKING_STR="Blocking"
  if [[ $BLOCKING == "FALSE" ]]; then
    BLOCKING_STR="Non-blocking"
  fi

  # tensor_sink (client) --> tensor_src (server), other/tensor
  rm -f marker.log
  timeout -k 11s 11s gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=${NUM_BUFFERS} idl=${IDL} blocking=${BLOCKING} ! 'other/tensor,dimension=(string)3:640:480,type=(string)uint8,framerate=(fraction)5/1' ! multifilesink async=false location=result_%1d.log ${WAIT4MARKER} &
  pid=$!
  waitformarker ${INDEX}-1 "tensor_src_grpc ${INDEX}-1 launching in parallel"
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter ! tensor_sink_grpc port=${PORT} idl=${IDL} blocking=${BLOCKING}" ${INDEX}-2 0 0 $PERFORMANCE
  kill -9 $pid &> /dev/null
  wait $pid

  for i in `seq 0 $((NUM_BUFFERS-1))`
  do
    callCompareTest original1_${i}.log result_${i}.log GoldenTest-${INDEX} "gRPC ${IDL}/${BLOCKING_STR} $((i+1))/${NUM_BUFFERS}" 0 0
  done

  INDEX=$((INDEX + 1))
  rm result_*.log

  PORT=`python3 ../get_available_port.py`
  # tensor_sink (server) --> tensor_src (client), other/tensor
  rm -f marker.log
  timeout -k 11s 11s gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter ! tensor_sink_grpc port=${PORT} server=true idl=${IDL} blocking=${BLOCKING} async=false ${WAIT4MARKER} &
  pid=$!
  waitformarker ${INDEX}-1 "tensor-sink_grpc ${INDEX}-1 launching in parallel"
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=${NUM_BUFFERS} server=false idl=${IDL} blocking=${BLOCKING} ! 'other/tensor,dimension=(string)3:640:480,type=(string)uint8,framerate=(fraction)5/1' ! multifilesink location=result_%1d.log" ${INDEX}-2 0 0 $PERFORMANCE
  kill -9 $pid &> /dev/null
  wait $pid

  for i in `seq 0 $((NUM_BUFFERS-1))`
  do
    callCompareTest original1_${i}.log result_${i}.log GoldenTest-${INDEX} "gRPC ${IDL}/${BLOCKING_STR} $((i+1))/${NUM_BUFFERS}" 0 0
  done

  INDEX=$((INDEX + 1))
  rm result_*.log

  PORT=`python3 ../get_available_port.py`
  # tensor_sink (client) --> tensor_src (server), other/tensors
  rm -f marker.log
  timeout -k 11s 11s gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=$((NUM_BUFFERS/2)) idl=${IDL} blocking=${BLOCKING} ! 'other/tensors,num_tensors=2,dimensions=(string)3:640:480.3:640:480,types=(string)uint8.uint8,framerate=(fraction)5/1' ! multifilesink async=false location=result_%1d.log ${WAIT4MARKER} &
  pid=$!
  waitformarker ${INDEX}-1 "tensor-src-grpc ${INDEX}-1 launching in parallel"
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter frames-per-tensor=2 ! tensor_sink_grpc port=${PORT} idl=${IDL} blocking=${BLOCKING}" ${INDEX}-2 0 0 $PERFORMANCE
  kill -9 $pid &> /dev/null
  wait $pid

  for i in `seq 0 $((NUM_BUFFERS/2-1))`
  do
    callCompareTest original2_${i}.log result_${i}.log GoldenTest-${INDEX} "gRPC ${IDL}/${BLOCKING_STR} $((i+1))/$((NUM_BUFFERS/2))" 0 0
  done

  INDEX=$((INDEX + 1))
  rm result_*.log

  PORT=`python3 ../get_available_port.py`
  # tensor_sink (server) --> tensor_src (client), other/tensors
  rm -f marker.log
  timeout -k 11s 11s gst-launch-1.0 --gst-plugin-path=${PATH_TO_PLUGIN} videotestsrc num-buffers=${NUM_BUFFERS} ! video/x-raw,width=640,height=480,framerate=5/1 ! tensor_converter frames-per-tensor=2 ! tensor_sink_grpc port=${PORT} server=true idl=${IDL} blocking=${BLOCKING} async=false ${WAIT4MARKER} &
  pid=$!
  waitformarker ${INDEX}-1 "tensor-sink-grpc ${INDEX}-1 launching in parallel"
  gstTest "--gst-plugin-path=${PATH_TO_PLUGIN} tensor_src_grpc port=${PORT} num-buffers=$((NUM_BUFFERS/2)) server=false idl=${IDL} blocking=${BLOCKING} ! 'other/tensors,num_tensors=2,dimensions=(string)3:640:480.3:640:480,types=(string)uint8.uint8,framerate=(fraction)5/1' ! multifilesink location=result_%1d.log" ${INDEX}-2 0 0 $PERFORMANCE
  kill -9 $pid &> /dev/null
  wait $pid

  for i in `seq 0 $((NUM_BUFFERS/2-1))`
  do
    callCompareTest original2_${i}.log result_${i}.log GoldenTest-${INDEX} "gRPC ${IDL}/${BLOCKING_STR} $((i+1))/$((NUM_BUFFERS/2))" 0 0
  done

  INDEX=$((INDEX + 1))
  rm result_*.log
done
done

rm original*.log

report

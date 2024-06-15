#!/bin/bash
NV_INFER_VERSION_H=$(find /usr/include -iname "NvInferVersion.h")
if [ -z ${NV_INFER_VERSION_H} ]; then
  exit
fi
NV_TENSORRT_MAJOR=$(cat ${NV_INFER_VERSION_H} | grep NV_TENSORRT_MAJOR | awk '{ print $3 }')
NV_TENSORRT_MINOR=$(cat ${NV_INFER_VERSION_H} | grep NV_TENSORRT_MINOR | awk '{ print $3 }')
echo "${NV_TENSORRT_MAJOR}.${NV_TENSORRT_MINOR}"

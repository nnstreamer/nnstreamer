---
title: NNStreamer tensor\_filter default subplugin manual
...

# Introduction

## What is tensor-filter subplugin?

(( TBU ))


## Which subplugins are located here?

(( TBU ))

# List of subplugins

## Vivante
- subplugin name: 'vivante'

## Movidius-ncsdk2
## NNFW
## Armnn
## Caffe2
## CPP (C++)
## Edgetpu
## Lua
## Mediapipe
## Openvino
## Python3
## Pytorch
## Snap
## SNPE
## Tensorflow
## Tensorflow-lite
- subplugin name: 'tensorflow1-lite'
- subplugin name: 'tensorflow2-lite'
- subplugin name: 'tensorflow2-lite-custom'

### How to use custom tensorflow-lite binaries

If you want to use tensorflow-lite custom operators with your own tensorflow-lite custom binaries, you can use tensorflow2-lite-custom subplugin. As its name suggests, this supports tensorflow-lite 2.x versions.

By default, this subplugin loads ```./libtensorflow2-lite-custom.so```, which is the user's custom tensorflow-lite binary.

## Tensorrt
## TRIx-engine
## TVM

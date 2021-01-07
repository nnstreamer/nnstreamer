---
title: android_source
...

# NNStreamer::android_source

## Motivation

Android's standard MMFW (i.e., StageFright) already has an optimized capability to pre-process media files. 
If we can utilize such features in pre-processing, it's expected to gain more optimized performance.
This source element (amcsrc) directly feeds the result (i.e., decoded frames) of Android MediaCodec (AMC) into directly feed into the pipeline of gstreamer.
To avoid extra memcpy() overhead, it uses the memory wrapping for output buffer of media codec.

## Sources
- gstamcsrc.c: main source file to implement a source element of Android MediaCodec (AMC) 
- gstamcsrc_looper.cc: a looper thread to perform event messages between amcsrc and media codec.

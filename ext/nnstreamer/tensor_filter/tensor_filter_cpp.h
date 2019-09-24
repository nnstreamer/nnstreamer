/**
 * GStreamer Tensor_Filter, Customized C++ Module
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file	tensor_filter_cpp.h
 * @date	24 Sep 2019
 * @brief	Custom tensor processing interface for C++ codes.
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * How To for NNdevelopers (C++ subplugin writers):
 *
 * @todo Write This
 *
 * @note This is experimental. The API and Class definition is NOT stable.
 *       If you want to write an OpenCV custom filter for nnstreamer, this is a good choice.
 *
 * To Packagers:
 *
 * This is to be exposed with "nnstreamer-c++-dev"
 */
#ifndef __NNS_TENSOR_FITLER_CPP_H__
#define __NNS_TENSOR_FITLER_CPP_H__

#include <stdint.h>

#ifdef __cplusplus
/**
 * @brief This allows to have a c++ class inserted as a filter in a neural network pipeline of nnstreamer
 * @note This is experimental.
 */
class tensor_filter_cpp {
  private:
    const uint32_t validity;
    const char *name; /**< The name of this C++ custom filter, searchable with "model" property */

  public:
    tensor_filter_cpp(const char *modelName); /**< modelName is the model property of tensor_filter, which could be the path to the model file (requires proper extension name) or the registered model name at runtime. */
    ~tensor_filter_cpp();

    /** C++ plugin writers need to fill {getInput/Output} inclusive-or {setInput} */
    virtual int getInputDim(GstTensorsInfo *info) = 0;
    virtual int getOutputDim(GstTensorsInfo *info) = 0;

    virtual int setIutputDim(const GstTensorsInfo *in, GstTensorsInfo *out) = 0;

    bool allocate_before_invoke; /**< TRUE, if you want nnstreamer to preallocate output buffers before calling invoke */
    virtual int invoke(const GstTensorMemory *in, GstTensorMemory *out) = 0;

    /** API */
    static int tensor_filter_cpp_register (class tensor_filter_cpp *filter) final; /**< Register a C++ custom filter with this API if you want to register it at runtime. (or at the constructor of a shared object if you want to load it dynamically.) This should be invoked before initialized (constructed) by tensor_filter at run-time. */

};

#endif /* __cplusplus */

#endif /* __NNS_TENSOR_FITLER_CPP_H__ */

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
 * @file	tensor_filter_cpp.hh
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
 * @details
 *    This is a subplugin for C++ custom filters.
 *    If you want to write a wrapper for neural network frameworks
 *    or a hardware adaptor in C++, this is not for you.
 *    If you want to attach a single C++ class/object as a filter
 *    (a.k.a. custom filter) to a pipeline, this is what you want.
 *
 * To Packagers:
 *
 * This is to be exposed with "nnstreamer-c++-dev"
 *
 * @details Usage examples
 *
 *          Case 1: myfilter class implemented in the application
 *          - class myfilter : public tensor_filter_cpp { ... };
 *          - myfilter fx("myfilter01");
 *          - fx.register():
 *          - gst pipeline with ( ... ! tensor_filter framework=cpp
 *                                      model=myfilter01 ! ... );
 *
 *          Case 2: class exists in abc.so
 *          - gst pipeline with ( ... ! tensor_filter framework=cpp
 *                                      model=myfilter,abc.so ! ... );
 *          - // if myfilter already exists, abc.so is not loaded
 *          - in abc.so,
 *          - - class myfilter : public tensor_filter_cpp { ... };
 *          - - so's init() { myfilter fx("myfilter");
 *                            fx.register(); }
 *
 *	User should aware that the model name of a filter should be
 *     unique across all the .so files used in a pipeline. Otherwise,
 *     "Already registered (-EINVAL)" may occur.
 *
 */
#ifndef __NNS_TENSOR_FITLER_CPP_H__
#define __NNS_TENSOR_FITLER_CPP_H__

#ifdef __cplusplus

#include <atomic>
#include <vector>
#include <stdint.h>
#include <unordered_map>
#include <nnstreamer_plugin_api_filter.h>

/**
 * @brief This allows to have a c++ class inserted as a filter in a neural network pipeline of nnstreamer
 * @note This is experimental.
 * @details
 *          The C++ custom filter writers are supposed to inherit
 *         tensor_filter_cpp class. They should NOT touch any private
 *         properties.
 */
class tensor_filter_cpp {
  private:
    const uint32_t validity;
    const char *name; /**< The name of this C++ custom filter, searchable with "model" property */

    std::atomic_uint ref_count;
    static std::unordered_map<std::string, tensor_filter_cpp*> filters;
    static std::vector<void *> handles;
    static bool close_all_called;

  protected:
    const GstTensorFilterProperties *prop;

  public:
    tensor_filter_cpp(const char *modelName); /**< modelName is the model property of tensor_filter, which could be the path to the model file (requires proper extension name) or the registered model name at runtime. */
    virtual ~tensor_filter_cpp();

    /** C++ plugin writers need to fill {getInput/Output} inclusive-or {setInput} */
    virtual int getInputDim(GstTensorsInfo *info) = 0;
    virtual int getOutputDim(GstTensorsInfo *info) = 0;

    virtual int setInputDim(const GstTensorsInfo *in, GstTensorsInfo *out) = 0;

    virtual int invoke(const GstTensorMemory *in, GstTensorMemory *out) = 0;

    virtual bool isAllocatedBeforeInvoke() = 0;
      /**< return true if you want nnstreamer to preallocate output buffers
           before calling invoke. This value should be configured at the
           constructor and cannot be changed afterwards.
           This should not change its return values. */

    /** API. Do not override. */
    static int __register(class tensor_filter_cpp *filter, unsigned int ref_count = 0);
      /**< Register a C++ custom filter with this API if you want to register
           it at runtime or at the constructor of a shared object if you want
           to load it dynamically. This should be invoked before initialized
           (constructed) by tensor_filter at run-time.
           If you don't want to touch initial ref_count, keep it 0.
      */
    /** @brief Register this instance (same effect with __register) */
    int _register(unsigned int ref_count = 0) {
      return __register(this, ref_count);
    }
    static int __unregister(const char *name);
      /**< Unregister the run-time registered c++ filter.
           Do not call this to filters loaded as a .so (independent)
           filter. */
    /** @brief Unregister this instance (same effect with __unregister */
    int _unregister() {
      return __unregister(this->name);
    }

    bool isValid();
      /**< Check if it's a valid filter_cpp object. Do not override. */

    /** Internal functions for tensor_filter main. Do not override */
    static int getInputDim (const GstTensorFilterProperties *prop, void **private_data, GstTensorsInfo *info);
    static int getOutputDim (const GstTensorFilterProperties *prop, void **private_data, GstTensorsInfo *info);
    static int setInputDim (const GstTensorFilterProperties *prop, void **private_data, const GstTensorsInfo *in, GstTensorsInfo *out);
    static int invoke (const GstTensorFilterProperties *prop, void **private_data, const GstTensorMemory *input, GstTensorMemory *output);
    static int open (const GstTensorFilterProperties *prop, void **private_data);
    static void close (const GstTensorFilterProperties *prop, void **private_data);
    static void close_all_handles (void);
};

#endif /* __cplusplus */

#endif /* __NNS_TENSOR_FITLER_CPP_H__ */

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
 * @file	tensor_filter_cpp.cc
 * @date	26 Sep 2019
 * @brief	Tensor_filter subplugin for C++ custom filters.
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * @note This is experimental. The API and Class definition is NOT stable.
 *       If you want to write an OpenCV custom filter for nnstreamer, this is a good choice.
 *
 */
#include <errno.h>
#include <string.h>
#include <glib.h>

#include <tensor_common.h>
#include <nnstreamer_plugin_api_filter.h>

#include "tensor_filter_cpp.h"

void init_filter_cpp (void) __attribute__ ((constructor));
void fini_filter_cpp (void) __attribute__ ((destructor));

#define loadClass(name, ptr) \
  class tensor_filter_cpp *name = (tensor_filter_cpp *) *(ptr); \
  g_assert (*(ptr)); \
  g_assert (name->isValid());

/**
 * @brief Class constructor
 */
tensor_filter_cpp::tensor_filter_cpp(const char *name): validity(0xdeafdead), name(g_strdup(name))
{
}

/**
 * @brief Class destructor
 */
tensor_filter_cpp::~tensor_filter_cpp()
{
  g_free((gpointer) name);
}

/**
 * @brief Check if the given object (this) is valid.
 */
bool tensor_filter_cpp::isValid()
{
  return this->validity == 0xdeafdead;
}

/**
 * @brief Standard tensor_filter callback
 */
static int cpp_getInputDim (const GstTensorFilterProperties *prop, void **private_data, GstTensorsInfo *info)
{
  loadClass(cpp, private_data);
  return cpp->getInputDim (info);
}

/**
 * @brief Standard tensor_filter callback
 */
static int cpp_getOutputDim (const GstTensorFilterProperties *prop, void **private_data, GstTensorsInfo *info)
{
  loadClass(cpp, private_data);
  return cpp->getOutputDim (info);
}

/**
 * @brief Standard tensor_filter callback
 */
static int cpp_setInputDim (const GstTensorFilterProperties *prop, void **private_data, const GstTensorsInfo *in, GstTensorsInfo *out)
{
  loadClass(cpp, private_data);
  return cpp->setInputDim (in, out);
}

/**
 * @brief Standard tensor_filter callback
 */
static int cpp_invoke (const GstTensorFilterProperties *prop, void **private_data, const GstTensorMemory *input, GstTensorMemory *output)
{
  loadClass(cpp, private_data);
  return cpp->invoke(input, output);
}

/**
 * @brief Standard tensor_filter callback
 */
static int cpp_open (const GstTensorFilterProperties *prop, void **private_data)
{
  /** @todo: load the class with the given name (either file path or registered name) */
  /** @todo: configure values of NNS_support_cpp (allocate_in_invoke) */
  return -EPERM; /** NYI */
}

/**
 * @brief Standard tensor_filter callback
 */
static void cpp_close (const GstTensorFilterProperties *prop, void **private_data)
{
  /** @todo Implement This! NYI! */
  g_assert(false); /** NYI */
}

static gchar filter_subplugin_cpp[] = "cpp";

static GstTensorFilterFramework NNS_support_cpp = {
  .name = filter_subplugin_cpp,
  .allow_in_place = FALSE,      /** @todo: support this to optimize performance later. */
  .allocate_in_invoke = FALSE,
  .run_without_model = FALSE,
  .invoke_NN = cpp_invoke,
  .getInputDimension = cpp_getInputDim,
  .getOutputDimension = cpp_getOutputDim,
  .setInputDimension = cpp_setInputDim,
  .open = cpp_open,
  .close = cpp_close,
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_cpp (void)
{
  nnstreamer_filter_probe (&NNS_support_cpp);
}

/** @brief Destruct the subplugin */
void
fini_filter_cpp (void)
{
  nnstreamer_filter_exit (NNS_support_cpp.name);
}

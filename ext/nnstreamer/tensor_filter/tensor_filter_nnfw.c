/**
 * GStreamer Tensor_Filter, Tizen NNFW Module
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
 * @file	tensor_filter_nnfw.c
 * @date	24 Sep 2019
 * @brief	Tizen-NNFW module for tensor_filter gstreamer plugin
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (Tizen nnfw) for tensor_filter.
 *
 * @todo Check if nnfw supports dynamic input dimension. (if so, we need to supply setInputDim)
 * @todo Decide whether to let nnfw allocate output buffers or we will feed output buffers to nnfw.
 *
 */

#include <glib.h>
#include <tensor_common.h>
#include <nnstreamer_plugin_api_filter.h>

#include <nnfw.h>

void init_filter_nnfw (void) __attribute__ ((constructor));
void fini_filter_nnfw (void) __attribute__ ((destructor));

/**
 * @brief The standard tensor_filter callback
 */
static int nnfw_open (const GstTensorFilterProperties * prop,
    void **private_data)
{
  return 0;
}

/**
 * @brief The standard tensor_filter callback
 */
static void nnfw_close (const GstTensorFilterProperties * prop,
    void **private_data)
{
}

/**
 * @brief The standard tensor_filter callback
 */
static int nnfw_getInputDim (const GstTensorFilterProperties * prop,
      void **private_data, GstTensorsInfo * info)
{
  return 0;
}

/**
 * @brief The standard tensor_filter callback
 */
static int nnfw_getOutputDim (const GstTensorFilterProperties * prop,
      void **private_data, GstTensorsInfo * info)
{
  return 0;
}

/**
 * @brief The standard tensor_filter callback
 */
static int nnfw_invoke (const GstTensorFilterProperties * prop,
    void **private_data, const GstTensorMemory * input,
    GstTensorMemory * output)
{
  return 0;
}

static gchar filter_subplugin_nnfw[] = "nnfw";

static GstTensorFilterFramework NNS_support_nnfw = {
  .name = filter_subplugin_nnfw,
  .allow_in_place = FALSE,
  .allocate_in_invoke = FALSE,
  .run_without_model = FALSE,
  .invoke_NN = nnfw_invoke,
  .getInputDimension = nnfw_getInputDim,
  .getOutputDimension = nnfw_getOutputDim,
  .open = nnfw_open,
  .close = nnfw_close,
};

/**@brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_nnfw (void)
{
  nnstreamer_filter_probe (&NNS_support_nnfw);
}

/** @brief Destruct the subplugin */
void
fini_filter_nnfw (void)
{
  nnstreamer_filter_exit (NNS_support_nnfw.name);
}

/**
 * A Tensor Filter Extension for NCSDK Ver.2 (Intel Movidius Neural Compute Stick)
 * Copyright (C) 2019 Wook Song <wook16.song@samsung.com>
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All rights reserved.
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
 * @file    tensor_filter_movidius_ncsdk2.c
 * @date    13 May 2019
 * @brief   NCSDK2 module for tensor_filter gstreamer plugin
 * @see     http://github.com/nnsuite/nnstreamer
 * @author  Wook16.song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (Intel Movidius NCSDK2) for tensor_filter.
 * TODO: Fill in "GstTensorFilterFramework" for tensor_filter.h/c
 *
 */

#include <glib.h>
#include <mvnc2/mvnc.h>
#include <nnstreamer_plugin_api_filter.h>

void init_filter_mvncsdk2 (void) __attribute__ ((constructor));
void fini_filter_mvncsdk2 (void) __attribute__ ((destructor));

/**
 * @brief internal data of mvncsdk2
 */
typedef struct _mvncsdk2_data
{
  void *private_data;
} mvncsdk2_data;

/**
 * @brief Free privateData and move on.
 */
static void
_mvncsdk2_close (const GstTensorFilterProperties * prop, void **private_data)
{
  return;
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param prop : property of tensor_filter instance
 * @param private_data : movidius-ncsdk2  plugin's private data
 * @todo : fill this function
 */
static int
_mvncsdk2_open (const GstTensorFilterProperties * prop, void **private_data)
{
  return 0;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param prop : property of tensor_filter instance
 * @param private_data : movidius-ncsdk2 plugin's private data
 * @param[in] input : The array of input tensors
 * @param[out] output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 * @todo : fill this function
 */
static int
_mvncsdk2_invoke (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  return 0;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop : property of tensor_filter instance
 * @param private_data : movidius-ncsdk2 plugin's private data
 * @param[out] info : The dimesions and types of input tensors
 * @todo : fill this function
 */
static int
_mvncsdk2_getInputDim (const GstTensorFilterProperties * prop, void **private_data,
    GstTensorsInfo * info)
{
  return 0;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop : property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @param[out] info : The dimesions and types of output tensors
 * @todo : fill this function
 */
static int
_mvncsdk2_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  return 0;
}

static gchar filter_subplugin_movidius_ncsdk2[] = "movidius-ncsdk2";

static GstTensorFilterFramework NNS_support_movidius_ncsdk2 = {
  .name = filter_subplugin_movidius_ncsdk2,
  .allow_in_place = FALSE,
  .allocate_in_invoke = FALSE,
  .invoke_NN = _mvncsdk2_invoke,
  .getInputDimension = _mvncsdk2_getInputDim,
  .getOutputDimension = _mvncsdk2_getOutputDim,
  .open = _mvncsdk2_open,
  .close = _mvncsdk2_close,
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_mvncsdk2 (void)
{
  nnstreamer_filter_probe (&NNS_support_movidius_ncsdk2);
}

/** @brief Destruct the subplugin */
void
fini_filter_mvncsdk2 (void)
{
  nnstreamer_filter_exit (NNS_support_movidius_ncsdk2.name);
}

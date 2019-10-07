/**
 * GStreamer Tensor_Filter, PyTorch Module
 * Copyright (C) 2019 Parichay Kapoor <pk.kapoor@samsung.com>
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
 * @file	tensor_filter_pytorch.c
 * @date	24 April 2019
 * @brief	PyTorch module for tensor_filter gstreamer plugin
 * @see   http://github.com/nnsuite/nnstreamer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug   No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (pytorch) for tensor_filter.
 * Fill in "GstTensorFilterFramework" for tensor_filter.h/c
 *
 */
#include <glib.h>
#include <string.h>

#include "nnstreamer_conf.h"
#include "tensor_filter_pytorch_core.h"

void init_filter_torch (void) __attribute__ ((constructor));
void fini_filter_torch (void) __attribute__ ((destructor));

/**
 * @brief internal data of pytorch
 */
struct _Torch_data
{
  void *torch_private_data;
};
typedef struct _Torch_data torch_data;


/**
 * @brief Free privateData and move on.
 */
static void
torch_close (const GstTensorFilterProperties * prop, void **private_data)
{
  torch_data *torch;
  torch = *private_data;
  torch_core_delete (torch->torch_private_data);
  g_free (torch);
  *private_data = NULL;
}

/**
 * @brief Load pytorch modelfile
 * @param prop property of tensor_filter instance
 * @param private_data : pytorch plugin's private data
 * @return 0 if successfully loaded. 1 if skipped (already loaded).
 *        -1 if the object construction is failed.
 *        -2 if the object initialization if failed
 */
static gint
torch_loadModelFile (const GstTensorFilterProperties * prop,
    void **private_data)
{
  torch_data *torch;
  gboolean torch_use_gpu;
  if (*private_data != NULL) {
    torch = *private_data;
    if (strcmp (prop->model_file,
            torch_core_getModelPath (torch->torch_private_data))) {
      torch_close (prop, private_data);
    } else {
      return 1;
    }
  }

  torch_use_gpu = nnsconf_get_custom_value_bool ("pytorch", "enable_use_gpu",
      FALSE);

  torch = g_new0 (torch_data, 1);
  *private_data = torch;
  torch->torch_private_data = torch_core_new (prop->model_file);

  if (torch->torch_private_data) {
    if (torch_core_init (torch->torch_private_data, prop, torch_use_gpu)) {
      g_printerr ("failed to initialize the object: PyTorch");
      return -2;
    }
    return 0;
  } else {
    g_printerr ("failed to create the object: PyTorch");
    return -1;
  }
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param prop property of tensor_filter instance
 * @param private_data : pytorch plugin's private data
 */
static gint
torch_open (const GstTensorFilterProperties * prop, void **private_data)
{
  gint ret = torch_loadModelFile (prop, private_data);
  g_assert (ret >= 0);       /** This must be called only once */
  return ret;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : pytorch plugin's private data
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
static gint
torch_invoke (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  gint retval;
  torch_data *torch;
  torch = *private_data;
  g_assert (*private_data);
  retval = torch_core_invoke (torch->torch_private_data, input, output);
  g_assert (retval == 0);
  return retval;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : pytorch plugin's private data
 * @param[out] info The dimesions and types of input tensors
 */
static gint
torch_getInputDim (const GstTensorFilterProperties * prop, void **private_data,
    GstTensorsInfo * info)
{
  torch_data *torch;
  torch = *private_data;
  g_assert (*private_data);
  return torch_core_getInputDim (torch->torch_private_data, info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : pytorch plugin's private data
 * @param[out] info The dimesions and types of output tensors
 */
static gint
torch_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  torch_data *torch;
  torch = *private_data;
  g_assert (*private_data);
  return torch_core_getOutputDim (torch->torch_private_data, info);
}

static gchar filter_subplugin_pytorch[] = "pytorch";

static GstTensorFilterFramework NNS_support_pytorch = {
  .name = filter_subplugin_pytorch,
  .allow_in_place = FALSE,
  .allocate_in_invoke = FALSE,
  .invoke_NN = torch_invoke,
  .getInputDimension = torch_getInputDim,
  .getOutputDimension = torch_getOutputDim,
  .open = torch_open,
  .close = torch_close,
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_torch (void)
{
  nnstreamer_filter_probe (&NNS_support_pytorch);
}

/** @brief Destruct the subplugin */
void
fini_filter_torch (void)
{
  nnstreamer_filter_exit (NNS_support_pytorch.name);
}

/**
 * GStreamer Tensor_Filter, Python Module
 * Copyright (C) 2019 Dongju Chae <dongju.chae@samsung.com>
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
 * @file	tensor_filter_python.c
 * @date	10 Apr 2019
 * @brief	python module for tensor_filter gstreamer plugin
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	Dongju Chae <dongju.chae@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#define NO_IMPORT_ARRAY
#include "tensor_filter_python_core.h"
#include <nnstreamer_plugin_api_filter.h>
#include <glib.h>
#include <string.h>
#include <nnstreamer_conf.h>

/**
 * @brief internal data of python
 */
struct _py_data
{
  void *py_private_data;
};
typedef struct _py_data py_data;

static GstTensorFilterFramework *NNS_support_python;

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param prop: property of tensor_filter instance
 * @param private_data : python plugin's private data
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 */
static int
py_run (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  py_data *py = *private_data;

  g_assert (py);

  return py_core_run (py->py_private_data, input, output);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param[in] data The data element.
 */
static void
py_destroyNotify (void *data)
{
  py_core_destroyNotify (data);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param[in] prop read-only property values
 * @param[in/out] private_data python plugin's private data
 * @param[in] in_info structure of input tensor info
 * @param[out] out_info structure of output tensor info
 */
static int
py_setInputDim (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  py_data *py = *private_data;

  g_assert (py);

  return py_core_setInputDim (py->py_private_data, in_info, out_info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop: property of tensor_filter instance
 * @param private_data : python plugin's private data
 * @param[out] info The dimesions and types of input tensors
 */
static int
py_getInputDim (const GstTensorFilterProperties * prop, void **private_data,
    GstTensorsInfo * info)
{
  py_data *py = *private_data;

  g_assert (py);

  return py_core_getInputDim (py->py_private_data, info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop: property of tensor_filter instance
 * @param private_data : python plugin's private data
 * @param[out] info The dimesions and types of output tensors
 */
static int
py_getOutputDim (const GstTensorFilterProperties * prop, void **private_data,
    GstTensorsInfo * info)
{
  py_data *py = *private_data;

  g_assert (py);
  
  return py_core_getOutputDim (py->py_private_data, info);
}

/**
 * @brief Free privateData and move on.
 */
static void
py_close (const GstTensorFilterProperties * prop, void **private_data)
{
  py_data *py = *private_data;

  g_assert (py);
  py_core_delete (py->py_private_data);
  g_free (py);

  *private_data = NULL;
}

/**
 * @brief Load python model
 * @param prop: property of tensor_filter instance
 * @param private_data : python plugin's private data
 * @return 0 if successfully loaded. 1 if skipped (already loaded).
 *        -1 if the object construction is failed.
 *        -2 if the object initialization if failed
 */
static int
py_loadScriptFile (const GstTensorFilterProperties * prop, void **private_data)
{
  /**
   * prop->model_file contains the path of a python script
   * prop->custom contains its arguments seperated by ' '
   */
  py_data *py;

  if (*private_data != NULL) {
    py = *private_data;
    if (strcmp (prop->model_file, py_core_getScriptPath (py->py_private_data))) {
      py_close (prop, private_data);
    } else {
      return 1;
    }
  }

  py = g_new0 (py_data, 1); /** initialize py Fill Zero! */
  *private_data = py;
  py->py_private_data = py_core_new (prop->model_file, prop->custom_properties);

  if (py->py_private_data) {
    if (py_core_init (py->py_private_data, prop)) {
      g_printerr ("failed to initailize the object: python");
      return -2;
    }
    
    /** check methods in python script */
    cb_type type = py_core_getCbType (py->py_private_data);
    switch (type) {
      case CB_SETDIM:
        NNS_support_python->getInputDimension = NULL;
        NNS_support_python->getOutputDimension = NULL;
        NNS_support_python->setInputDimension = &py_setInputDim;
        break;
      case CB_GETDIM:
        NNS_support_python->getInputDimension = &py_getInputDim;
        NNS_support_python->getOutputDimension = &py_getOutputDim;
        NNS_support_python->setInputDimension = NULL;
        break;
      default:
        g_printerr ("Wrong callback type");
        return -2;
    }

    return 0;
  } else {
    g_printerr ("failed to create the object: python");
    return -1;
  }
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param prop: property of tensor_filter instance
 * @param private_data : python plugin's private data
 */
static int
py_open (const GstTensorFilterProperties * prop, void **private_data)
{
  return py_loadScriptFile (prop, private_data);
}

GstTensorFilterFramework _NNS_support_python = {
#if PY_VERSION_HEX >= 0x03000000
  .name = "python3",
#else
  .name = "python2",
#endif
  .allow_in_place = FALSE,      /** @todo: support this to optimize performance later. */
  .allocate_in_invoke = TRUE,
  .invoke_NN = py_run,
  .destroyNotify = py_destroyNotify,
  /** dimension-related callbacks are dynamically assigned */
  .getInputDimension = py_getInputDim,
  .getOutputDimension = py_getOutputDim,
  .setInputDimension = py_setInputDim,
  .open = py_open,
  .close = py_close,
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
__attribute__ ((constructor))
     void init_filter_py (void)
{
  NNS_support_python = &_NNS_support_python;
  tensor_filter_probe (NNS_support_python);
}

/** @brief Destruct the subplugin */
__attribute__ ((destructor))
     void fini_filter_py (void)
{
  tensor_filter_exit (NNS_support_python->name);
}

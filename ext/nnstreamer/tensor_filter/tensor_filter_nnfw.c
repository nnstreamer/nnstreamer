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
 * @todo NYI: configuring backends
 *
 */

#include <string.h>
#include <glib.h>
#include <tensor_common.h>
#include <nnstreamer_plugin_api_filter.h>

#include <nnfw.h>

void init_filter_nnfw (void) __attribute__ ((constructor));
void fini_filter_nnfw (void) __attribute__ ((destructor));

typedef struct
{
  nnfw_tensorinfo i_in;
  nnfw_tensorinfo i_out;
  nnfw_session *session;
  gchar *model_path;
} nnfw_pdata;

static void nnfw_close (const GstTensorFilterProperties * prop,
    void **private_data);

/**
 * @brief The standard tensor_filter callback
 */
static int
nnfw_open (const GstTensorFilterProperties * prop, void **private_data)
{
  NNFW_STATUS status;
  int err = 0;
  nnfw_pdata *pdata;

  if (*private_data != NULL) {
    pdata = *private_data;
    if (g_strcmp0 (prop->model_file, pdata->model_path)) {
      nnfw_close (prop, private_data);  /* "reopen" */
    } else {
      return 1;
    }
  }

  pdata = g_new0 (nnfw_pdata, 1);
  if (pdata == NULL)
    return -ENOMEM;

  *private_data = (void *) pdata;

  status = nnfw_create_session (&pdata->session);
  if (status != NNFW_STATUS_NO_ERROR) {
    err = -EINVAL;
    g_printerr ("Cannot create nnfw-runtime session");
    goto unalloc_exit;
  }

  status = nnfw_load_model_from_file (pdata->session, prop->model_file);
  if (status != NNFW_STATUS_NO_ERROR) {
    err = -EINVAL;
    g_printerr ("Cannot load the model file: %s", prop->model_file);
    goto session_exit;
  }

  status = nnfw_prepare (pdata->session);
  if (status != NNFW_STATUS_NO_ERROR) {
    err = -EINVAL;
    g_printerr ("nnfw-runtime cannot prepare the session for %s",
        prop->model_file);
    goto session_exit;
  }

  pdata->model_path = g_strdup (prop->model_file);
  return 0;

session_exit:
  status = nnfw_close_session(pdata->session);
  if (status != NNFW_STATUS_NO_ERROR)
    g_printerr ("Closing the session just opened by %s has failed", __func__);
unalloc_exit:
  g_free (pdata);
  *private_data = NULL;
  return err;
}

/**
 * @brief The standard tensor_filter callback
 * @todo Determine if we need to do "assert" for close-failure.
 */
static void
nnfw_close (const GstTensorFilterProperties * prop, void **private_data)
{
  nnfw_pdata *pdata;
  pdata = *private_data;

  if (pdata && pdata->session) {
    NNFW_STATUS status = nnfw_close_session (pdata->session);

    if (status != NNFW_STATUS_NO_ERROR) {
      g_printerr ("cannot close nnfw-runtime session for %s",
          pdata->model_path);
    }
  } else {
    g_printerr ("nnfw_close called without proper nnfw_open");
    if (pdata == NULL)
      return;
  }
  pdata->session = NULL;

  g_free (pdata->model_path);
  pdata->model_path = NULL;

  g_free (pdata);
  *private_data = NULL;
}

/**
 * @brief The standard tensor_filter callback
 */
static int
nnfw_getInputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  return 0;
}

/**
 * @brief The standard tensor_filter callback
 */
static int
nnfw_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  return 0;
}

/**
 * @brief The standard tensor_filter callback
 */
static int
nnfw_invoke (const GstTensorFilterProperties * prop,
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

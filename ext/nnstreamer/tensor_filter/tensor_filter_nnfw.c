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
 * @todo NYI: support quant8,bool type
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
 * @brief Convert from nnfw type to gst tensor type
 * @param[in] nnfw_type type given in nnfw format
 * @param[out] type container to receive type in gst tensor format
 * @return 0 on sucess, errno on error
 */
static int
nnfw_tensor_type_to_gst (const NNFW_TYPE nnfw_type, tensor_type * type)
{
  int err = 0;

  switch (nnfw_type) {
    case NNFW_TYPE_TENSOR_FLOAT32:
      *type = _NNS_FLOAT32;
      break;
    case NNFW_TYPE_TENSOR_INT32:
      *type = _NNS_INT32;
      break;
    default:
      *type = _NNS_END;
      err = -EINVAL;
  }

  return err;
}


/**
 * @brief Copy nnfw format info of tensor to gst format info
 * @param[in] nnfw_info info give in gst format
 * @param[out] gst_info info container to receive tensor info in gst format
 * @return 0 on success, errno on failure
 */
static int
nnfw_tensor_info_copy (const nnfw_tensorinfo * nnfw_info,
    GstTensorInfo * gst_info)
{
  guint idx;
  int status;

  g_return_val_if_fail (gst_info != NULL, -EINVAL);
  g_return_val_if_fail (nnfw_info != NULL, -EINVAL);

  if (nnfw_info->rank > NNS_TENSOR_RANK_LIMIT)
    return -ERANGE;

  gst_info->name = NULL;
  if ((status = nnfw_tensor_type_to_gst (nnfw_info->dtype, &gst_info->type)) < 0)
    return status;

  for (idx = 0; idx < nnfw_info->rank; idx ++)
    *(gst_info->dimension + idx) = nnfw_info->dims[idx];

  for (idx = nnfw_info->rank; idx < NNS_TENSOR_RANK_LIMIT; idx ++)
    *(gst_info->dimension + idx) = 1;

  return 0;
}

/**
 * @brief get nnfw tensor info in gst format info format from private data
 * @param[in] pdata private data for nnfw opened instance
 * @param[in] is_input to get info about input/output
 * @param[out] info info of tensors give in gst format
 * @return 0 on success, errno on failure
 */
static int
nnfw_tensors_info_get (const nnfw_pdata *pdata, const gboolean is_input,
    GstTensorsInfo * info)
{
  NNFW_STATUS status;
  struct nnfw_tensorinfo nnfw_info_t;
  int err;
  guint idx;

  g_return_val_if_fail (info != NULL, -EINVAL);
  g_return_val_if_fail (pdata != NULL, -EINVAL);
  g_return_val_if_fail (pdata->session != NULL, -EINVAL);

  /** First get number of outputs */
  if (is_input)
    status = nnfw_input_size (pdata->session, &info->num_tensors);
  else
    status = nnfw_output_size (pdata->session, &info->num_tensors);
  if (status != NNFW_STATUS_NO_ERROR)
    return -EPERM;

  if (info->num_tensors > NNS_TENSOR_SIZE_LIMIT)
    return -ERANGE;

  /** Now fill each outputs */
  for (idx = 0; idx < info->num_tensors; idx ++) {
    if (is_input)
      status = nnfw_input_tensorinfo (pdata->session, idx, &nnfw_info_t);
    else
      status = nnfw_output_tensorinfo (pdata->session, idx, &nnfw_info_t);
    if (status != NNFW_STATUS_NO_ERROR)
      return -EINVAL;

    err = nnfw_tensor_info_copy (&nnfw_info_t, &info->info[idx]);
    if (err < 0)
      return err;
  }

  return 0;
}

/**
 * @brief The standard tensor_filter callback
 */
static int
nnfw_getInputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  nnfw_pdata *pdata;
  int err = 0;

  g_return_val_if_fail (private_data != NULL, -EINVAL);

  pdata = (nnfw_pdata *) *private_data;
  err = nnfw_tensors_info_get (pdata, TRUE, info);

  return err;
}

/**
 * @brief The standard tensor_filter callback
 */
static int
nnfw_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  nnfw_pdata *pdata;
  int err = 0;

  g_return_val_if_fail (private_data != NULL, -EINVAL);

  pdata = (nnfw_pdata *) *private_data;
  err = nnfw_tensors_info_get (pdata, FALSE, info);

  return err;
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

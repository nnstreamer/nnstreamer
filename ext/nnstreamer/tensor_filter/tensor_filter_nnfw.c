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
 * @todo NYI: support quant8,bool type
 *
 */

#include <string.h>
#include <glib.h>
#include <tensor_common.h>
#include <nnstreamer_plugin_api_filter.h>

#include <nnfw.h>

/** backends supported by nnfw */
#define NNFW_CPU_BACKEND  "cpu"
#define NNFW_GPU_BACKEND  "acl_cl"
#define NNFW_NEON_BACKEND "acl_neon"
#define NNFW_SRCN_BACKEND  "srcn"
#define NNFW_DEFAULT_BACKEND NNFW_CPU_BACKEND

/** Maximum rank allowed for tensor dimension */
#define NNFW_TENSOR_RANK_LIMIT 6

static const gchar *nnfw_accl_support[] = {
  ACCL_AUTO_STR,
  ACCL_DEFAULT_STR,
  ACCL_CPU_NEON_STR,
  ACCL_CPU_STR,
  ACCL_GPU_STR,
  ACCL_NPU_SRCN_STR,
  ACCL_NPU_STR,
  NULL
};

void init_filter_nnfw (void) __attribute__ ((constructor));
void fini_filter_nnfw (void) __attribute__ ((destructor));

/**
 * @brief private data structure for the nnfw framework
 */
typedef struct
{
  GstTensorsInfo in_info;
  GstTensorsInfo out_info;
  nnfw_session *session;
  gchar *model_file;
  accl_hw accelerator;
} nnfw_pdata;

static void nnfw_close (const GstTensorFilterProperties * prop,
    void **private_data);
static int nnfw_tensors_info_get (const nnfw_pdata *pdata,
    const gboolean is_input, GstTensorsInfo * info);
static int nnfw_tensor_type_from_gst (const tensor_type type,
    NNFW_TYPE * nnfw_type);

/**
 * @brief parse user given input to extract accelerator to be used by nnfw
 * @param[in] pdata nnfw private data
 * @param[in] accelerators user given input
 * @return accelerator configuration to be set for nnfw
 */
static const char *
nnfw_get_accelerator (nnfw_pdata * pdata, const char *accelerators)
{
  pdata->accelerator = parse_accl_hw (accelerators, nnfw_accl_support);

  switch (pdata->accelerator) {
    case ACCL_NPU:
      return NNFW_SRCN_BACKEND;
    case ACCL_NPU_SRCN:
      return NNFW_SRCN_BACKEND;
    case ACCL_CPU_NEON:
      return NNFW_NEON_BACKEND;
    case ACCL_GPU:
      return NNFW_GPU_BACKEND;
    case ACCL_CPU:
      return NNFW_CPU_BACKEND;
    case ACCL_DEFAULT:
      /** intended */
    default:
      return NNFW_DEFAULT_BACKEND;
  }
}

/**
 * @brief The standard tensor_filter callback
 */
static int
nnfw_open (const GstTensorFilterProperties * prop, void **private_data)
{
  NNFW_STATUS status;
  int err = 0;
  nnfw_pdata *pdata;
  char *model_path = NULL;
  char *meta_file = NULL;
  const char *accelerator = NULL;

  if (*private_data != NULL) {
    pdata = *private_data;
    if (g_strcmp0 (prop->model_files[0], pdata->model_file) != 0) {
      nnfw_close (prop, private_data);  /* "reopen" */
    } else {
      return 1;
    }
  }

  pdata = *private_data = g_new0 (nnfw_pdata, 1);
  if (pdata == NULL) {
    g_printerr ("Failed to allocate memory for filter subplugin.");
    return -ENOMEM;
  }

  status = nnfw_create_session (&pdata->session);
  if (status != NNFW_STATUS_NO_ERROR) {
    err = -EINVAL;
    g_printerr ("Cannot create nnfw-runtime session");
    goto unalloc_exit;
  }

  accelerator = nnfw_get_accelerator (pdata, prop->accl_str);
  status = nnfw_set_available_backends (pdata->session, accelerator);
  if (status != NNFW_STATUS_NO_ERROR) {
    err = -EINVAL;
    g_printerr ("Cannot set nnfw-runtime backend to %s", accelerator);
    goto unalloc_exit;
  }

  /** @note nnfw opens the first model listed in the MANIFEST file */
  model_path = g_path_get_dirname (prop->model_files[0]);
  meta_file = g_build_filename (model_path, "metadata", "MANIFEST", NULL);

  if (!g_file_test (prop->model_files[0], G_FILE_TEST_IS_REGULAR) ||
      !g_file_test (meta_file, G_FILE_TEST_IS_REGULAR)) {
    err = -EINVAL;
    g_printerr ("Model file (%s) or its metadata is not valid (not regular).",
        prop->model_files[0]);
    goto session_exit;
  }

  /** @todo open using model_file once nnfw works with it */
  status = nnfw_load_model_from_file (pdata->session, model_path);
  if (status != NNFW_STATUS_NO_ERROR) {
    err = -EINVAL;
    g_printerr ("Cannot load the model file: %s", prop->model_files[0]);
    goto session_exit;
  }

  status = nnfw_prepare (pdata->session);
  if (status != NNFW_STATUS_NO_ERROR) {
    err = -EINVAL;
    g_printerr ("nnfw-runtime cannot prepare the session for %s",
        prop->model_files[0]);
    goto session_exit;
  }

  err = nnfw_tensors_info_get (pdata, TRUE, &pdata->in_info);
  if (err) {
    g_printerr ("Error retrieving input info from nnfw-runtime.");
    goto session_exit;
  }

  err = nnfw_tensors_info_get (pdata, FALSE, &pdata->out_info);
  if (err) {
    g_printerr ("Error retrieving output info from nnfw-runtime.");
    goto session_exit;
  }

  pdata->model_file = g_strdup (prop->model_files[0]);
  g_free (meta_file);
  g_free (model_path);

  return 0;

session_exit:
  status = nnfw_close_session (pdata->session);
  if (status != NNFW_STATUS_NO_ERROR)
    g_printerr ("Closing the session just opened by %s has failed", __func__);
  g_free (meta_file);
  g_free (model_path);
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
          pdata->model_file);
    }
  } else {
    g_printerr ("nnfw_close called without proper nnfw_open");
    if (pdata == NULL)
      return;
  }
  pdata->session = NULL;

  g_free (pdata->model_file);
  pdata->model_file = NULL;

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
    case NNFW_TYPE_TENSOR_QUANT8_ASYMM:
      /** @todo: update this to NNFW_TYPE_TENSOR_UINT8 type once nnfw is updated */
      *type = _NNS_UINT8;
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
  gint idx;
  int status;

  g_return_val_if_fail (gst_info != NULL, -EINVAL);
  g_return_val_if_fail (nnfw_info != NULL, -EINVAL);

  if (nnfw_info->rank > NNS_TENSOR_RANK_LIMIT)
    return -ERANGE;

  gst_info->name = NULL;
  if ((status =
          nnfw_tensor_type_to_gst (nnfw_info->dtype, &gst_info->type)) < 0)
    return status;

  for (idx = nnfw_info->rank - 1; idx >= 0; idx--)
    gst_info->dimension[idx] = nnfw_info->dims[nnfw_info->rank - idx - 1];

  for (idx = NNS_TENSOR_RANK_LIMIT - 1; idx >= nnfw_info->rank; idx--)
    gst_info->dimension[idx] = 1;

  return 0;
}

/**
 * @brief register/set input tensor info with nnfw
 * @param[in] pdata private data for nnfw opened instance
 * @param[in] tensors_info info of tensors to be registered
 * @param[in] tensor_idx idx of the tensor to be registered
 * @return 0 on success, errno on failure
 */
static int
nnfw_tensor_info_set (const nnfw_pdata *pdata, const GstTensorsInfo * tensors_info,
    guint tensor_idx)
{
  struct nnfw_tensorinfo nnfw_info;
  gint err;
  gint idx;
  const GstTensorInfo *info = &tensors_info->info[tensor_idx];

  err = nnfw_tensor_type_from_gst (info->type, &nnfw_info.dtype);
  if (err)
    return err;

  nnfw_info.rank = gst_tensor_info_get_rank (info);

  /** reverse the order of dimension */
  for (idx = nnfw_info.rank-1; idx >= 0; idx--)
    nnfw_info.dims[nnfw_info.rank - idx - 1] = info->dimension[idx];

  for (idx = NNFW_TENSOR_RANK_LIMIT-1; idx >= nnfw_info.rank; idx--)
    nnfw_info.dims[idx] = 0;

  nnfw_apply_tensorinfo (pdata->session, tensor_idx, nnfw_info);

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
nnfw_tensors_info_get (const nnfw_pdata * pdata, const gboolean is_input,
    GstTensorsInfo * info)
{
  NNFW_STATUS status;
  struct nnfw_tensorinfo nnfw_info_t;
  int err;
  guint idx;

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
  for (idx = 0; idx < info->num_tensors; idx++) {
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

  g_return_val_if_fail (private_data != NULL, -EINVAL);
  pdata = (nnfw_pdata *) *private_data;

  g_return_val_if_fail (pdata != NULL, -EINVAL);
  g_return_val_if_fail (info != NULL, -EINVAL);

  gst_tensors_info_copy (info, &pdata->in_info);
  return 0;
}

/**
 * @brief The standard tensor_filter callback
 */
static int
nnfw_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  nnfw_pdata *pdata;

  g_return_val_if_fail (private_data != NULL, -EINVAL);
  pdata = (nnfw_pdata *) *private_data;

  g_return_val_if_fail (pdata != NULL, -EINVAL);
  g_return_val_if_fail (info != NULL, -EINVAL);

  gst_tensors_info_copy (info, &pdata->out_info);

  return 0;
}

/**
 * @brief The standard tensor_filter callback
 */
static int
nnfw_setInputDim (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  nnfw_pdata *pdata;
  int err, idx;
  GstTensorsInfo updated_info;

  g_return_val_if_fail (private_data != NULL, -EINVAL);
  g_return_val_if_fail (in_info != NULL, -EINVAL);
  g_return_val_if_fail (out_info != NULL, -EINVAL);

  pdata = (nnfw_pdata *) *private_data;
  g_return_val_if_fail (pdata != NULL, -EINVAL);

  if (in_info->num_tensors != pdata->in_info.num_tensors)
    return -EPERM;

  for (idx = 0; idx < pdata->in_info.num_tensors; idx ++) {
    err = nnfw_tensor_info_set (pdata, in_info, idx);
    if (err)
      goto error;
  }

  err = nnfw_tensors_info_get (pdata, TRUE, &updated_info);
  if (err || !gst_tensors_info_is_equal (in_info, &updated_info))
    goto error;

  err = nnfw_tensors_info_get (pdata, FALSE, out_info);
  if (err)
    goto error;

  gst_tensors_info_copy (&pdata->in_info, in_info);
  gst_tensors_info_copy (&pdata->out_info, out_info);

  return 0;

error:
  g_printerr ("Unable to set the provided input tensor info");
  /** Reset input dimensions */
  for (idx = 0; idx < pdata->in_info.num_tensors; idx ++) {
    nnfw_tensor_info_set (pdata, &pdata->in_info, idx);
  }

  return err;
}

/**
 * @brief Convert from gst tensor type to NNFW type
 * @param[in] type type given in gst format
 * @param[out] nnfw_type container to receive type in nnfw tensor format
 * @return 0 on sucess, negative errno on error
 */
static int
nnfw_tensor_type_from_gst (const tensor_type type, NNFW_TYPE * nnfw_type)
{
  int err = 0;

  switch (type) {
    case _NNS_FLOAT32:
      *nnfw_type = NNFW_TYPE_TENSOR_FLOAT32;
      break;
    case _NNS_INT32:
      *nnfw_type = NNFW_TYPE_TENSOR_INT32;
      break;
    case _NNS_UINT8:
      /** @todo: update this to NNFW_TYPE_TENSOR_UINT8 type once nnfw is updated */
      *nnfw_type = NNFW_TYPE_TENSOR_QUANT8_ASYMM;
      break;
    default:
      err = -EINVAL;
  }

  return err;
}

/**
 * @brief Set tensor memory information in nnfw for input/output
 * @param[in] prop Tensor-filter properties for this nnfw
 * @param[in] pdata nnfw private data
 * @param[in] mem Tensor memory containing input/output information
 * @param[in] is_input given memory is for input or output
 * @return 0 on sucess, negative errno on error
 */
static int
nnfw_tensor_memory_set (const GstTensorFilterProperties * prop,
    const nnfw_pdata * pdata, const GstTensorMemory * mem,
    const gboolean is_input)
{
  NNFW_TYPE nnfw_type;
  NNFW_STATUS nnfw_status;
  int err = 0;
  guint idx;
  unsigned int num_tensors = 0;

  g_return_val_if_fail (prop != NULL, -EINVAL);
  g_return_val_if_fail (mem != NULL, -EINVAL);
  g_return_val_if_fail (pdata != NULL, -EINVAL);
  g_return_val_if_fail (pdata->session != NULL, -EPERM);

  if (is_input)
    num_tensors = pdata->in_info.num_tensors;
  else
    num_tensors = pdata->out_info.num_tensors;

  for (idx = 0; idx < num_tensors; idx++) {
    err = nnfw_tensor_type_from_gst (mem[idx].type, &nnfw_type);
    if (err != 0)
      return err;

    if (is_input) {
      g_return_val_if_fail (mem[idx].size ==
          gst_tensor_info_get_size (&pdata->in_info.info[idx]), -EINVAL);
      nnfw_status = nnfw_set_input (pdata->session, idx, nnfw_type,
        mem[idx].data, mem[idx].size);
    } else {
      g_return_val_if_fail (mem[idx].size ==
          gst_tensor_info_get_size (&pdata->out_info.info[idx]), -EINVAL);
      nnfw_status = nnfw_set_output (pdata->session, idx, nnfw_type,
        mem[idx].data, mem[idx].size);
    }
    if (nnfw_status != NNFW_STATUS_NO_ERROR)
      return -EINVAL;
  }

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
  nnfw_pdata *pdata;
  int err = 0;
  NNFW_STATUS nnfw_status;

  g_return_val_if_fail (private_data != NULL, -EINVAL);
  pdata = (nnfw_pdata *) * private_data;

  err = nnfw_tensor_memory_set (prop, pdata, input, TRUE);
  if (err < 0)
    return err;

  err = nnfw_tensor_memory_set (prop, pdata, output, FALSE);
  if (err < 0)
    return err;

  nnfw_status = nnfw_run (pdata->session);
  if (nnfw_status != NNFW_STATUS_NO_ERROR)
    err = -EINVAL;

  return err;
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
  .setInputDimension = nnfw_setInputDim,
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

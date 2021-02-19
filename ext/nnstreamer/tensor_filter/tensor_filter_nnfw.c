/**
 * GStreamer Tensor_Filter, Tizen NNFW Module
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
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
 * @see		http://github.com/nnstreamer/nnstreamer
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
#include <glib-object.h>

#include <nnstreamer_log.h>
#include <tensor_common.h>
#define NO_ANONYMOUS_NESTED_STRUCT
#include <nnstreamer_plugin_api_filter.h>
#undef NO_ANONYMOUS_NESTED_STRUCT
#include <nnfw.h>

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief Internal structure for nnfw tensor info. The max size is NNS_TENSOR_SIZE_LIMIT.
 */
typedef struct
{
  guint num_tensors;
  nnfw_tensorinfo info[NNS_TENSOR_SIZE_LIMIT];
} nnfw_tinfo_s;

/** backends supported by nnfw */
#define NNFW_CPU_BACKEND  "cpu;bcq"
#define NNFW_GPU_BACKEND  "acl_cl"
#define NNFW_NEON_BACKEND "acl_neon"
#define NNFW_SRCN_BACKEND  "srcn"
#define NNFW_DEFAULT_BACKEND NNFW_CPU_BACKEND

static const gchar *nnfw_accl_support[] = {
  ACCL_CPU_NEON_STR,
  ACCL_CPU_STR,
  ACCL_GPU_STR,
  ACCL_NPU_SRCN_STR,
  ACCL_NPU_STR,
  NULL
};

#if defined(__aarch64__) || defined(__arm__)
static const gchar *nnfw_accl_auto = ACCL_CPU_NEON_STR;
#else
static const gchar *nnfw_accl_auto = ACCL_CPU_STR;
#endif
static const gchar *nnfw_accl_default = ACCL_CPU_STR;

void init_filter_nnfw (void) __attribute__ ((constructor));
void fini_filter_nnfw (void) __attribute__ ((destructor));

static GstTensorFilterFrameworkStatistics nnfw_internal_stats = {
  .total_invoke_num = 0,
  .total_invoke_latency = 0,
  .total_overhead_latency = 0,
};

/**
 * @brief private data structure for the nnfw framework
 */
typedef struct
{
  nnfw_tinfo_s in_info; /**< cached input tensor info */
  nnfw_tinfo_s out_info; /**< cached output tensor info */
  nnfw_session *session;
  gchar *model_file;
  gchar *accelerator;
} nnfw_pdata;

static void nnfw_close (const GstTensorFilterProperties * prop,
    void **private_data);
static int nnfw_tensors_info_get (const nnfw_pdata * pdata,
    const gboolean is_input, nnfw_tinfo_s * info);
static int nnfw_invoke_internal (const nnfw_pdata * pdata,
    const nnfw_tinfo_s * in_info, const nnfw_tinfo_s * out_info,
    const GstTensorMemory * input, GstTensorMemory * output);

/**
 * @brief parse user given input to extract accelerator to be used by nnfw
 * @param[in] pdata nnfw private data
 * @param[in] accelerators user given input
 * @return accelerator configuration to be set for nnfw
 */
static const char *
nnfw_get_accelerator (nnfw_pdata * pdata, const char *accelerators)
{
  accl_hw accel = parse_accl_hw (accelerators, nnfw_accl_support,
      nnfw_accl_auto, nnfw_accl_default);

  switch (accel) {
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
    default:
      return NNFW_DEFAULT_BACKEND;
  }
}

/**
 * @brief Parse accelerator and other option.
 */
static void
nnfw_parse_custom_option (const GstTensorFilterProperties * prop,
    nnfw_pdata * pdata)
{
  if (prop->custom_properties) {
    gchar **options;
    guint i, len;

    options = g_strsplit (prop->custom_properties, ",", -1);
    len = g_strv_length (options);

    for (i = 0; i < len; ++i) {
      gchar **option = g_strsplit (options[i], ":", -1);

      if (g_strv_length (option) > 1) {
        g_strstrip (option[0]);
        g_strstrip (option[1]);

        if (g_ascii_strcasecmp (option[0], "Runtime") == 0) {
          pdata->accelerator = g_strdup (option[1]);
        } else {
          nns_logw ("Unknown option (%s).", options[i]);
        }
      }

      g_strfreev (option);
    }

    g_strfreev (options);
  }

  /* set accelerator if custom option does not include accelerator */
  if (pdata->accelerator == NULL) {
    const char *accel = nnfw_get_accelerator (pdata, prop->accl_str);
    pdata->accelerator = g_strdup (accel);
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

  if (*private_data != NULL) {
    pdata = *private_data;
    if (g_strcmp0 (prop->model_files[0], pdata->model_file) != 0) {
      nnfw_close (prop, private_data);  /* "reopen" */
    } else {
      return 1;
    }
  }

  /* validate model file first */
  if (!g_file_test (prop->model_files[0], G_FILE_TEST_EXISTS)) {
    nns_loge ("Cannot find model file %s.", prop->model_files[0]);
    return -EINVAL;
  }

  pdata = *private_data = g_new0 (nnfw_pdata, 1);
  if (pdata == NULL) {
    nns_loge ("Failed to allocate memory for filter subplugin.\n");
    return -ENOMEM;
  }

  nnfw_parse_custom_option (prop, pdata);

  status = nnfw_create_session (&pdata->session);
  if (status != NNFW_STATUS_NO_ERROR) {
    err = -EINVAL;
    nns_loge ("Cannot create nnfw-runtime session\n");
    goto error_exit;
  }

  /** @todo NNFW now uses package path. Fix when file path is available to open NNFW. */
  if (g_file_test (prop->model_files[0], G_FILE_TEST_IS_DIR)) {
    model_path = g_strdup (prop->model_files[0]);
  } else {
    model_path = g_path_get_dirname (prop->model_files[0]);
  }

  status = nnfw_load_model_from_file (pdata->session, model_path);
  g_free (model_path);

  if (status != NNFW_STATUS_NO_ERROR) {
    err = -EINVAL;
    nns_loge ("Cannot load the model file: %s\n", prop->model_files[0]);
    goto error_exit;
  }

  status = nnfw_set_available_backends (pdata->session, pdata->accelerator);
  if (status != NNFW_STATUS_NO_ERROR) {
    err = -EINVAL;
    nns_loge ("Cannot set nnfw-runtime backend to %s\n", pdata->accelerator);
    goto error_exit;
  }

  status = nnfw_prepare (pdata->session);
  if (status != NNFW_STATUS_NO_ERROR) {
    err = -EINVAL;
    nns_loge ("nnfw-runtime cannot prepare the session for %s\n",
        prop->model_files[0]);
    goto error_exit;
  }

  err = nnfw_tensors_info_get (pdata, TRUE, &pdata->in_info);
  if (err) {
    nns_loge ("Error retrieving input info from nnfw-runtime.\n");
    goto error_exit;
  }

  err = nnfw_tensors_info_get (pdata, FALSE, &pdata->out_info);
  if (err) {
    nns_loge ("Error retrieving output info from nnfw-runtime.\n");
    goto error_exit;
  }

  pdata->model_file = g_strdup (prop->model_files[0]);
  return 0;

error_exit:
  nnfw_close (prop, private_data);
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

  if (private_data == NULL || *private_data == NULL)
    return;

  pdata = *private_data;

  if (pdata->session) {
    NNFW_STATUS status = nnfw_close_session (pdata->session);

    if (status != NNFW_STATUS_NO_ERROR) {
      nns_loge ("cannot close nnfw-runtime session for %s\n",
          pdata->model_file);
    }

    pdata->session = NULL;
  }

  g_free (pdata->model_file);
  pdata->model_file = NULL;

  g_free (pdata->accelerator);
  pdata->accelerator = NULL;

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
    case NNFW_TYPE_TENSOR_INT64:
      *type = _NNS_INT64;
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
    case _NNS_INT64:
      *nnfw_type = NNFW_TYPE_TENSOR_INT64;
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
 * @brief Convert nnfw format info of tensors to gst format info
 * @param[in] nnfw_info info given in nnfw format
 * @param[out] gst_info info container to receive tensor info in gst format
 * @return 0 on success, errno on failure
 */
static int
nnfw_convert_to_gst_info (const nnfw_tinfo_s * nnfw_info,
    GstTensorsInfo * gst_info)
{
  guint i;
  int status;

  gst_tensors_info_init (gst_info);

  for (i = 0; i < nnfw_info->num_tensors; i++) {
    const nnfw_tensorinfo *ninfo = &nnfw_info->info[i];
    GstTensorInfo *ginfo = &gst_info->info[i];
    gint idx;

    if (ninfo->rank > NNS_TENSOR_RANK_LIMIT)
      return -ERANGE;

    if ((status = nnfw_tensor_type_to_gst (ninfo->dtype, &ginfo->type)) != 0)
      return status;

    for (idx = ninfo->rank - 1; idx >= 0; idx--)
      ginfo->dimension[idx] = ninfo->dims[ninfo->rank - idx - 1];

    for (idx = NNS_TENSOR_RANK_LIMIT - 1; idx >= ninfo->rank; idx--)
      ginfo->dimension[idx] = 1;
  }

  gst_info->num_tensors = nnfw_info->num_tensors;
  return 0;
}

/**
 * @brief Internal function to set input tensor info.
 * @todo nnfw_apply_tensorinfo() will be deprecated. Use nnfw_set_input_tensorinfo() later (nnfw ver >= 1.6.0).
 */
static int
nnfw_set_input_info (const nnfw_pdata * pdata, guint idx,
    nnfw_tensorinfo * info)
{
  NNFW_STATUS status;

#if defined (NNFW_USE_OLD_API)
  status = nnfw_apply_tensorinfo (pdata->session, idx, *info);
#else
  status = nnfw_set_input_tensorinfo (pdata->session, idx, info);
#endif

  return (status != NNFW_STATUS_NO_ERROR) ? -EPERM : 0;
}

/**
 * @brief register/set input tensor info with nnfw
 * @param[in] pdata private data for nnfw opened instance
 * @param[in] tensors_info info of tensors to be registered
 * @param[in] tensor_idx idx of the tensor to be registered
 * @return 0 on success, errno on failure
 */
static int
nnfw_tensor_info_set (const nnfw_pdata * pdata,
    const GstTensorsInfo * tensors_info, guint tensor_idx)
{
  struct nnfw_tensorinfo nnfw_info;
  gint err;
  gint idx;
  const GstTensorInfo *info = &tensors_info->info[tensor_idx];

  err = nnfw_tensor_type_from_gst (info->type, &nnfw_info.dtype);
  if (err)
    return err;

  /**
   * We should handle proper rank value.
   * The rank returned from gst may be invalid, e.g. the case if batch is 1.
   */
  nnfw_info.rank = gst_tensor_info_get_rank (info);
  if (nnfw_info.rank < pdata->in_info.info[tensor_idx].rank)
    nnfw_info.rank = pdata->in_info.info[tensor_idx].rank;

  /** reverse the order of dimension */
  for (idx = nnfw_info.rank - 1; idx >= 0; idx--)
    nnfw_info.dims[nnfw_info.rank - idx - 1] = info->dimension[idx];

  /** @note Maximum rank expressible with nnfw is 6 (NNFW_MAX_RANK) */
  for (idx = NNFW_MAX_RANK - 1; idx >= nnfw_info.rank; idx--)
    nnfw_info.dims[idx] = 0;

  return nnfw_set_input_info (pdata, tensor_idx, &nnfw_info);
}

/**
 * @brief get nnfw tensor info in gst format info format from private data
 * @param[in] pdata private data for nnfw opened instance
 * @param[in] is_input to get info about input/output
 * @param[out] info info of tensors given in nnfw format
 * @return 0 on success, errno on failure
 */
static int
nnfw_tensors_info_get (const nnfw_pdata * pdata, const gboolean is_input,
    nnfw_tinfo_s * info)
{
  NNFW_STATUS status;
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
      status = nnfw_input_tensorinfo (pdata->session, idx, &info->info[idx]);
    else
      status = nnfw_output_tensorinfo (pdata->session, idx, &info->info[idx]);
    if (status != NNFW_STATUS_NO_ERROR)
      return -EINVAL;
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
  pdata = (nnfw_pdata *) * private_data;

  g_return_val_if_fail (pdata != NULL, -EINVAL);
  g_return_val_if_fail (info != NULL, -EINVAL);

  return nnfw_convert_to_gst_info (&pdata->in_info, info);
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
  pdata = (nnfw_pdata *) * private_data;

  g_return_val_if_fail (pdata != NULL, -EINVAL);
  g_return_val_if_fail (info != NULL, -EINVAL);

  return nnfw_convert_to_gst_info (&pdata->out_info, info);
}

/**
 * @brief Internal function to invoke with dummy data.
 * When changing the input shape, NNFW will update the output shape after the invoke process is done.
 */
static gboolean
nnfw_invoke_dummy (const nnfw_pdata * pdata, const nnfw_tinfo_s * in_info,
    const nnfw_tinfo_s * out_info)
{
  GstTensorsInfo gst_in_info, gst_out_info;
  GstTensorMemory input[NNS_TENSOR_SIZE_LIMIT] = { 0, };
  GstTensorMemory output[NNS_TENSOR_SIZE_LIMIT] = { 0, };
  gboolean failed = FALSE;
  guint i, retry;
  int err;

  if (nnfw_convert_to_gst_info (in_info, &gst_in_info) != 0 ||
      nnfw_convert_to_gst_info (out_info, &gst_out_info) != 0) {
    nns_loge ("Failed to convert nnfw info.");
    return FALSE;
  }

  for (i = 0; i < gst_in_info.num_tensors; ++i) {
    input[i].size = gst_tensor_info_get_size (&gst_in_info.info[i]);
    input[i].data = g_malloc0 (input[i].size);
  }

  /* The output shape would be changed, set enough size for output buffer. */
  for (i = 0; i < gst_out_info.num_tensors; ++i) {
    output[i].size = gst_tensor_info_get_size (&gst_out_info.info[i]) * 2;
    output[i].data = g_malloc0 (output[i].size);
  }

  retry = 0;
  while ((err = nnfw_invoke_internal (pdata, in_info, out_info, input, output)) != 0) {
    if (err != -EAGAIN) {
      nns_loge ("Invoke failed, cannot update input info.");
      failed = TRUE;
      break;
    }

    nns_logw ("Invoke failed, reallocate output tensors and retry (%u).",
        ++retry);

    for (i = 0; i < gst_out_info.num_tensors; ++i) {
      output[i].size *= 2;
      g_free (output[i].data);
      output[i].data = g_malloc0 (output[i].size);
    }
  }

  for (i = 0; i < gst_in_info.num_tensors; ++i) {
    g_free (input[i].data);
    input[i].data = NULL;
  }
  for (i = 0; i < gst_out_info.num_tensors; ++i) {
    g_free (output[i].data);
    output[i].data = NULL;
  }

  return !failed;
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
  nnfw_tinfo_s nnfw_in_info, nnfw_out_info;
  GstTensorsInfo gst_in_info;

  g_return_val_if_fail (private_data != NULL, -EINVAL);
  g_return_val_if_fail (in_info != NULL, -EINVAL);
  g_return_val_if_fail (out_info != NULL, -EINVAL);

  pdata = (nnfw_pdata *) * private_data;
  g_return_val_if_fail (pdata != NULL, -EINVAL);

  if (in_info->num_tensors != pdata->in_info.num_tensors)
    return -EPERM;

  for (idx = 0; idx < pdata->in_info.num_tensors; idx++) {
    err = nnfw_tensor_info_set (pdata, in_info, idx);
    if (err)
      goto error;
  }

  err = nnfw_tensors_info_get (pdata, TRUE, &nnfw_in_info);
  if (err)
    goto error;

  err = nnfw_convert_to_gst_info (&nnfw_in_info, &gst_in_info);
  if (err || !gst_tensors_info_is_equal (in_info, &gst_in_info))
    goto error;

  /* Invoke with dummy. NNFW updates output info after the invoke is done. */
  if (!nnfw_invoke_dummy (pdata, &nnfw_in_info, &pdata->out_info)) {
    nns_loge ("Failed to invoke the model with changed input info.");
    goto error;
  }

  err = nnfw_tensors_info_get (pdata, FALSE, &nnfw_out_info);
  if (err)
    goto error;

  /* Fill output info and update it in pdata. */
  err = nnfw_convert_to_gst_info (&nnfw_out_info, out_info);
  if (err)
    goto error;

  memcpy (&pdata->in_info, &nnfw_in_info, sizeof (nnfw_tinfo_s));
  memcpy (&pdata->out_info, &nnfw_out_info, sizeof (nnfw_tinfo_s));
  return 0;

error:
  nns_loge ("Unable to set the provided input tensor info\n");
  /** Reset input dimensions */
  for (idx = 0; idx < pdata->in_info.num_tensors; idx++)
    nnfw_set_input_info (pdata, idx, &pdata->in_info.info[idx]);

  nnfw_invoke_dummy (pdata, &pdata->in_info, &pdata->out_info);
  return err;
}

/**
 * @brief Set tensor memory information in nnfw for input/output
 * @param[in] pdata nnfw private data
 * @param[in] mem Tensor memory containing input/output information
 * @param[in] info Tensor information in nnfw format
 * @param[in] is_input given memory is for input or output
 * @return 0 on sucess, negative errno on error
 */
static int
nnfw_tensor_memory_set (const nnfw_pdata * pdata, const GstTensorMemory * mem,
    const nnfw_tinfo_s * info, const gboolean is_input)
{
  NNFW_STATUS nnfw_status;
  guint idx;

  g_return_val_if_fail (G_UNLIKELY (pdata != NULL), -EINVAL);
  g_return_val_if_fail (G_UNLIKELY (mem != NULL && info != NULL), -EINVAL);
  g_return_val_if_fail (G_UNLIKELY (pdata->session != NULL), -EPERM);

  for (idx = 0; idx < info->num_tensors; idx++) {
    if (is_input) {
      nnfw_status = nnfw_set_input (pdata->session, idx,
          info->info[idx].dtype, mem[idx].data, mem[idx].size);
    } else {
      nnfw_status = nnfw_set_output (pdata->session, idx,
          info->info[idx].dtype, mem[idx].data, mem[idx].size);
    }
    if (nnfw_status != NNFW_STATUS_NO_ERROR)
      return -EINVAL;
  }

  return 0;
}

/**
 * @brief Internal function to run nnfw session.
 */
static int
nnfw_invoke_internal (const nnfw_pdata * pdata,
    const nnfw_tinfo_s * in_info, const nnfw_tinfo_s * out_info,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  int64_t start_time, stop_time;
  int err;
  NNFW_STATUS status;

  start_time = g_get_monotonic_time ();

  err = nnfw_tensor_memory_set (pdata, input, in_info, TRUE);
  if (G_UNLIKELY (err < 0))
    return err;

  err = nnfw_tensor_memory_set (pdata, output, out_info, FALSE);
  if (G_UNLIKELY (err < 0))
    return err;

  stop_time = g_get_monotonic_time ();
  nnfw_internal_stats.total_overhead_latency += stop_time - start_time;

  start_time = g_get_monotonic_time ();
  status = nnfw_run (pdata->session);
  stop_time = g_get_monotonic_time ();

  if (G_UNLIKELY (status != NNFW_STATUS_NO_ERROR)) {
    nns_loge ("Failed to invoke the model in nnfw (%d).", status);
    err = (status == NNFW_STATUS_INSUFFICIENT_OUTPUT_SIZE) ? -EAGAIN : -EINVAL;
  }

  nnfw_internal_stats.total_invoke_latency += stop_time - start_time;
  nnfw_internal_stats.total_invoke_num += 1;

#if (DBG)
  g_message ("Invoke() is finished: %" G_GINT64_FORMAT "ms, model path: %s", (stop_time - start_time) / 1000, pdata->model_file);
  g_message ("%" G_GINT64_FORMAT " invoke average %" G_GINT64_FORMAT ", total overhead %" G_GINT64_FORMAT,
      nnfw_internal_stats.total_invoke_num,
      (nnfw_internal_stats.total_invoke_latency / nnfw_internal_stats.total_invoke_num),
      nnfw_internal_stats.total_overhead_latency);
#endif

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

  g_return_val_if_fail (G_UNLIKELY (private_data != NULL), -EINVAL);
  pdata = (nnfw_pdata *) (*private_data);

  return nnfw_invoke_internal (pdata, &pdata->in_info, &pdata->out_info, input,
      output);
}

/**
 * @brief Check support of the backend
 * @param hw: backend to check support of
 */
static int
nnfw_checkAvailability (accl_hw hw)
{
  if (g_strv_contains (nnfw_accl_support, get_accl_hw_str (hw)))
    return 0;

  return -ENOENT;
}

static gchar filter_subplugin_nnfw[] = "nnfw";

static GstTensorFilterFramework NNS_support_nnfw = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = nnfw_open,
  .close = nnfw_close,
  {
    .v0 = {
      .name = filter_subplugin_nnfw,
      .allow_in_place = FALSE,
      .allocate_in_invoke = FALSE,
      .run_without_model = FALSE,
      .verify_model_path = FALSE,
      .statistics = &nnfw_internal_stats,
      .invoke_NN = nnfw_invoke,
      .getInputDimension = nnfw_getInputDim,
      .getOutputDimension = nnfw_getOutputDim,
      .setInputDimension = nnfw_setInputDim,
      .destroyNotify = NULL,
      .reloadModel = NULL,
      .checkAvailability = nnfw_checkAvailability,
      .allocateInInvoke = NULL,
    }
  }
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
  nnstreamer_filter_exit (NNS_support_nnfw.v0.name);
}

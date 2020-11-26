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

#include <tensor_common.h>
#include <nnstreamer_plugin_api_filter.h>
#include <nnfw.h>

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

static GstTensorFilterFrameworkStatistics nnfw_internal_stats;

/**
 * @brief private data structure for the nnfw framework
 */
typedef struct
{
  GstTensorsInfo in_info;
  GstTensorsInfo out_info;
  nnfw_session *session;
  gchar *model_file;
  gchar *accelerator;

  NNFW_TYPE in_type[NNS_TENSOR_SIZE_LIMIT];   /**< cached input tensor types */
  NNFW_TYPE out_type[NNS_TENSOR_SIZE_LIMIT];  /**< cached output tensor types */
} nnfw_pdata;

static void nnfw_close (const GstTensorFilterProperties * prop,
    void **private_data);
static int nnfw_tensors_info_get (const nnfw_pdata * pdata,
    const gboolean is_input, GstTensorsInfo * info, NNFW_TYPE * type);
static int nnfw_invoke (const GstTensorFilterProperties * prop,
    void **private_data, const GstTensorMemory * input,
    GstTensorMemory * output);

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
          g_warning ("Unknown option (%s).", options[i]);
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
    g_printerr ("Cannot find model file %s.", prop->model_files[0]);
    return -EINVAL;
  }

  pdata = *private_data = g_new0 (nnfw_pdata, 1);
  if (pdata == NULL) {
    g_printerr ("Failed to allocate memory for filter subplugin.\n");
    return -ENOMEM;
  }

  nnfw_parse_custom_option (prop, pdata);

  status = nnfw_create_session (&pdata->session);
  if (status != NNFW_STATUS_NO_ERROR) {
    err = -EINVAL;
    g_printerr ("Cannot create nnfw-runtime session\n");
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
    g_printerr ("Cannot load the model file: %s\n", prop->model_files[0]);
    goto error_exit;
  }

  status = nnfw_set_available_backends (pdata->session, pdata->accelerator);
  if (status != NNFW_STATUS_NO_ERROR) {
    err = -EINVAL;
    g_printerr ("Cannot set nnfw-runtime backend to %s\n", pdata->accelerator);
    goto error_exit;
  }

  status = nnfw_prepare (pdata->session);
  if (status != NNFW_STATUS_NO_ERROR) {
    err = -EINVAL;
    g_printerr ("nnfw-runtime cannot prepare the session for %s\n",
        prop->model_files[0]);
    goto error_exit;
  }

  err = nnfw_tensors_info_get (pdata, TRUE, &pdata->in_info, pdata->in_type);
  if (err) {
    g_printerr ("Error retrieving input info from nnfw-runtime.\n");
    goto error_exit;
  }

  err = nnfw_tensors_info_get (pdata, FALSE, &pdata->out_info, pdata->out_type);
  if (err) {
    g_printerr ("Error retrieving output info from nnfw-runtime.\n");
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
      g_printerr ("cannot close nnfw-runtime session for %s\n",
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
  NNFW_STATUS status;
  gint err;
  gint idx;
  const GstTensorInfo *info = &tensors_info->info[tensor_idx];

  err = nnfw_tensor_type_from_gst (info->type, &nnfw_info.dtype);
  if (err)
    return err;

  nnfw_info.rank = gst_tensor_info_get_rank (info);

  /** reverse the order of dimension */
  for (idx = nnfw_info.rank - 1; idx >= 0; idx--)
    nnfw_info.dims[nnfw_info.rank - idx - 1] = info->dimension[idx];

  /** @note Maximum rank expressible with nnfw is 6 (NNFW_MAX_RANK) */
  for (idx = NNFW_MAX_RANK - 1; idx >= nnfw_info.rank; idx--)
    nnfw_info.dims[idx] = 0;

#if defined (NNFW_USE_OLD_API)
  /**
   * @todo nnfw_apply_tensorinfo() will be deprecated.
   * Use nnfw_set_input_tensorinfo() later (nnfw ver >= 1.6.0).
   */
  status = nnfw_apply_tensorinfo (pdata->session, tensor_idx, nnfw_info);
#else
  status = nnfw_set_input_tensorinfo (pdata->session, tensor_idx, &nnfw_info);
#endif
  if (status != NNFW_STATUS_NO_ERROR)
    return -EPERM;

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
    GstTensorsInfo * info, NNFW_TYPE * type)
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

    type[idx] = nnfw_info_t.dtype;
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
  pdata = (nnfw_pdata *) * private_data;

  g_return_val_if_fail (pdata != NULL, -EINVAL);
  g_return_val_if_fail (info != NULL, -EINVAL);

  gst_tensors_info_copy (info, &pdata->out_info);

  return 0;
}

/**
 * @brief Internal function to invoke with dummy data.
 * When changing the input shape, NNFW will update the output shape after the invoke process is done.
 */
static void
nnfw_invoke_dummy (const GstTensorFilterProperties * prop, void **private_data,
    GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  GstTensorMemory input[NNS_TENSOR_SIZE_LIMIT] = { 0, };
  GstTensorMemory output[NNS_TENSOR_SIZE_LIMIT] = { 0, };
  guint i;

  for (i = 0; i < in_info->num_tensors; ++i) {
    input[i].size = gst_tensor_info_get_size (&in_info->info[i]);
    input[i].data = g_malloc (input[i].size);
  }

  /* The output shape would be changed, set enough size for output buffer. */
  for (i = 0; i < out_info->num_tensors; ++i) {
    output[i].size = gst_tensor_info_get_size (&out_info->info[i]) * 2;
    output[i].data = g_malloc (output[i].size);
  }

  while (nnfw_invoke (prop, private_data,  input, output) != 0) {
    g_warning ("Invoke failed, reallocate output tensors and retry.");
    for (i = 0; i < out_info->num_tensors; ++i) {
      output[i].size *= 2;
      output[i].data = g_realloc (output[i].data, output[i].size);
    }
  }

  for (i = 0; i < in_info->num_tensors; ++i) {
    g_free (input[i].data);
    input[i].data = NULL;
  }
  for (i = 0; i < out_info->num_tensors; ++i) {
    g_free (output[i].data);
    output[i].data = NULL;
  }
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
  NNFW_TYPE in_type[NNS_TENSOR_SIZE_LIMIT];
  NNFW_TYPE out_type[NNS_TENSOR_SIZE_LIMIT];

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

  err = nnfw_tensors_info_get (pdata, TRUE, &updated_info, in_type);
  if (err || !gst_tensors_info_is_equal (in_info, &updated_info))
    goto error;

  /* Invoke with dummy. NNFW updates output info after the invoke is done. */
  nnfw_invoke_dummy (prop, private_data, &updated_info, &pdata->out_info);

  err = nnfw_tensors_info_get (pdata, FALSE, out_info, out_type);
  if (err)
    goto error;

  gst_tensors_info_copy (&pdata->in_info, in_info);
  gst_tensors_info_copy (&pdata->out_info, out_info);
  memcpy (pdata->in_type, in_type, sizeof (NNFW_TYPE) * NNS_TENSOR_SIZE_LIMIT);
  memcpy (pdata->out_type, out_type, sizeof (NNFW_TYPE) * NNS_TENSOR_SIZE_LIMIT);

  return 0;

error:
  g_printerr ("Unable to set the provided input tensor info\n");
  /** Reset input dimensions */
  for (idx = 0; idx < pdata->in_info.num_tensors; idx++) {
    nnfw_tensor_info_set (pdata, &pdata->in_info, idx);
  }

  nnfw_invoke_dummy (prop, private_data, &pdata->in_info, &pdata->out_info);
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
  NNFW_STATUS nnfw_status;
  guint idx;
  unsigned int num_tensors = 0;

  g_return_val_if_fail (
      G_UNLIKELY (prop != NULL && mem != NULL && pdata != NULL), -EINVAL);
  g_return_val_if_fail (G_UNLIKELY (pdata->session != NULL), -EPERM);

  if (is_input) {
    num_tensors = pdata->in_info.num_tensors;
  } else {
    num_tensors = pdata->out_info.num_tensors;
  }

  for (idx = 0; idx < num_tensors; idx++) {
    if (is_input) {
      nnfw_status = nnfw_set_input (pdata->session, idx, pdata->in_type[idx],
          mem[idx].data, mem[idx].size);
    } else {
      nnfw_status = nnfw_set_output (pdata->session, idx, pdata->out_type[idx],
          mem[idx].data, mem[idx].size);
    }
    if (nnfw_status != NNFW_STATUS_NO_ERROR)
      return -EINVAL;
  }

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
  int64_t start_time, stop_time;
  nnfw_pdata *pdata;
  int err = 0;
  NNFW_STATUS nnfw_status;

  start_time = g_get_monotonic_time ();
  g_return_val_if_fail (G_UNLIKELY(private_data != NULL), -EINVAL);
  pdata = (nnfw_pdata *) * private_data;

  err = nnfw_tensor_memory_set (prop, pdata, input, TRUE);
  if (G_UNLIKELY(err < 0))
    return err;

  err = nnfw_tensor_memory_set (prop, pdata, output, FALSE);
  if (G_UNLIKELY(err < 0))
    return err;
  stop_time = g_get_monotonic_time ();

  nnfw_internal_stats.total_overhead_latency += stop_time - start_time;

  start_time = g_get_monotonic_time ();
  nnfw_status = nnfw_run (pdata->session);
  stop_time = g_get_monotonic_time ();

  if (G_UNLIKELY (nnfw_status != NNFW_STATUS_NO_ERROR)) {
    g_printerr ("Failed to invoke the model in nnfw (%d).", nnfw_status);
    err = -EINVAL;
  }

  nnfw_internal_stats.total_invoke_latency += stop_time - start_time;
  nnfw_internal_stats.total_invoke_num += 1;

  return err;
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
};

/**@brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_nnfw (void)
{
  nnfw_internal_stats.total_invoke_num = 0;
  nnfw_internal_stats.total_invoke_latency = 0;
  nnfw_internal_stats.total_overhead_latency = 0;

  NNS_support_nnfw.name = filter_subplugin_nnfw;
  NNS_support_nnfw.allow_in_place = FALSE;
  NNS_support_nnfw.allocate_in_invoke = FALSE;
  NNS_support_nnfw.run_without_model = FALSE;
  NNS_support_nnfw.verify_model_path = FALSE;
  NNS_support_nnfw.invoke_NN = nnfw_invoke;
  NNS_support_nnfw.getInputDimension = nnfw_getInputDim;
  NNS_support_nnfw.getOutputDimension = nnfw_getOutputDim;
  NNS_support_nnfw.setInputDimension = nnfw_setInputDim;
  NNS_support_nnfw.checkAvailability = nnfw_checkAvailability;
  NNS_support_nnfw.statistics = &nnfw_internal_stats;

  nnstreamer_filter_probe (&NNS_support_nnfw);
}

/** @brief Destruct the subplugin */
void
fini_filter_nnfw (void)
{
  nnstreamer_filter_exit (NNS_support_nnfw.name);
}

/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 */
/**
 * @file nnstreamer-capi-single.c
 * @date 29 Aug 2019
 * @brief NNStreamer/Single C-API Wrapper.
 *        This allows to invoke individual input frame with NNStreamer.
 * @see	https://github.com/nnstreamer/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <nnstreamer-single.h>
#include <nnstreamer-capi-private.h>
#include <nnstreamer_plugin_api.h>

#include "tensor_filter_single.h"
#include "nnstreamer_profile.h"

#define ML_SINGLE_MAGIC 0xfeedfeed

/**
 * @brief Default time to wait for an output in appsink (3 seconds).
 */
#define SINGLE_DEFAULT_TIMEOUT 3000

/**
 * @brief Global lock for single shot API
 * @detail This lock ensures that ml_single_close is thread safe. All other API
 *         functions use the mutex from the single handle. However for close,
 *         single handle mutex cannot be used as single handle is destroyed at
 *         close
 * @note This mutex is automatically initialized as it is statically declared
 */
G_LOCK_DEFINE_STATIC (magic);

/**
 * @brief Get valid handle after magic verification
 * @note handle's mutex (single_h->mutex) is acquired after this
 * @param[out] single_h The handle properly casted: (ml_single *).
 * @param[in] single The handle to be validated: (void *).
 * @param[in] reset Set TRUE if the handle is to be reset (magic = 0).
 */
#define ML_SINGLE_GET_VALID_HANDLE_LOCKED(single_h, single, reset) do { \
  G_LOCK (magic); \
  single_h = (ml_single *) single; \
  if (single_h->magic != ML_SINGLE_MAGIC) { \
    ml_loge ("The given param, single is invalid."); \
    G_UNLOCK (magic); \
    return ML_ERROR_INVALID_PARAMETER; \
  } \
  if (reset) \
    single_h->magic = 0; \
  g_mutex_lock (&single_h->mutex); \
  G_UNLOCK (magic); \
} while (0)

/**
 * @brief This is for the symmetricity with ML_SINGLE_GET_VALID_HANDLE_LOCKED
 * @param[in] single_h The casted handle (ml_single *).
 */
#define ML_SINGLE_HANDLE_UNLOCK(single_h) g_mutex_unlock (&single_h->mutex);

/** define string names for input/output */
#define INPUT_STR "input"
#define OUTPUT_STR "output"
#define TYPE_STR "type"
#define NAME_STR "name"

/** concat string from #define */
#define CONCAT_MACRO_STR(STR1,STR2) STR1 STR2

/** States for invoke thread */
typedef enum
{
  IDLE = 0,           /**< ready to accept next input */
  RUNNING,            /**< running an input, cannot accept more input */
  JOIN_REQUESTED,     /**< should join the thread, will exit soon */
  ERROR               /**< error on thread, will exit soon */
} thread_state;

/** ML single api data structure for handle */
typedef struct
{
  GTensorFilterSingle *filter;  /**< tensor filter element */
  ml_tensors_info_s in_info;    /**< info about input */
  ml_tensors_info_s out_info;   /**< info about output */
  ml_nnfw_type_e nnfw;          /**< nnfw type for this filter */
  guint magic;                  /**< code to verify valid handle */

  GThread *thread;              /**< thread for invoking */
  GMutex mutex;                 /**< mutex for synchronization */
  GCond cond;                   /**< condition for synchronization */
  ml_tensors_data_h input;      /**< input received from user */
  ml_tensors_data_h *output;    /**< output to be sent back to user */
  guint timeout;                /**< timeout for invoking */
  thread_state state;           /**< current state of the thread */
  gboolean ignore_output;       /**< ignore and free the output */
  int status;                   /**< status of processing */
} ml_single;

/**
 * @brief thread to execute calls to invoke
 */
static void *
invoke_thread (void *arg)
{
  ml_single *single_h;
  GTensorFilterSingleClass *klass;
  GstTensorMemory in_tensors[NNS_TENSOR_SIZE_LIMIT];
  GstTensorMemory out_tensors[NNS_TENSOR_SIZE_LIMIT];
  ml_tensors_data_s *in_data, *out_data;
  unsigned int i;
  int status = ML_ERROR_NONE;

  single_h = (ml_single *) arg;

  g_mutex_lock (&single_h->mutex);

  /** get the tensor_filter element */
  klass = g_type_class_peek (G_TYPE_TENSOR_FILTER_SINGLE);
  if (!klass) {
    single_h->state = ERROR;
    single_h->status = ML_ERROR_STREAMS_PIPE;
    goto exit;
  }

  while (single_h->state <= RUNNING) {

    /** wait for data */
    while (single_h->state != RUNNING) {
      g_cond_wait (&single_h->cond, &single_h->mutex);
      if (single_h->state >= JOIN_REQUESTED)
        goto exit;
    }

    in_data = (ml_tensors_data_s *) single_h->input;

    /** Setup input buffer */
    for (i = 0; i < in_data->num_tensors; i++) {
      in_tensors[i].data = in_data->tensors[i].tensor;
      in_tensors[i].size = in_data->tensors[i].size;
      in_tensors[i].type = (tensor_type) single_h->in_info.info[i].type;
    }

    /** Setup output buffer */
    for (i = 0; i < single_h->out_info.num_tensors; i++) {
      /** memory will be allocated by tensor_filter_single */
      out_tensors[i].data = NULL;
      out_tensors[i].size =
          ml_tensor_info_get_size (&single_h->out_info.info[i]);
      out_tensors[i].type = (tensor_type) single_h->out_info.info[i].type;
    }
    g_mutex_unlock (&single_h->mutex);

    /** invoke the thread */
    if (klass->invoke (single_h->filter, in_tensors, out_tensors) == FALSE) {
      status = ML_ERROR_STREAMS_PIPE;
      g_mutex_lock (&single_h->mutex);
      goto wait_for_next;
    }

    g_mutex_lock (&single_h->mutex);
    /** Allocate output buffer */
    if (single_h->ignore_output == FALSE) {
      status = ml_tensors_data_create_no_alloc (&single_h->out_info,
          single_h->output);
      if (status != ML_ERROR_NONE) {
        ml_loge ("Failed to allocate the memory block.");
        (*single_h->output) = NULL;
        goto wait_for_next;
      }

      /** set the result */
      out_data = (ml_tensors_data_s *) (*single_h->output);
      for (i = 0; i < single_h->out_info.num_tensors; i++) {
        out_data->tensors[i].tensor = out_tensors[i].data;
      }
    } else {
      /**
       * Caller of the invoke thread has returned back with timeout
       * so, free the memory allocated by the invoke as their is no receiver
       */
      for (i = 0; i < single_h->out_info.num_tensors; i++)
        g_free (out_tensors[i].data);
    }

    /** loop over to wait for the next element */
  wait_for_next:
    single_h->status = status;
    if (single_h->state == RUNNING)
      single_h->state = IDLE;
    g_cond_broadcast (&single_h->cond);
  }

exit:
  if (single_h->state != ERROR)
    single_h->state = IDLE;
  g_mutex_unlock (&single_h->mutex);
  return NULL;
}

/**
 * @brief Sets the information (tensor dimension, type, name and so on) of required input data for the given model, and get updated output data information.
 * @details Note that a model/framework may not support setting such information.
 * @since_tizen 6.0
 * @param[in] single The model handle.
 * @param[in] in_info The handle of input tensors information.
 * @param[out] out_info The handle of output tensors information. The caller is responsible for freeing the information with ml_tensors_info_destroy().
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_NOT_SUPPORTED This implies that the given framework does not support dynamic dimensions.
 *         Use ml_single_get_input_info() and ml_single_get_output_info() instead for this framework.
 * @retval #ML_ERROR_INVALID_PARAMETER Fail. The parameter is invalid.
 */
static int
ml_single_update_info (ml_single_h single,
    const ml_tensors_info_h in_info, ml_tensors_info_h * out_info)
{
  int status;

  if (!single || !in_info || !out_info)
    return ML_ERROR_INVALID_PARAMETER;

  /* init null */
  *out_info = NULL;

  status = ml_single_set_input_info (single, in_info);
  if (status != ML_ERROR_NONE)
    return status;

  return ml_single_get_output_info (single, out_info);
}

/**
 * @brief Internal function to start the filter.
 */
static gboolean
ml_single_start (ml_single * single_h)
{
  GTensorFilterSingleClass *klass;

  klass = g_type_class_peek (G_TYPE_TENSOR_FILTER_SINGLE);
  if (!klass || !single_h)
    return FALSE;

  return klass->start (single_h->filter);
}

/**
 * @brief Internal function to stop the filter.
 */
static gboolean
ml_single_stop (ml_single * single_h)
{
  GTensorFilterSingleClass *klass;

  klass = g_type_class_peek (G_TYPE_TENSOR_FILTER_SINGLE);
  if (!klass || !single_h)
    return FALSE;

  return klass->stop (single_h->filter);
}

/**
 * @brief Internal function to get the gst info from tensor-filter.
 */
static void
ml_single_get_gst_info (ml_single * single_h, gboolean is_input,
    GstTensorsInfo * gst_info)
{
  const gchar *prop_prefix, *prop_name, *prop_type;
  gchar *val;
  guint num;

  if (is_input) {
    prop_prefix = INPUT_STR;
    prop_type = CONCAT_MACRO_STR (INPUT_STR, TYPE_STR);
    prop_name = CONCAT_MACRO_STR (INPUT_STR, NAME_STR);
  } else {
    prop_prefix = OUTPUT_STR;
    prop_type = CONCAT_MACRO_STR (OUTPUT_STR, TYPE_STR);
    prop_name = CONCAT_MACRO_STR (OUTPUT_STR, NAME_STR);
  }

  gst_tensors_info_init (gst_info);

  /* get dimensions */
  g_object_get (single_h->filter, prop_prefix, &val, NULL);
  num = gst_tensors_info_parse_dimensions_string (gst_info, val);
  g_free (val);

  /* set the number of tensors */
  gst_info->num_tensors = num;

  /* get types */
  g_object_get (single_h->filter, prop_type, &val, NULL);
  num = gst_tensors_info_parse_types_string (gst_info, val);
  g_free (val);

  if (gst_info->num_tensors != num) {
    ml_logw ("The number of tensor type is mismatched in filter.");
  }

  /* get names */
  g_object_get (single_h->filter, prop_name, &val, NULL);
  num = gst_tensors_info_parse_names_string (gst_info, val);
  g_free (val);

  if (gst_info->num_tensors != num) {
    ml_logw ("The number of tensor name is mismatched in filter.");
  }
}

/**
 * @brief Internal function to set the gst info in tensor-filter.
 */
static int
ml_single_set_gst_info (ml_single * single_h, const ml_tensors_info_h info)
{
  GTensorFilterSingleClass *klass;
  GstTensorsInfo gst_in_info, gst_out_info;
  int status = ML_ERROR_NONE;

  switch (single_h->nnfw) {
    case ML_NNFW_TYPE_TENSORFLOW_LITE:
    case ML_NNFW_TYPE_CUSTOM_FILTER:
      ml_tensors_info_copy_from_ml (&gst_in_info, info);

      klass = g_type_class_peek (G_TYPE_TENSOR_FILTER_SINGLE);
      if (klass == NULL
          || klass->set_input_info (single_h->filter, &gst_in_info,
              &gst_out_info) == FALSE) {
        status = ML_ERROR_INVALID_PARAMETER;
        goto exit;
      }

      ml_tensors_info_copy_from_gst (&single_h->in_info, &gst_in_info);
      ml_tensors_info_copy_from_gst (&single_h->out_info, &gst_out_info);
      break;
    default:
      status = ML_ERROR_NOT_SUPPORTED;
      break;
  }

exit:
  return status;
}

/**
 * @brief Set the info for input/output tensors
 */
static int
ml_single_set_inout_tensors_info (GObject * object,
    const gboolean is_input, ml_tensors_info_s * tensors_info)
{
  int status = ML_ERROR_NONE;
  GstTensorsInfo info;
  gchar *str_dim, *str_type, *str_name;
  const gchar *str_type_name, *str_name_name;
  const gchar *prefix;

  if (is_input) {
    prefix = INPUT_STR;
    str_type_name = CONCAT_MACRO_STR (INPUT_STR, TYPE_STR);
    str_name_name = CONCAT_MACRO_STR (INPUT_STR, NAME_STR);
  } else {
    prefix = OUTPUT_STR;
    str_type_name = CONCAT_MACRO_STR (OUTPUT_STR, TYPE_STR);
    str_name_name = CONCAT_MACRO_STR (OUTPUT_STR, NAME_STR);
  }

  ml_tensors_info_copy_from_ml (&info, tensors_info);

  /* Set input option */
  str_dim = gst_tensors_info_get_dimensions_string (&info);
  str_type = gst_tensors_info_get_types_string (&info);
  str_name = gst_tensors_info_get_names_string (&info);

  if (!str_dim || !str_type || !str_name || !str_type_name || !str_name_name) {
    status = ML_ERROR_INVALID_PARAMETER;
  } else {
    g_object_set (object, prefix, str_dim, str_type_name, str_type,
        str_name_name, str_name, NULL);
  }

  g_free (str_dim);
  g_free (str_type);
  g_free (str_name);

  gst_tensors_info_free (&info);

  return status;
}

/**
 * @brief Internal static function to set tensors info in the handle.
 */
static gboolean
ml_single_set_info_in_handle (ml_single_h single, gboolean is_input,
    ml_tensors_info_s * tensors_info)
{
  int status;
  ml_single *single_h;
  ml_tensors_info_s *dest;
  gboolean configured = FALSE;
  gboolean is_valid = FALSE;
  GTensorFilterSingleClass *klass;
  GObject *filter_obj;

  single_h = (ml_single *) single;
  filter_obj = G_OBJECT (single_h->filter);
  klass = g_type_class_peek (G_TYPE_TENSOR_FILTER_SINGLE);
  if (!klass) {
    return FALSE;
  }

  if (is_input) {
    dest = &single_h->in_info;
    configured = klass->input_configured (single_h->filter);
  } else {
    dest = &single_h->out_info;
    configured = klass->output_configured (single_h->filter);
  }

  if (configured) {
    /* get configured info and compare with input info */
    GstTensorsInfo gst_info;
    ml_tensors_info_h info = NULL;

    ml_single_get_gst_info (single_h, is_input, &gst_info);
    ml_tensors_info_create_from_gst (&info, &gst_info);

    gst_tensors_info_free (&gst_info);

    if (tensors_info && !ml_tensors_info_is_equal (tensors_info, info)) {
      /* given input info is not matched with configured */
      ml_tensors_info_destroy (info);
      if (is_input) {
        /* try to update tensors info */
        status = ml_single_update_info (single, tensors_info, &info);
        if (status != ML_ERROR_NONE)
          goto done;
      } else {
        goto done;
      }
    }

    ml_tensors_info_clone (dest, info);
    ml_tensors_info_destroy (info);
  } else if (tensors_info) {
    status =
        ml_single_set_inout_tensors_info (filter_obj, is_input, tensors_info);
    if (status != ML_ERROR_NONE)
      goto done;
    ml_tensors_info_clone (dest, tensors_info);
  }

  is_valid = ml_tensors_info_is_valid (dest);

done:
  return is_valid;
}

/**
 * @brief Internal function to create and initialize the single handle.
 */
static ml_single *
ml_single_create_handle (ml_nnfw_type_e nnfw)
{
  ml_single *single_h;
  GError *error;

  single_h = g_new0 (ml_single, 1);
  if (single_h == NULL) {
    ml_loge ("Failed to allocate the single handle.");
    return NULL;
  }

  single_h->filter = g_object_new (G_TYPE_TENSOR_FILTER_SINGLE, NULL);
  if (single_h->filter == NULL) {
    ml_loge ("Failed to create a new instance for filter.");
    g_free (single_h);
    return NULL;
  }

  single_h->magic = ML_SINGLE_MAGIC;
  single_h->timeout = SINGLE_DEFAULT_TIMEOUT;
  single_h->nnfw = nnfw;
  single_h->state = IDLE;
  single_h->ignore_output = FALSE;

  ml_tensors_info_initialize (&single_h->in_info);
  ml_tensors_info_initialize (&single_h->out_info);
  g_mutex_init (&single_h->mutex);
  g_cond_init (&single_h->cond);

  single_h->thread =
      g_thread_try_new (NULL, invoke_thread, (gpointer) single_h, &error);
  if (single_h->thread == NULL) {
    ml_loge ("Failed to create the invoke thread, error: %s.", error->message);
    g_clear_error (&error);
    ml_single_close (single_h);
    return NULL;
  }

  return single_h;
}

/**
 * @brief Validate arguments for open
 */
static int
_ml_single_open_custom_validate_arguments (ml_single_h * single,
    ml_single_preset * info)
{
  if (!single) {
    ml_loge ("The given param is invalid: 'single' is NULL.");
    return ML_ERROR_INVALID_PARAMETER;
  }
  if (!info) {
    ml_loge ("The given param is invalid: 'info' is NULL.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /* Validate input tensor info. */
  if (info->input_info && !ml_tensors_info_is_valid (info->input_info)) {
    ml_loge ("The given param, input tensor info is invalid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /* Validate output tensor info. */
  if (info->output_info && !ml_tensors_info_is_valid (info->output_info)) {
    ml_loge ("The given param, output tensor info is invalid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (!info->models) {
    ml_loge ("The given param, model is invalid: info->models is NULL.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

/**
 * @brief Opens an ML model with the custom options and returns the instance as a handle.
 */
int
ml_single_open_custom (ml_single_h * single, ml_single_preset * info)
{
  ml_single *single_h;
  GObject *filter_obj;
  int status = ML_ERROR_NONE;
  ml_tensors_info_s *in_tensors_info, *out_tensors_info;
  ml_nnfw_type_e nnfw;
  ml_nnfw_hw_e hw;
  const char *fw_name;
  gchar **list_models;
  guint num_models;

  check_feature_state ();

  /* Validate the params */
  status = _ml_single_open_custom_validate_arguments (single, info);
  if (ML_ERROR_NONE != status)
    return status;

  /* init null */
  *single = NULL;

  in_tensors_info = (ml_tensors_info_s *) info->input_info;
  out_tensors_info = (ml_tensors_info_s *) info->output_info;
  nnfw = info->nnfw;
  hw = info->hw;

  /**
   * 1. Determine nnfw and validate model file
   */
  list_models = g_strsplit (info->models, ",", -1);
  num_models = g_strv_length (list_models);

  status = ml_validate_model_file (list_models, num_models, &nnfw);
  if (status != ML_ERROR_NONE) {
    g_strfreev (list_models);
    return status;
  }

  g_strfreev (list_models);

  /**
   * 2. Determine hw
   * (Supposed CPU only) Support others later.
   */
  if (!ml_nnfw_is_available (nnfw, hw)) {
    ml_loge ("The given nnfw is not available.");
    return ML_ERROR_NOT_SUPPORTED;
  }

  /** Create ml_single object */
  if ((single_h = ml_single_create_handle (nnfw)) == NULL)
    return ML_ERROR_OUT_OF_MEMORY;

  filter_obj = G_OBJECT (single_h->filter);

  /**
   * 3. Construct a pipeline
   * @todo Set the hw property
   * Set the pipeline desc with nnfw.
   */
  if (nnfw == ML_NNFW_TYPE_TENSORFLOW || nnfw == ML_NNFW_TYPE_SNAP) {
    /* set input and output tensors information */
    if (in_tensors_info && out_tensors_info) {
      status =
          ml_single_set_inout_tensors_info (filter_obj, TRUE, in_tensors_info);
      if (status != ML_ERROR_NONE)
        goto error;

      status =
          ml_single_set_inout_tensors_info (filter_obj, FALSE,
          out_tensors_info);
      if (status != ML_ERROR_NONE)
        goto error;
    } else {
      ml_loge
          ("To run the pipeline, input and output information should be initialized.");
      status = ML_ERROR_INVALID_PARAMETER;
      goto error;
    }
  } else if (nnfw == ML_NNFW_TYPE_ARMNN) {
    /* set input and output tensors information, if available */
    if (in_tensors_info) {
      status =
          ml_single_set_inout_tensors_info (filter_obj, TRUE, in_tensors_info);
      if (status != ML_ERROR_NONE)
        goto error;
    }
    if (out_tensors_info) {
      status =
          ml_single_set_inout_tensors_info (filter_obj, FALSE,
          out_tensors_info);
      if (status != ML_ERROR_NONE)
        goto error;
    }
  }

  /* set framework, model files and custom option */
  fw_name = ml_get_nnfw_subplugin_name (nnfw);
  g_object_set (filter_obj, "framework", fw_name, "model", info->models, NULL);

  if (info->custom_option) {
    g_object_set (filter_obj, "custom", info->custom_option, NULL);
  }

  /* 4. Start the nnfw to get inout configurations if needed */
  if (!ml_single_start (single_h)) {
    status = ML_ERROR_STREAMS_PIPE;
    goto error;
  }

  /* 5. Set in/out configs and metadata */
  if (!ml_single_set_info_in_handle (single_h, TRUE, in_tensors_info)) {
    ml_loge ("The input tensor info is invalid.");
    status = ML_ERROR_INVALID_PARAMETER;
    goto error;
  }

  if (!ml_single_set_info_in_handle (single_h, FALSE, out_tensors_info)) {
    ml_loge ("The output tensor info is invalid.");
    status = ML_ERROR_INVALID_PARAMETER;
    goto error;
  }

  *single = single_h;
  return ML_ERROR_NONE;

error:
  ml_single_close (single_h);
  return status;
}

/**
 * @brief Opens an ML model and returns the instance as a handle.
 */
int
ml_single_open (ml_single_h * single, const char *model,
    const ml_tensors_info_h input_info, const ml_tensors_info_h output_info,
    ml_nnfw_type_e nnfw, ml_nnfw_hw_e hw)
{
  ml_single_preset info = { 0, };

  info.input_info = input_info;
  info.output_info = output_info;
  info.nnfw = nnfw;
  info.hw = hw;
  info.models = (char *) model;
  info.custom_option = NULL;

  return ml_single_open_custom (single, &info);
}

/**
 * @brief Closes the opened model handle.
 */
int
ml_single_close (ml_single_h single)
{
  ml_single *single_h;

  check_feature_state ();

  if (!single) {
    ml_loge ("The given param, single is invalid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  ML_SINGLE_GET_VALID_HANDLE_LOCKED (single_h, single, 1);

  single_h->state = JOIN_REQUESTED;
  g_cond_broadcast (&single_h->cond);

  ML_SINGLE_HANDLE_UNLOCK (single_h);

  if (single_h->thread != NULL)
    g_thread_join (single_h->thread);

  /** locking ensures correctness with parallel calls on close */
  if (single_h->filter) {
    ml_single_stop (single_h);

    gst_object_unref (single_h->filter);
    single_h->filter = NULL;
  }

  ml_tensors_info_free (&single_h->in_info);
  ml_tensors_info_free (&single_h->out_info);

  g_cond_clear (&single_h->cond);
  g_mutex_clear (&single_h->mutex);

  g_free (single_h);
  return ML_ERROR_NONE;
}

/**
 * @brief Invokes the model with the given input data.
 */
int
ml_single_invoke (ml_single_h single,
    const ml_tensors_data_h input, ml_tensors_data_h * output)
{
  ml_single *single_h;
  ml_tensors_data_s *in_data;
  gint64 end_time;
  unsigned int i;
  int status = ML_ERROR_NONE;

  profile_log ("ml_single_invoke", PROFILE_START);

  check_feature_state ();

  if (!single) {
    ml_loge
        ("The first argument of ml_single_invoke() is not valid. Please check the single handle.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (!input) {
    ml_loge
        ("The second argument of ml_single_invoke() is not valid. Please check the input data handle.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (!output) {
    ml_loge
        ("The third argument of ml_single_invoke() is not valid. Please check the output data handle.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  ML_SINGLE_GET_VALID_HANDLE_LOCKED (single_h, single, 0);

  in_data = (ml_tensors_data_s *) input;
  *output = NULL;

  if (!single_h->filter) {
    ml_loge
        ("The tensor_filter element is not valid. It is not correctly created or already freed.");
    status = ML_ERROR_INVALID_PARAMETER;
    goto exit;
  }

  if (single_h->state == JOIN_REQUESTED) {
    ml_loge ("The handle is closed or being closed.");
    status = ML_ERROR_STREAMS_PIPE;
    goto exit;
  }

  if (single_h->state == ERROR) {
    ml_loge ("There was error on getting tesnor_filter element.");
    status = ML_ERROR_STREAMS_PIPE;
    goto exit;
  }

  /* Validate input data */
  if (in_data->num_tensors != single_h->in_info.num_tensors) {
    ml_loge
        ("The number of input tensors is not compatible with model. Given: %u, Expected: %u.",
        in_data->num_tensors, single_h->in_info.num_tensors);
    status = ML_ERROR_INVALID_PARAMETER;
    goto exit;
  }

  for (i = 0; i < in_data->num_tensors; i++) {
    size_t raw_size;

    if (!in_data->tensors[i].tensor) {
      ml_loge ("The %d-th input tensor is not valid.", i);
      status = ML_ERROR_INVALID_PARAMETER;
      goto exit;
    }

    raw_size = ml_tensor_info_get_size (&single_h->in_info.info[i]);

    if (in_data->tensors[i].size != raw_size) {
      ml_loge
          ("The size of %d-th input tensor is not compatible with model. Given: %zu, Expected: %zu (type: %d).",
          i, in_data->tensors[i].size, raw_size,
          single_h->in_info.info[i].type);
      status = ML_ERROR_INVALID_PARAMETER;
      goto exit;
    }
  }

  if (single_h->state != IDLE) {
    ml_loge ("The single invoking thread is not idle.");
    status = ML_ERROR_TRY_AGAIN;
    goto exit;
  }

  single_h->input = input;
  single_h->output = output;
  single_h->state = RUNNING;
  single_h->ignore_output = FALSE;

  end_time = g_get_monotonic_time () +
      single_h->timeout * G_TIME_SPAN_MILLISECOND;

  g_cond_broadcast (&single_h->cond);
  if (g_cond_wait_until (&single_h->cond, &single_h->mutex, end_time)) {
    status = single_h->status;
  } else {
    ml_logw ("Wait for invoke has timed out");
    status = ML_ERROR_TIMED_OUT;
    /** This is set to notify invoke_thread to not process if timedout */
    single_h->ignore_output = TRUE;

    /** Free if any output memory was allocated */
    if (*single_h->output != NULL) {
      ml_tensors_data_destroy ((ml_tensors_data_h) * single_h->output);
      *single_h->output = NULL;
    }
  }

exit:
  ML_SINGLE_HANDLE_UNLOCK (single_h);

  if (status == ML_ERROR_NONE)
    profile_log ("ml_single_invoke", PROFILE_END);
  return status;
}

/**
 * @brief Gets the tensors info for the given handle.
 */
static int
ml_single_get_tensors_info (ml_single_h single, gboolean is_input,
    ml_tensors_info_h * info)
{
  ml_single *single_h;
  int status = ML_ERROR_NONE;
  ml_tensors_info_s *input_info;

  check_feature_state ();

  if (!single || !info)
    return ML_ERROR_INVALID_PARAMETER;

  ML_SINGLE_GET_VALID_HANDLE_LOCKED (single_h, single, 0);

  /* allocate handle for tensors info */
  status = ml_tensors_info_create (info);
  if (status != ML_ERROR_NONE)
    goto exit;

  input_info = (ml_tensors_info_s *) (*info);

  if (is_input)
    status = ml_tensors_info_clone (input_info, &single_h->in_info);
  else
    status = ml_tensors_info_clone (input_info, &single_h->out_info);

  if (status != ML_ERROR_NONE)
    ml_tensors_info_destroy (input_info);

exit:
  ML_SINGLE_HANDLE_UNLOCK (single_h);
  return status;
}

/**
 * @brief Gets the information of required input data for the given handle.
 * @note information = (tensor dimension, type, name and so on)
 */
int
ml_single_get_input_info (ml_single_h single, ml_tensors_info_h * info)
{
  return ml_single_get_tensors_info (single, TRUE, info);
}

/**
 * @brief Gets the information of output data for the given handle.
 * @note information = (tensor dimension, type, name and so on)
 */
int
ml_single_get_output_info (ml_single_h single, ml_tensors_info_h * info)
{
  return ml_single_get_tensors_info (single, FALSE, info);
}

/**
 * @brief Sets the maximum amount of time to wait for an output, in milliseconds.
 */
int
ml_single_set_timeout (ml_single_h single, unsigned int timeout)
{
  ml_single *single_h;

  check_feature_state ();

  if (!single || timeout == 0)
    return ML_ERROR_INVALID_PARAMETER;

  ML_SINGLE_GET_VALID_HANDLE_LOCKED (single_h, single, 0);

  single_h->timeout = (guint) timeout;

  ML_SINGLE_HANDLE_UNLOCK (single_h);
  return ML_ERROR_NONE;
}

/**
 * @brief Sets the information (tensor dimension, type, name and so on) of required input data for the given model.
 */
int
ml_single_set_input_info (ml_single_h single, const ml_tensors_info_h info)
{
  ml_single *single_h;
  int status = ML_ERROR_NONE;

  check_feature_state ();

  if (!single || !info)
    return ML_ERROR_INVALID_PARAMETER;

  if (!ml_tensors_info_is_valid (info))
    return ML_ERROR_INVALID_PARAMETER;

  ML_SINGLE_GET_VALID_HANDLE_LOCKED (single_h, single, 0);
  status = ml_single_set_gst_info (single_h, info);
  ML_SINGLE_HANDLE_UNLOCK (single_h);

  return status;
}

/**
 * @brief Invokes the model with the given input data with the given info.
 */
int
ml_single_invoke_dynamic (ml_single_h single,
    const ml_tensors_data_h input, const ml_tensors_info_h in_info,
    ml_tensors_data_h * output, ml_tensors_info_h * out_info)
{
  int status;
  ml_tensors_info_h cur_in_info = NULL;

  if (!single || !input || !in_info || !output || !out_info)
    return ML_ERROR_INVALID_PARAMETER;

  /* init null */
  *output = NULL;
  *out_info = NULL;

  status = ml_single_get_input_info (single, &cur_in_info);
  if (status != ML_ERROR_NONE)
    goto exit;

  status = ml_single_update_info (single, in_info, out_info);
  if (status != ML_ERROR_NONE)
    goto exit;

  status = ml_single_invoke (single, input, output);
  if (status != ML_ERROR_NONE) {
    ml_single_set_input_info (single, cur_in_info);
  }

exit:
  if (cur_in_info)
    ml_tensors_info_destroy (cur_in_info);

  if (status != ML_ERROR_NONE) {
    if (*out_info) {
      ml_tensors_info_destroy (*out_info);
      *out_info = NULL;
    }
  }

  return status;
}

/**
 * @brief Sets the property value for the given model.
 */
int
ml_single_set_property (ml_single_h single, const char *name, const char *value)
{
  ml_single *single_h;
  int status = ML_ERROR_NONE;
  char *old_value = NULL;

  check_feature_state ();

  if (!single || !name || !value)
    return ML_ERROR_INVALID_PARAMETER;

  /* get old value, also check the property is updatable. */
  status = ml_single_get_property (single, name, &old_value);
  if (status != ML_ERROR_NONE)
    return status;

  /* if sets same value, do not change. */
  if (old_value != NULL && g_ascii_strcasecmp (old_value, value) == 0) {
    g_free (old_value);
    return ML_ERROR_NONE;
  }

  ML_SINGLE_GET_VALID_HANDLE_LOCKED (single_h, single, 0);

  /* update property */
  if (g_str_equal (name, "is-updatable")) {
    /* boolean */
    if (g_ascii_strcasecmp (value, "true") == 0) {
      if (g_ascii_strcasecmp (old_value, "true") != 0)
        g_object_set (G_OBJECT (single_h->filter), name, (gboolean) TRUE, NULL);
    } else if (g_ascii_strcasecmp (value, "false") == 0) {
      if (g_ascii_strcasecmp (old_value, "false") != 0)
        g_object_set (G_OBJECT (single_h->filter), name, (gboolean) FALSE,
            NULL);
    } else {
      ml_loge ("The property value (%s) is not available.", value);
      status = ML_ERROR_INVALID_PARAMETER;
    }
  } else if (g_str_equal (name, "input") || g_str_equal (name, "inputtype")
      || g_str_equal (name, "inputname") || g_str_equal (name, "output")
      || g_str_equal (name, "outputtype") || g_str_equal (name, "outputname")) {
    GstTensorsInfo gst_info;
    gboolean is_input = g_str_has_prefix (name, "input");
    guint num;

    ml_single_get_gst_info (single_h, is_input, &gst_info);

    if (g_str_has_suffix (name, "type"))
      num = gst_tensors_info_parse_types_string (&gst_info, value);
    else if (g_str_has_suffix (name, "name"))
      num = gst_tensors_info_parse_names_string (&gst_info, value);
    else
      num = gst_tensors_info_parse_dimensions_string (&gst_info, value);

    if (num == gst_info.num_tensors) {
      ml_tensors_info_h ml_info;

      ml_tensors_info_create_from_gst (&ml_info, &gst_info);

      /* change configuration */
      status = ml_single_set_gst_info (single_h, ml_info);

      ml_tensors_info_destroy (ml_info);
    } else {
      ml_loge ("The property value (%s) is not available.", value);
      status = ML_ERROR_INVALID_PARAMETER;
    }

    gst_tensors_info_free (&gst_info);
  } else {
    g_object_set (G_OBJECT (single_h->filter), name, value, NULL);
  }

  ML_SINGLE_HANDLE_UNLOCK (single_h);

  g_free (old_value);
  return status;
}

/**
 * @brief Gets the property value for the given model.
 */
int
ml_single_get_property (ml_single_h single, const char *name, char **value)
{
  ml_single *single_h;
  int status = ML_ERROR_NONE;

  check_feature_state ();

  if (!single || !name || !value)
    return ML_ERROR_INVALID_PARAMETER;

  /* init null */
  *value = NULL;

  ML_SINGLE_GET_VALID_HANDLE_LOCKED (single_h, single, 0);

  if (g_str_equal (name, "input") || g_str_equal (name, "inputtype") ||
      g_str_equal (name, "inputname") || g_str_equal (name, "inputlayout") ||
      g_str_equal (name, "output") || g_str_equal (name, "outputtype") ||
      g_str_equal (name, "outputname") || g_str_equal (name, "outputlayout") ||
      g_str_equal (name, "accelerator") || g_str_equal (name, "custom")) {
    /* string */
    g_object_get (G_OBJECT (single_h->filter), name, value, NULL);
  } else if (g_str_equal (name, "is-updatable")) {
    gboolean bool_value = FALSE;

    /* boolean */
    g_object_get (G_OBJECT (single_h->filter), name, &bool_value, NULL);
    *value = (bool_value) ? g_strdup ("true") : g_strdup ("false");
  } else {
    ml_loge ("The property %s is not available.", name);
    status = ML_ERROR_NOT_SUPPORTED;
  }

  ML_SINGLE_HANDLE_UNLOCK (single_h);
  return status;
}

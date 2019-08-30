/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd All Rights Reserved
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
 */
/**
 * @file nnstreamer-capi-single-new.c
 * @date 29 Aug 2019
 * @brief NNStreamer/Single C-API Wrapper.
 *        This allows to invoke individual input frame with NNStreamer.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <string.h>

#include <nnstreamer-single.h>
#include <nnstreamer-capi-private.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer/nnstreamer_conf.h>
#include <nnstreamer/tensor_filter/tensor_filter.h>

#include "tensor_filter_single.h"

/* ML single api data structure for handle */
typedef struct
{
  GTensorFilterSingle *filter;
  ml_tensors_info_s in_info;
  ml_tensors_info_s out_info;
} ml_single;

/**
 * @brief Set the info for input/output tensors
 */
static int
ml_single_set_inout_tensors_info (GObject *object,
    const gchar *prefix, ml_tensors_info_s *tensors_info)
{
  int status = ML_ERROR_NONE;
  GstTensorsInfo info;
  gchar *str_dim, *str_type, *str_name;
  gchar *str_type_name, *str_name_name;

  ml_tensors_info_copy_from_ml (&info, tensors_info);

  /* Set input option */
  str_dim = gst_tensors_info_get_dimensions_string (&info);
  str_type = gst_tensors_info_get_types_string (&info);
  str_name = gst_tensors_info_get_names_string (&info);

  str_type_name = g_strdup_printf("%s%s", prefix, "type");
  str_name_name = g_strdup_printf("%s%s", prefix, "name");

  if (!str_dim || !str_type || !str_name || !str_type_name || !str_name_name) {
    status = ML_ERROR_INVALID_PARAMETER;
  } else {
    g_object_set (object, prefix, str_dim, str_type_name, str_type,
        str_name_name, str_name, NULL);
  }

  g_free (str_type_name);
  g_free (str_name_name);
  g_free (str_dim);
  g_free (str_type);
  g_free (str_name);

  gst_tensors_info_free (&info);

  return status;
}

/**
 * @brief Check the availability of the nnfw type and model
 */
static int
ml_single_check_nnfw (const char *model, ml_nnfw_type_e *nnfw)
{
  gchar *path_down;
  int status = ML_ERROR_NONE;

  if (!g_file_test (model, G_FILE_TEST_IS_REGULAR)) {
    ml_loge ("The given param, model path [%s] is invalid.",
        GST_STR_NULL (model));
    return ML_ERROR_INVALID_PARAMETER;
  }

  /* Check file extention. */
  path_down = g_ascii_strdown (model, -1);

  switch (*nnfw) {
    case ML_NNFW_TYPE_ANY:
      if (g_str_has_suffix (path_down, ".tflite")) {
        ml_logi ("The given model [%s] is supposed a tensorflow-lite model.",
            model);
        *nnfw = ML_NNFW_TYPE_TENSORFLOW_LITE;
      } else if (g_str_has_suffix (path_down, ".pb")) {
        ml_logi ("The given model [%s] is supposed a tensorflow model.", model);
        *nnfw = ML_NNFW_TYPE_TENSORFLOW;
      } else if (!g_str_has_suffix (path_down, NNSTREAMER_SO_FILE_EXTENSION)) {
        ml_logi ("The given model [%s] is supposed a custom filter model.",
            model);
        *nnfw = ML_NNFW_TYPE_CUSTOM_FILTER;
      } else {
        ml_loge ("The given model [%s] has unknown extension.", model);
        status = ML_ERROR_INVALID_PARAMETER;
      }
      break;
    case ML_NNFW_TYPE_CUSTOM_FILTER:
      if (!g_str_has_suffix (path_down, NNSTREAMER_SO_FILE_EXTENSION)) {
        ml_loge ("The given model [%s] has invalid extension.", model);
        status = ML_ERROR_INVALID_PARAMETER;
      }
      break;
    case ML_NNFW_TYPE_TENSORFLOW_LITE:
      if (!g_str_has_suffix (path_down, ".tflite")) {
        ml_loge ("The given model [%s] has invalid extension.", model);
        status = ML_ERROR_INVALID_PARAMETER;
      }
      break;
    case ML_NNFW_TYPE_TENSORFLOW:
      if (!g_str_has_suffix (path_down, ".pb")) {
        ml_loge ("The given model [%s] has invalid extension.", model);
        status = ML_ERROR_INVALID_PARAMETER;
      }
      break;
    case ML_NNFW_TYPE_NNFW:
      /** @todo Need to check method for NNFW */
      ml_loge ("NNFW is not supported.");
      status = ML_ERROR_NOT_SUPPORTED;
      break;
    default:
      break;
  }

  g_free (path_down);

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
  ml_single *single_h;
  GObject *filter_obj;
  int status = ML_ERROR_NONE;
  GTensorFilterSingleClass *klass;
  ml_tensors_info_s *in_tensors_info, *out_tensors_info;
  bool available = false;
  bool valid = false;

  check_feature_state ();

  /* Validate the params */
  if (!single) {
    ml_loge ("The given param, single is invalid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /* init null */
  *single = NULL;

  in_tensors_info = (ml_tensors_info_s *) input_info;
  out_tensors_info = (ml_tensors_info_s *) output_info;

  if (input_info) {
    /* Validate input tensor info. */
    if (ml_tensors_info_validate (input_info, &valid) != ML_ERROR_NONE ||
        valid == false) {
      ml_loge ("The given param, input tensor info is invalid.");
      return ML_ERROR_INVALID_PARAMETER;
    }
  }

  if (output_info) {
    /* Validate output tensor info. */
    if (ml_tensors_info_validate (output_info, &valid) != ML_ERROR_NONE ||
        valid == false) {
      ml_loge ("The given param, output tensor info is invalid.");
      return ML_ERROR_INVALID_PARAMETER;
    }
  }

  /**
   * 1. Determine nnfw
   */
  if ((status = ml_single_check_nnfw (model, &nnfw)) != ML_ERROR_NONE)
    return status;

  /**
   * 2. Determine hw
   * @todo Now the param hw is ignored.
   * (Supposed CPU only) Support others later.
   */
  status = ml_check_nnfw_availability (nnfw, hw, &available);
  if (status != ML_ERROR_NONE)
    return status;

  if (!available) {
    ml_loge ("The given nnfw is not available.");
    status = ML_ERROR_NOT_SUPPORTED;
    return status;
  }

  /** Create ml_single object */
  single_h = g_new0 (ml_single, 1);
  g_assert (single_h);
  single_h->filter = g_object_new (G_TYPE_TENSOR_FILTER_SINGLE, NULL);
  if (single_h->filter == NULL) {
    status = ML_ERROR_INVALID_PARAMETER;
    goto error;
  }
  filter_obj = G_OBJECT (single_h->filter);

  /**
   * 3. Construct a pipeline
   * Set the pipeline desc with nnfw.
   */
  switch (nnfw) {
    case ML_NNFW_TYPE_CUSTOM_FILTER:
      g_object_set (filter_obj, "framework", "custom",
          "model", model, NULL);
      break;
    case ML_NNFW_TYPE_TENSORFLOW_LITE:
      /* We can get the tensor meta from tf-lite model. */
      g_object_set (filter_obj, "framework", "tensorflow-lite",
          "model", model, NULL);
      break;
    case ML_NNFW_TYPE_TENSORFLOW:
      if (in_tensors_info && out_tensors_info) {
        status = ml_single_set_inout_tensors_info (filter_obj, "input",
            in_tensors_info);
        if (status != ML_ERROR_NONE)
          goto error;

        status = ml_single_set_inout_tensors_info (filter_obj, "output",
            out_tensors_info);
        if (status != ML_ERROR_NONE)
          goto error;

        g_object_set (filter_obj, "framework", "tensorflow",
            "model", model, NULL);
      } else {
        ml_loge ("To run the pipeline with tensorflow model, \
            input and output information should be initialized.");
        status = ML_ERROR_INVALID_PARAMETER;
        goto error;
      }
      break;
    default:
      /** @todo Add other fw later. */
      ml_loge ("The given nnfw is not supported.");
      status = ML_ERROR_NOT_SUPPORTED;
      goto error;
  }

  /* 4. Allocate */
  ml_tensors_info_initialize (&single_h->in_info);
  ml_tensors_info_initialize (&single_h->out_info);

  /* 5. Start the nnfw to egt inout configurations if needed */
  klass = g_type_class_peek (G_TYPE_TENSOR_FILTER_SINGLE);
  if (!klass) {
    status = ML_ERROR_INVALID_PARAMETER;
    goto error;
  }
  if (klass->start (single_h->filter) == FALSE) {
    status = ML_ERROR_INVALID_PARAMETER;
    goto error;
  }

  /* 6. Set in/out configs and metadata */
  if (in_tensors_info) {
    /** set the tensors info here */
    if (!klass->input_configured(single_h->filter)) {
      status = ml_single_set_inout_tensors_info (filter_obj, "input",
          in_tensors_info);
      if (status != ML_ERROR_NONE)
        goto error;
    }
    status = ml_tensors_info_clone (&single_h->in_info, in_tensors_info);
    if (status != ML_ERROR_NONE)
      goto error;
  } else {
    ml_tensors_info_h in_info;

    status = ML_ERROR_INVALID_PARAMETER;
    if (!klass->input_configured(single_h->filter))
      goto error;

    status = ml_single_get_input_info (single_h, &in_info);
    if (status != ML_ERROR_NONE)
      goto error;

    status = ml_tensors_info_clone (&single_h->in_info, in_info);
    ml_tensors_info_destroy (in_info);
    if (status != ML_ERROR_NONE)
      goto error;

    status = ml_tensors_info_validate (&single_h->in_info, &valid);
    if (status != ML_ERROR_NONE || valid == false) {
      ml_loge ("Failed to get the input tensor info.");
      goto error;
    }
  }


  if (out_tensors_info) {
    /** set the tensors info here */
    if (!klass->output_configured(single_h->filter)) {
      status = ml_single_set_inout_tensors_info (filter_obj, "output",
          out_tensors_info);
      if (status != ML_ERROR_NONE)
        goto error;
    }
    status = ml_tensors_info_clone (&single_h->out_info, out_tensors_info);
    if (status != ML_ERROR_NONE)
      goto error;
  } else {
    ml_tensors_info_h out_info;

    status = ML_ERROR_INVALID_PARAMETER;
    if (!klass->output_configured(single_h->filter))
      goto error;

    status = ml_single_get_output_info (single_h, &out_info);
    if (status != ML_ERROR_NONE)
      goto error;

    status = ml_tensors_info_clone (&single_h->out_info, out_info);
    ml_tensors_info_destroy (out_info);
    if (status != ML_ERROR_NONE)
      goto error;

    status = ml_tensors_info_validate (&single_h->out_info, &valid);
    if (status != ML_ERROR_NONE || valid == false) {
      ml_loge ("Failed to get the output tensor info.");
      goto error;
    }
  }

  *single = single_h;
  return ML_ERROR_NONE;

error:
  ml_single_close (single_h);
  return status;
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

  single_h = (ml_single *) single;

  if (single_h->filter) {
    GTensorFilterSingleClass *klass;
    klass = g_type_class_peek (G_TYPE_TENSOR_FILTER_SINGLE);
    if (klass)
      klass->stop (single_h->filter);
    gst_object_unref (single_h->filter);
    single_h->filter = NULL;
  }

  ml_tensors_info_free (&single_h->in_info);
  ml_tensors_info_free (&single_h->out_info);

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
  /**
   * @todo:
   * Setup input and output buffer
   * Output buffer
   * Do invoke
   * return result
   */

  return ML_ERROR_NONE;
}

/**
 * @brief Gets the type of required input data for the given handle.
 * @note type = (tensor dimension, type, name and so on)
 */
int
ml_single_get_input_info (ml_single_h single, ml_tensors_info_h * info)
{
  ml_single *single_h;
  ml_tensors_info_s *input_info;
  GstTensorsInfo gst_info;
  gchar *val;
  guint rank;

  check_feature_state ();

  if (!single || !info)
    return ML_ERROR_INVALID_PARAMETER;

  single_h = (ml_single *) single;

  /* allocate handle for tensors info */
  ml_tensors_info_create (info);
  input_info = (ml_tensors_info_s *) (*info);

  gst_tensors_info_init (&gst_info);

  g_object_get (single_h->filter, "input", &val, NULL);
  rank = gst_tensors_info_parse_dimensions_string (&gst_info, val);
  g_free (val);

  /* set the number of tensors */
  gst_info.num_tensors = rank;

  g_object_get (single_h->filter, "inputtype", &val, NULL);
  rank = gst_tensors_info_parse_types_string (&gst_info, val);
  g_free (val);

  if (gst_info.num_tensors != rank) {
    ml_logw ("Invalid state, input tensor type is mismatched in filter.");
  }

  g_object_get (single_h->filter, "inputname", &val, NULL);
  rank = gst_tensors_info_parse_names_string (&gst_info, val);
  g_free (val);

  if (gst_info.num_tensors != rank) {
    ml_logw ("Invalid state, input tensor name is mismatched in filter.");
  }

  ml_tensors_info_copy_from_gst (input_info, &gst_info);
  gst_tensors_info_free (&gst_info);
  return ML_ERROR_NONE;
}

/**
 * @brief Gets the type of required output data for the given handle.
 * @note type = (tensor dimension, type, name and so on)
 */
int
ml_single_get_output_info (ml_single_h single, ml_tensors_info_h * info)
{
  ml_single *single_h;
  ml_tensors_info_s *output_info;
  GstTensorsInfo gst_info;
  gchar *val;
  guint rank;

  check_feature_state ();

  if (!single || !info)
    return ML_ERROR_INVALID_PARAMETER;

  single_h = (ml_single *) single;

  /* allocate handle for tensors info */
  ml_tensors_info_create (info);
  output_info = (ml_tensors_info_s *) (*info);

  gst_tensors_info_init (&gst_info);

  g_object_get (single_h->filter, "output", &val, NULL);
  rank = gst_tensors_info_parse_dimensions_string (&gst_info, val);
  g_free (val);

  /* set the number of tensors */
  gst_info.num_tensors = rank;

  g_object_get (single_h->filter, "outputtype", &val, NULL);
  rank = gst_tensors_info_parse_types_string (&gst_info, val);
  g_free (val);

  if (gst_info.num_tensors != rank) {
    ml_logw ("Invalid state, output tensor type is mismatched in filter.");
  }

  g_object_get (single_h->filter, "outputname", &val, NULL);
  gst_tensors_info_parse_names_string (&gst_info, val);
  g_free (val);

  if (gst_info.num_tensors != rank) {
    ml_logw ("Invalid state, output tensor name is mismatched in filter.");
  }

  ml_tensors_info_copy_from_gst (output_info, &gst_info);
  gst_tensors_info_free (&gst_info);
  return ML_ERROR_NONE;
}

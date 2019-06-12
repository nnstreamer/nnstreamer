/**
 * Copyright (c) 2019 Samsung Electronics Co., Ltd All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file nnstreamer-capi-single.c
 * @date 08 May 2019
 * @brief NNStreamer/Single C-API Wrapper.
 *        This allows to invoke individual input frame with NNStreamer.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <string.h>
#include <gst/app/app.h>

#include <nnstreamer/nnstreamer_plugin_api.h>

#include "nnstreamer.h"         /* Uses NNStreamer/Pipeline C-API */
#include "nnstreamer-single.h"
#include "nnstreamer-capi-private.h"

typedef struct
{
  ml_pipeline_h pipe;

  GstElement *src;
  GstElement *sink;
  GstElement *filter;

  ml_tensors_info_s in_info;
  ml_tensors_info_s out_info;
} ml_single;

/**
 * @brief Gets caps from tensors info.
 */
static GstCaps *
ml_single_get_caps_from_tensors_info (const ml_tensors_info_s * info)
{
  GstCaps *caps;
  GstTensorsConfig config;

  if (!info)
    return NULL;

  ml_util_copy_tensors_info_from_ml (&config.info, info);

  /* set framerate 0/1 */
  config.rate_n = 0;
  config.rate_d = 1;

  /* Supposed input type is single tensor if the number of tensors is 1. */
  if (config.info.num_tensors == 1) {
    GstTensorConfig c;

    gst_tensor_info_copy (&c.info, &config.info.info[0]);
    c.rate_n = 0;
    c.rate_d = 1;

    caps = gst_tensor_caps_from_config (&c);
    gst_tensor_info_free (&c.info);
  } else {
    caps = gst_tensors_caps_from_config (&config);
  }

  gst_tensors_info_free (&config.info);
  return caps;
}

/**
 * @brief Opens an ML model and returns the instance as a handle.
 */
int
ml_single_open (ml_single_h * single, const char *model_path,
    const ml_tensors_info_s * input_info, const ml_tensors_info_s * output_info,
    ml_nnfw_e nnfw, ml_nnfw_hw_e hw)
{
  ml_single *single_h;
  ml_pipeline_h pipe;
  ml_pipeline *pipe_h;
  GstElement *appsrc, *appsink, *filter;
  GstCaps *caps;
  int status = ML_ERROR_NONE;
  gchar *pipeline_desc = NULL;

  /* Validate the params */
  if (!single) {
    ml_loge ("The given param, single is invalid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /* init null */
  *single = NULL;

  if (!g_file_test (model_path, G_FILE_TEST_IS_REGULAR)) {
    ml_loge ("The given param, model path [%s] is invalid.",
        GST_STR_NULL (model_path));
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (input_info &&
      ml_util_validate_tensors_info (input_info) != ML_ERROR_NONE) {
    ml_loge ("The given param, input tensor info is invalid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (output_info &&
      ml_util_validate_tensors_info (output_info) != ML_ERROR_NONE) {
    ml_loge ("The given param, output tensor info is invalid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  status = ml_util_check_nnfw (nnfw, hw);
  if (status < 0) {
    ml_loge ("The given nnfw is not available.");
    return status;
  }

  /* 1. Determine nnfw */
  /** @todo Check nnfw with file extention. */
  switch (nnfw) {
    case ML_NNFW_CUSTOM_FILTER:
      pipeline_desc =
          g_strdup_printf
          ("appsrc name=srcx ! tensor_filter name=filterx framework=custom model=%s ! appsink name=sinkx sync=false",
          model_path);
      break;
    case ML_NNFW_TENSORFLOW_LITE:
      if (!g_str_has_suffix (model_path, ".tflite")) {
        ml_loge ("The given model file [%s] has invalid extension.", model_path);
        return ML_ERROR_INVALID_PARAMETER;
      }

      pipeline_desc =
          g_strdup_printf
          ("appsrc name=srcx ! tensor_filter name=filterx framework=tensorflow-lite model=%s ! appsink name=sinkx sync=false",
          model_path);
      break;
    case ML_NNFW_TENSORFLOW:
      if (!g_str_has_suffix (model_path, ".pb")) {
        ml_loge ("The given model file [%s] has invalid extension.", model_path);
        return ML_ERROR_INVALID_PARAMETER;
      }

      if (input_info && output_info) {
        GstTensorsInfo in_info, out_info;
        gchar *str_dim, *str_type, *str_name;
        gchar *in_option, *out_option;

        ml_util_copy_tensors_info_from_ml (&in_info, input_info);
        ml_util_copy_tensors_info_from_ml (&out_info, output_info);

        /* Set input option */
        str_dim = gst_tensors_info_get_dimensions_string (&in_info);
        str_type = gst_tensors_info_get_types_string (&in_info);
        str_name = gst_tensors_info_get_names_string (&in_info);
        in_option = g_strdup_printf ("input=%s inputtype=%s inputname=%s",
            str_dim, str_type, str_name);
        g_free (str_dim);
        g_free (str_type);
        g_free (str_name);

        /* Set output option */
        str_dim = gst_tensors_info_get_dimensions_string (&out_info);
        str_type = gst_tensors_info_get_types_string (&out_info);
        str_name = gst_tensors_info_get_names_string (&out_info);
        out_option = g_strdup_printf ("output=%s outputtype=%s outputname=%s",
            str_dim, str_type, str_name);
        g_free (str_dim);
        g_free (str_type);
        g_free (str_name);

        pipeline_desc =
            g_strdup_printf
            ("appsrc name=srcx ! tensor_filter name=filterx framework=tensorflow model=%s %s %s ! appsink name=sinkx sync=false",
            model_path, in_option, out_option);

        g_free (in_option);
        g_free (out_option);
        gst_tensors_info_free (&in_info);
        gst_tensors_info_free (&out_info);
      } else {
        ml_loge ("To run the pipeline with tensorflow model, input and output information should be initialized.");
        return ML_ERROR_INVALID_PARAMETER;
      }
      break;
    default:
      /** @todo Add other fw later. */
      ml_loge ("The given nnfw is not supported.");
      return ML_ERROR_NOT_SUPPORTED;
  }

  /* 2. Determine hw */
  /** @todo Now the param hw is ignored. (Supposed CPU only) Support others later. */

  /* 3. Construct a pipeline */
  status = ml_pipeline_construct (pipeline_desc, &pipe);
  g_free (pipeline_desc);
  if (status != ML_ERROR_NONE) {
    /* Failed to construct pipeline. */
    return status;
  }

  /* 4. Allocate */
  pipe_h = (ml_pipeline *) pipe;
  appsrc = gst_bin_get_by_name (GST_BIN (pipe_h->element), "srcx");
  appsink = gst_bin_get_by_name (GST_BIN (pipe_h->element), "sinkx");
  filter = gst_bin_get_by_name (GST_BIN (pipe_h->element), "filterx");

  single_h = g_new0 (ml_single, 1);
  g_assert (single_h);

  single_h->pipe = pipe;
  single_h->src = appsrc;
  single_h->sink = appsink;
  single_h->filter = filter;
  ml_util_initialize_tensors_info (&single_h->in_info);
  ml_util_initialize_tensors_info (&single_h->out_info);

  /* 5. Set in/out caps and metadata */
  if (input_info) {
    caps = ml_single_get_caps_from_tensors_info (input_info);
    ml_util_copy_tensors_info (&single_h->in_info, input_info);
  } else {
    ml_single_get_input_info (single_h, &single_h->in_info);

    status = ml_util_validate_tensors_info (&single_h->in_info);
    if (status != ML_ERROR_NONE) {
      ml_loge ("Failed to get the input tensor info.");
      goto error;
    }

    caps = ml_single_get_caps_from_tensors_info (&single_h->in_info);
  }

  gst_app_src_set_caps (GST_APP_SRC (appsrc), caps);
  gst_caps_unref (caps);

  if (output_info) {
    caps = ml_single_get_caps_from_tensors_info (output_info);
    ml_util_copy_tensors_info (&single_h->out_info, output_info);
  } else {
    ml_single_get_output_info (single_h, &single_h->out_info);

    status = ml_util_validate_tensors_info (&single_h->out_info);
    if (status != ML_ERROR_NONE) {
      ml_loge ("Failed to get the output tensor info.");
      goto error;
    }

    caps = ml_single_get_caps_from_tensors_info (&single_h->out_info);
  }

  gst_app_sink_set_caps (GST_APP_SINK (appsink), caps);
  gst_caps_unref (caps);

  /* 6. Start pipeline */
  status = ml_pipeline_start (pipe);
  if (status != ML_ERROR_NONE) {
    /* Failed to construct pipeline. */
    goto error;
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
  int ret;

  if (!single) {
    ml_loge ("The given param, single is invalid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  single_h = (ml_single *) single;

  if (single_h->src) {
    gst_object_unref (single_h->src);
    single_h->src = NULL;
  }

  if (single_h->sink) {
    gst_object_unref (single_h->sink);
    single_h->sink = NULL;
  }

  if (single_h->filter) {
    gst_object_unref (single_h->filter);
    single_h->filter = NULL;
  }

  ml_util_free_tensors_info (&single_h->in_info);
  ml_util_free_tensors_info (&single_h->out_info);

  ret = ml_pipeline_destroy (single_h->pipe);
  g_free (single_h);
  return ret;
}

/**
 * @brief Invokes the model with the given input data.
 */
ml_tensors_data_s *
ml_single_inference (ml_single_h single,
    const ml_tensors_data_s * input, ml_tensors_data_s * output)
{
  ml_single *single_h;
  ml_tensors_data_s *result = NULL;
  GstSample *sample;
  GstBuffer *buffer;
  GstMemory *mem;
  GstMapInfo mem_info;
  GstFlowReturn ret;
  int i, status = ML_ERROR_NONE;

  if (!single || !input) {
    ml_loge ("The given param is invalid.");
    status = ML_ERROR_INVALID_PARAMETER;
    goto error;
  }

  single_h = (ml_single *) single;

  /* Validate output memory and size */
  if (output) {
    if (output->num_tensors != single_h->out_info.num_tensors) {
      ml_loge ("Invalid output data, the number of output is different.");
      status = ML_ERROR_INVALID_PARAMETER;
      goto error;
    }

    for (i = 0; i < output->num_tensors; i++) {
      if (!output->tensors[i].tensor ||
          output->tensors[i].size !=
          ml_util_get_tensor_size (&single_h->out_info.info[i])) {
        ml_loge ("Invalid output data, the size of output is different.");
        status = ML_ERROR_INVALID_PARAMETER;
        goto error;
      }
    }
  }

  buffer = gst_buffer_new ();

  for (i = 0; i < input->num_tensors; i++) {
    mem = gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY,
        input->tensors[i].tensor, input->tensors[i].size, 0,
        input->tensors[i].size, NULL, NULL);
    gst_buffer_append_memory (buffer, mem);
  }

  ret = gst_app_src_push_buffer (GST_APP_SRC (single_h->src), buffer);
  if (ret != GST_FLOW_OK) {
    ml_loge ("Cannot push a buffer into source element.");
    status = ML_ERROR_STREAMS_PIPE;
    goto error;
  }

  /* Try to get the result */
  sample =
      gst_app_sink_try_pull_sample (GST_APP_SINK (single_h->sink), GST_SECOND);
  if (!sample) {
    ml_loge ("Failed to get the result from sink element.");
    status = ML_ERROR_TIMED_OUT;
    goto error;
  }

  if (output) {
    result = output;
  } else {
    result = ml_util_allocate_tensors_data (&single_h->out_info);

    if (!result) {
      ml_loge ("Failed to allocate the memory block.");
      status = ml_util_get_last_error ();
      goto error;
    }
  }

  /* Copy the result */
  buffer = gst_sample_get_buffer (sample);
  for (i = 0; i < result->num_tensors; i++) {
    mem = gst_buffer_peek_memory (buffer, i);
    gst_memory_map (mem, &mem_info, GST_MAP_READ);

    memcpy (result->tensors[i].tensor, mem_info.data, mem_info.size);

    gst_memory_unmap (mem, &mem_info);
  }

  gst_sample_unref (sample);
  status = ML_ERROR_NONE;

error:
  ml_util_set_error (status);
  return result;
}

/**
 * @brief Gets the type (tensor dimension, type, name and so on) of required input data for the given handle.
 */
int
ml_single_get_input_info (ml_single_h single,
    ml_tensors_info_s * input_info)
{
  ml_single *single_h;
  GstTensorsInfo info;
  gchar *val;
  guint rank;

  if (!single || !input_info)
    return ML_ERROR_INVALID_PARAMETER;

  single_h = (ml_single *) single;

  gst_tensors_info_init (&info);

  g_object_get (single_h->filter, "input", &val, NULL);
  rank = gst_tensors_info_parse_dimensions_string (&info, val);
  g_free (val);

  /* set the number of tensors */
  info.num_tensors = rank;

  g_object_get (single_h->filter, "inputtype", &val, NULL);
  rank = gst_tensors_info_parse_types_string (&info, val);
  g_free (val);

  if (info.num_tensors != rank) {
    ml_logw ("Invalid state, input tensor type is mismatched in filter.");
  }

  g_object_get (single_h->filter, "inputname", &val, NULL);
  rank = gst_tensors_info_parse_names_string (&info, val);
  g_free (val);

  if (info.num_tensors != rank) {
    ml_logw ("Invalid state, input tensor name is mismatched in filter.");
  }

  ml_util_copy_tensors_info_from_gst (input_info, &info);
  gst_tensors_info_free (&info);
  return ML_ERROR_NONE;
}

/**
 * @brief Gets the type (tensor dimension, type, name and so on) of output data for the given handle.
 */
int
ml_single_get_output_info (ml_single_h single,
    ml_tensors_info_s * output_info)
{
  ml_single *single_h;
  GstTensorsInfo info;
  gchar *val;
  guint rank;

  if (!single || !output_info)
    return ML_ERROR_INVALID_PARAMETER;

  single_h = (ml_single *) single;

  gst_tensors_info_init (&info);

  g_object_get (single_h->filter, "output", &val, NULL);
  rank = gst_tensors_info_parse_dimensions_string (&info, val);
  g_free (val);

  /* set the number of tensors */
  info.num_tensors = rank;

  g_object_get (single_h->filter, "outputtype", &val, NULL);
  rank = gst_tensors_info_parse_types_string (&info, val);
  g_free (val);

  if (info.num_tensors != rank) {
    ml_logw ("Invalid state, output tensor type is mismatched in filter.");
  }

  g_object_get (single_h->filter, "outputname", &val, NULL);
  gst_tensors_info_parse_names_string (&info, val);
  g_free (val);

  if (info.num_tensors != rank) {
    ml_logw ("Invalid state, output tensor name is mismatched in filter.");
  }

  ml_util_copy_tensors_info_from_gst (output_info, &info);
  gst_tensors_info_free (&info);
  return ML_ERROR_NONE;
}

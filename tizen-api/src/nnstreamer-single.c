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
 * @file nnstreamer-simple.c
 * @date 08 May 2019
 * @brief Tizen NNStreamer/Simple C-API Wrapper.
 *        This allows to invoke individual input frame with NNStreamer.
 * @see	https://github.com/nnsuite/nnstreamer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gst/app/app.h>

#include <nnstreamer.h>         /* Uses NNStreamer/Pipeline C-API */
#include <nnstreamer-single.h>
#include <tizen-api-private.h>
#include <nnstreamer/nnstreamer_plugin_api.h>

typedef struct
{
  ml_pipeline_h pipe;

  GstElement *src;
  GstElement *sink;
  GstElement *filter;
} ml_simpleshot_model;

/**
 * @brief Check the given tensor info is valid.
 * @todo move this function to common
 */
static int
ml_util_validate_tensor_info (const ml_tensor_info_s * info)
{
  guint i;

  if (!info)
    return FALSE;

  if (info->type < 0 || info->type >= ML_TENSOR_TYPE_UNKNOWN)
    return FALSE;

  for (i = 0; i < ML_TENSOR_RANK_LIMIT; i++) {
    if (info->dimension[i] == 0)
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief Check the given tensors info is valid.
 * @todo move this function to common
 */
static int
ml_util_validate_tensors_info (const ml_tensors_info_s * info)
{
  guint i;

  if (!info || info->num_tensors < 1)
    return FALSE;

  for (i = 0; i < info->num_tensors; i++) {
    if (!ml_util_validate_tensor_info (&info->info[i]))
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief Get caps from tensors info.
 */
static GstCaps *
ml_model_get_caps_from_tensors_info (const ml_tensors_info_s * info)
{
  GstCaps *caps;
  GstTensorsConfig config;

  if (!info)
    return NULL;

  /** @todo Make common structure for tensor config */
  memcpy (&config.info, info, sizeof (GstTensorsInfo));

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

  return caps;
}

/**
 * @brief Open an ML model and return the model as a handle. (more info in nnstreamer-single.h)
 */
int
ml_model_open (const char *model_path, ml_simpleshot_model_h * model,
    const ml_tensors_info_s * input_type, const ml_tensors_info_s * output_type,
    ml_model_nnfw nnfw, ml_model_hw hw)
{
  ml_simpleshot_model *model_h;
  ml_pipeline_h pipe;
  ml_pipeline *pipe_h;
  GstElement *appsrc, *appsink, *filter;
  GstCaps *caps;
  int ret = ML_ERROR_NONE;
  gchar *pipeline_desc = NULL;

  /* Validate the params */
  if (!model) {
    dloge ("The given param, model is invalid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /* init null */
  *model = NULL;

  if (!g_file_test (model_path, G_FILE_TEST_IS_REGULAR)) {
    dloge ("The given param, model path [%s] is invalid.",
        GST_STR_NULL (model_path));
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (input_type && !ml_util_validate_tensors_info (input_type)) {
    dloge ("The given param, input tensor info is invalid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (output_type && !ml_util_validate_tensors_info (output_type)) {
    dloge ("The given param, output tensor info is invalid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /* 1. Determine nnfw */
  /** @todo Check nnfw with file extention. */
  switch (nnfw) {
    case ML_NNFW_CUSTOM_FILTER:
      pipeline_desc =
          g_strdup_printf
          ("appsrc name=srcx ! tensor_filter name=filterx framework=custom model=%s ! appsink name=sinkx async=false sync=false",
          model_path);
      break;
    case ML_NNFW_TENSORFLOW_LITE:
      if (!g_str_has_suffix (model_path, ".tflite")) {
        dloge ("The given model file [%s] has invalid extension.", model_path);
        return ML_ERROR_INVALID_PARAMETER;
      }

      pipeline_desc =
          g_strdup_printf
          ("appsrc name=srcx ! tensor_filter name=filterx framework=tensorflow-lite model=%s ! appsink name=sinkx async=false sync=false",
          model_path);
      break;
    default:
      /** @todo Add other fw later. */
      dloge ("The given nnfw is not supported.");
      return ML_ERROR_NOT_SUPPORTED;
  }

  /* 2. Determine hw */
  /** @todo Now the param hw is ignored. (Supposed CPU only) Support others later. */

  /* 3. Construct a pipeline */
  ret = ml_pipeline_construct (pipeline_desc, &pipe);
  g_free (pipeline_desc);
  if (ret != ML_ERROR_NONE) {
    /* Failed to construct pipeline. */
    return ret;
  }

  /* 4. Allocate */
  pipe_h = (ml_pipeline *) pipe;
  appsrc = gst_bin_get_by_name (GST_BIN (pipe_h->element), "srcx");
  appsink = gst_bin_get_by_name (GST_BIN (pipe_h->element), "sinkx");
  filter = gst_bin_get_by_name (GST_BIN (pipe_h->element), "filterx");

  model_h = g_new0 (ml_simpleshot_model, 1);
  *model = model_h;

  model_h->pipe = pipe;
  model_h->src = appsrc;
  model_h->sink = appsink;
  model_h->filter = filter;

  /* 5. Set in/out caps */
  if (input_type) {
    caps = ml_model_get_caps_from_tensors_info (input_type);
  } else {
    ml_tensors_info_s in_info;

    ml_model_get_input_type (model_h, &in_info);
    if (!ml_util_validate_tensors_info (&in_info)) {
      dloge ("Failed to get the input tensor info.");
      goto error;
    }

    caps = ml_model_get_caps_from_tensors_info (&in_info);
  }

  gst_app_src_set_caps (GST_APP_SRC (appsrc), caps);
  gst_caps_unref (caps);

  if (output_type) {
    caps = ml_model_get_caps_from_tensors_info (output_type);
  } else {
    ml_tensors_info_s out_info;

    ml_model_get_output_type (model_h, &out_info);
    if (!ml_util_validate_tensors_info (&out_info)) {
      dloge ("Failed to get the output tensor info.");
      goto error;
    }

    caps = ml_model_get_caps_from_tensors_info (&out_info);
  }

  gst_app_sink_set_caps (GST_APP_SINK (appsink), caps);
  gst_caps_unref (caps);

  /* 5. Start pipeline */
  ret = ml_pipeline_start (pipe);
  if (ret != ML_ERROR_NONE) {
    /* Failed to construct pipeline. */
    goto error;
  }

  return ML_ERROR_NONE;

error:
  ml_model_close (pipe);
  return ret;
}

/**
 * @brief Close the opened model handle. (more info in nnstreamer-single.h)
 */
int
ml_model_close (ml_simpleshot_model_h model)
{
  ml_simpleshot_model *model_h;
  int ret;

  if (!model) {
    dloge ("The given param, model is invalid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  model_h = (ml_simpleshot_model *) model;

  if (model_h->src) {
    gst_object_unref (model_h->src);
    model_h->src = NULL;
  }

  if (model_h->sink) {
    gst_object_unref (model_h->sink);
    model_h->sink = NULL;
  }

  if (model_h->filter) {
    gst_object_unref (model_h->filter);
    model_h->filter = NULL;
  }

  ret = ml_pipeline_destroy (model_h->pipe);
  g_free (model_h);
  return ret;
}

/**
 * @brief Invoke the model with the given input data. (more info in nnstreamer-single.h)
 */
ml_tensor_data_s *
ml_model_inference (ml_simpleshot_model_h model,
    const ml_tensor_data_s * input, ml_tensor_data_s * output)
{
  ml_simpleshot_model *model_h;
  ml_tensors_info_s out_info;
  ml_tensor_data_s *result;
  GstSample *sample;
  GstBuffer *buffer;
  GstMemory *mem;
  GstMapInfo mem_info;
  GstFlowReturn ret;
  int i, status;

  if (!model || !input) {
    dloge ("The given param is invalid.");
    return NULL;
  }

  model_h = (ml_simpleshot_model *) model;

  status = ml_model_get_output_type (model, &out_info);
  if (status != ML_ERROR_NONE)
    return NULL;

  /* Validate output memory and size */
  if (output) {
    if (output->num_tensors != out_info.num_tensors) {
      dloge ("Invalid output data, the number of output is different.");
      return NULL;
    }

    for (i = 0; i < output->num_tensors; i++) {
      if (!output->tensor[i] ||
          output->size[i] !=
          ml_util_get_tensor_size (&out_info.info[i])) {
        dloge ("Invalid output data, the size of output is different.");
        return NULL;
      }
    }
  }

  buffer = gst_buffer_new ();

  for (i = 0; i < input->num_tensors; i++) {
    mem = gst_memory_new_wrapped (GST_MEMORY_FLAG_READONLY,
        input->tensor[i], input->size[i], 0, input->size[i], NULL, NULL);
    gst_buffer_append_memory (buffer, mem);
  }

  ret = gst_app_src_push_buffer (GST_APP_SRC (model_h->src), buffer);
  if (ret != GST_FLOW_OK) {
    dloge ("Cannot push a buffer into source element.");
    return NULL;
  }

  /* Try to get the result */
  sample =
      gst_app_sink_try_pull_sample (GST_APP_SINK (model_h->sink), GST_SECOND);
  if (!sample) {
    dloge ("Failed to get the result from sink element.");
    return NULL;
  }

  if (output) {
    result = output;
  } else {
    result = ml_model_allocate_tensor_data (&out_info);
  }

  if (!result) {
    dloge ("Failed to allocate the memory block.");
    return NULL;
  }

  /* Copy the result */
  buffer = gst_sample_get_buffer (sample);
  for (i = 0; i < result->num_tensors; i++) {
    mem = gst_buffer_peek_memory (buffer, i);
    gst_memory_map (mem, &mem_info, GST_MAP_READ);

    memcpy (result->tensor[i], mem_info.data, mem_info.size);

    gst_memory_unmap (mem, &mem_info);
  }

  gst_sample_unref (sample);
  return result;
}

/**
 * @brief Get type (tensor dimension, type, name and so on) of required input data for the given model. (more info in nnstreamer-single.h)
 */
int
ml_model_get_input_type (ml_simpleshot_model_h model,
    ml_tensors_info_s * input_type)
{
  ml_simpleshot_model *model_h;
  GstTensorsInfo info;
  gchar *val;
  guint rank;

  if (!model || !input_type)
    return ML_ERROR_INVALID_PARAMETER;

  model_h = (ml_simpleshot_model *) model;

  gst_tensors_info_init (&info);

  g_object_get (model_h->filter, "input", &val, NULL);
  rank = gst_tensors_info_parse_dimensions_string (&info, val);
  g_free (val);

  /* set the number of tensors */
  info.num_tensors = rank;

  g_object_get (model_h->filter, "inputtype", &val, NULL);
  rank = gst_tensors_info_parse_types_string (&info, val);
  g_free (val);

  if (info.num_tensors != rank) {
    dlogw ("Invalid state, input tensor type is mismatched in filter.");
  }

  g_object_get (model_h->filter, "inputname", &val, NULL);
  rank = gst_tensors_info_parse_names_string (&info, val);
  g_free (val);

  if (info.num_tensors != rank) {
    dlogw ("Invalid state, input tensor name is mismatched in filter.");
  }
  /** @todo Make common structure for tensor config */
  memcpy (input_type, &info, sizeof (GstTensorsInfo));
  return ML_ERROR_NONE;
}

/**
 * @brief Get type (tensor dimension, type, name and so on) of output data of the given model. (more info in nnstreamer-single.h)
 */
int
ml_model_get_output_type (ml_simpleshot_model_h model,
    ml_tensors_info_s * output_type)
{
  ml_simpleshot_model *model_h;
  GstTensorsInfo info;
  gchar *val;
  guint rank;

  if (!model || !output_type)
    return ML_ERROR_INVALID_PARAMETER;

  model_h = (ml_simpleshot_model *) model;

  gst_tensors_info_init (&info);

  g_object_get (model_h->filter, "output", &val, NULL);
  rank = gst_tensors_info_parse_dimensions_string (&info, val);
  g_free (val);

  /* set the number of tensors */
  info.num_tensors = rank;

  g_object_get (model_h->filter, "outputtype", &val, NULL);
  rank = gst_tensors_info_parse_types_string (&info, val);
  g_free (val);

  if (info.num_tensors != rank) {
    dlogw ("Invalid state, output tensor type is mismatched in filter.");
  }

  g_object_get (model_h->filter, "outputname", &val, NULL);
  gst_tensors_info_parse_names_string (&info, val);
  g_free (val);

  if (info.num_tensors != rank) {
    dlogw ("Invalid state, output tensor name is mismatched in filter.");
  }
  /** @todo Make common structure for tensor config */
  memcpy (output_type, &info, sizeof (GstTensorsInfo));
  return ML_ERROR_NONE;
}

/**
 * @brief Get the byte size of the given tensor type. (more info in nnstreamer-single.h)
 */
size_t
ml_util_get_tensor_size (const ml_tensor_info_s * info)
{
  size_t tensor_size;
  gint i;

  if (!info) {
    dloge ("The given param tensor info is invalid.");
    return 0;
  }

  switch (info->type) {
  case ML_TENSOR_TYPE_INT8:
  case ML_TENSOR_TYPE_UINT8:
    tensor_size = 1;
    break;
  case ML_TENSOR_TYPE_INT16:
  case ML_TENSOR_TYPE_UINT16:
    tensor_size = 2;
    break;
  case ML_TENSOR_TYPE_INT32:
  case ML_TENSOR_TYPE_UINT32:
  case ML_TENSOR_TYPE_FLOAT32:
    tensor_size = 4;
    break;
  case ML_TENSOR_TYPE_FLOAT64:
  case ML_TENSOR_TYPE_INT64:
  case ML_TENSOR_TYPE_UINT64:
    tensor_size = 8;
    break;
  default:
    dloge ("The given param tensor_type is invalid.");
    return 0;
  }

  for (i = 0; i < ML_TENSOR_RANK_LIMIT; i++) {
    tensor_size *= info->dimension[i];
  }

  return tensor_size;
}

/**
 * @brief Get the byte size of the given tensors info. (more info in nnstreamer-single.h)
 */
size_t
ml_util_get_tensors_size (const ml_tensors_info_s * info)
{
  size_t tensor_size;
  gint i;

  tensor_size = 0;
  for (i = 0; i < info->num_tensors; i++) {
    tensor_size += ml_util_get_tensor_size (&info->info[i]);
  }

  return tensor_size;
}

/**
 * @brief Free the tensors type pointer. (more info in nnstreamer-single.h)
 */
void
ml_model_free_tensors_info (ml_tensors_info_s * type)
{
  /** @todo Make common structure for tensor config and use gst_tensors_info_free () */
}

/**
 * @brief Free the tensors data pointer. (more info in nnstreamer-single.h)
 */
void
ml_model_free_tensor_data (ml_tensor_data_s * tensor)
{
  gint i;

  if (!tensor) {
    dloge ("The given param tensor is invalid.");
    return;
  }

  for (i = 0; i < tensor->num_tensors; i++) {
    if (tensor->tensor[i]) {
      g_free (tensor->tensor[i]);
      tensor->tensor[i] = NULL;
    }

    tensor->size[i] = 0;
  }

  tensor->num_tensors = 0;
}

/**
 * @brief Allocate a tensor data frame with the given tensors type. (more info in nnstreamer-single.h)
 */
ml_tensor_data_s *
ml_model_allocate_tensor_data (const ml_tensors_info_s * info)
{
  ml_tensor_data_s *data;
  gint i;

  if (!info) {
    dloge ("The given param type is invalid.");
    return NULL;
  }

  data = g_new0 (ml_tensor_data_s, 1);
  if (!data) {
    dloge ("Failed to allocate the memory block.");
    return NULL;
  }

  data->num_tensors = info->num_tensors;
  for (i = 0; i < data->num_tensors; i++) {
    data->size[i] = ml_util_get_tensor_size (&info->info[i]);
    data->tensor[i] = g_malloc0 (data->size[i]);
  }

  return data;
}

/**
 * @brief Check the availability of the given execution environments. (more info in nnstreamer-single.h)
 */
int
ml_model_check_nnfw (ml_model_nnfw nnfw, ml_model_hw hw)
{
  /** @todo fill this function */
  return 0;
}

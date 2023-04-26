/**
 * GStreamer Tensor_Filter, Customized Module, Easy Mode
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
 * @file	tensor_filter_custom_easy.c
 * @date	24 Oct 2019
 * @brief	Custom tensor processing interface for simple functions
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <errno.h>
#include <glib.h>
#include <tensor_filter_custom_easy.h>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_subplugin.h>
#include <nnstreamer_util.h>

void init_filter_custom_easy (void) __attribute__((constructor));
void fini_filter_custom_easy (void) __attribute__((destructor));

const char *fw_name = "custom-easy";

/**
 * @brief internal_data
 */
typedef struct _internal_data
{
  NNS_custom_invoke func;
  GstTensorsInfo in_info;
  GstTensorsInfo out_info;
  void *data; /**< The easy-filter writer's data */
} internal_data;

/**
 * @brief The easy-filter user's data
 */
typedef struct
{
  const internal_data *model;
} runtime_data;

/**
 * @brief Internal function to release internal data.
 */
static void
custom_free_internal_data (internal_data * data)
{
  if (data) {
    gst_tensors_info_free (&data->in_info);
    gst_tensors_info_free (&data->out_info);
    g_free (data);
  }
}

/**
 * @brief Register the custom-easy tensor function. More info in .h
 * @return 0 if success. -ERRNO if error.
 */
int
NNS_custom_easy_register (const char *modelname,
    NNS_custom_invoke func, void *data,
    const GstTensorsInfo * in_info, const GstTensorsInfo * out_info)
{
  internal_data *ptr;

  if (!func || !in_info || !out_info)
    return -EINVAL;

  if (!gst_tensors_info_validate (in_info) ||
      !gst_tensors_info_validate (out_info))
    return -EINVAL;

  ptr = g_new0 (internal_data, 1);

  if (!ptr)
    return -ENOMEM;

  ptr->func = func;
  ptr->data = data;
  gst_tensors_info_copy (&ptr->in_info, in_info);
  gst_tensors_info_copy (&ptr->out_info, out_info);

  if (register_subplugin (NNS_EASY_CUSTOM_FILTER, modelname, ptr))
    return 0;

  custom_free_internal_data (ptr);
  return -EINVAL;
}

/**
 * @brief Unregister the custom-easy tensor function.
 * @return 0 if success. -EINVAL if invalid model name.
 */
int
NNS_custom_easy_unregister (const char *modelname)
{
  internal_data *ptr;

  /* get internal data before unregistering the custom filter */
  ptr = (internal_data *) get_subplugin (NNS_EASY_CUSTOM_FILTER, modelname);

  if (!unregister_subplugin (NNS_EASY_CUSTOM_FILTER, modelname)) {
    ml_loge ("Failed to unregister custom filter %s.", modelname);
    return -EINVAL;
  }

  /* free internal data */
  custom_free_internal_data (ptr);
  return 0;
}

/**
 * @brief Callback required by tensor_filter subplugin
 */
static int
custom_open (const GstTensorFilterProperties * prop, void **private_data)
{
  runtime_data *rd;

  rd = g_new (runtime_data, 1);
  if (!rd)
    return -ENOMEM;
  rd->model = get_subplugin (NNS_EASY_CUSTOM_FILTER, prop->model_files[0]);

  if (NULL == rd->model) {
    ml_loge
        ("Cannot find the easy-custom model, \"%s\". You should provide a valid model name of easy-custom.",
        prop->model_files[0]);
    goto errorreturn;
  }

  if (NULL == rd->model->func) {
    ml_logf
        ("A custom-easy filter, \"%s\", should provide invoke function body, 'func'. A null-ptr is supplied instead.\n",
        prop->model_files[0]);
    goto errorreturn;
  }
  if (!gst_tensors_info_validate (&rd->model->in_info)) {
    ml_logf
        ("A custom-easy filter, \"%s\", should provide input stream metadata, 'in_info'.\n",
        prop->model_files[0]);
    goto errorreturn;
  }
  if (!gst_tensors_info_validate (&rd->model->out_info)) {
    ml_logf
        ("A custom-easy filter, \"%s\", should provide output stream metadata, 'out_info'.\n",
        prop->model_files[0]);
    goto errorreturn;
  }

  *private_data = rd;
  return 0;
errorreturn:
  g_free (rd);
  return -EINVAL;
}

/**
 * @brief Callback required by tensor_filter subplugin
 */
static void
custom_close (const GstTensorFilterProperties * prop, void **private_data)
{
  runtime_data *rd = *private_data;
  UNUSED (prop);
  g_free (rd);
  *private_data = NULL;
}


/**
 * @brief Callback required by tensor_filter subplugin
 */
static int
custom_invoke (const GstTensorFilterFramework * self,
    GstTensorFilterProperties * prop, void *private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  runtime_data *rd = (runtime_data *) private_data;
  UNUSED (self);

  /* Internal Logic Error */
  g_assert (rd && rd->model && rd->model->func);

  return rd->model->func (rd->model->data, prop, input, output);
}

/**
 * @brief V1 tensor-filter wrapper callback function, "getFrameworkInfo"
 */
static int
custom_getFrameworkInfo (const GstTensorFilterFramework * self,
    const GstTensorFilterProperties * prop, void *private_data,
    GstTensorFilterFrameworkInfo * fw_info)
{
  UNUSED (self);
  UNUSED (prop);
  UNUSED (private_data);
  fw_info->name = fw_name;
  fw_info->allow_in_place = 0;
  fw_info->allocate_in_invoke = 0;
  fw_info->run_without_model = 1;
  fw_info->verify_model_path = 0;
  fw_info->hw_list = NULL;
  fw_info->num_hw = 0;

  return 0;
}

/**
 * @brief C V1 tensor-filter wrapper callback function, "getModelInfo"
 */
static int
custom_getModelInfo (const GstTensorFilterFramework * self,
    const GstTensorFilterProperties * prop, void *private_data,
    model_info_ops ops, GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  runtime_data *rd = private_data;
  UNUSED (self);
  UNUSED (prop);
  UNUSED (ops);

  gst_tensors_info_copy (in_info, &rd->model->in_info);
  gst_tensors_info_copy (out_info, &rd->model->out_info);

  return 0;
}

/**
 * @brief C V1 tensor-filter wrapper callback function, "eventHandler"
 */
static int
custom_eventHandler (const GstTensorFilterFramework * self,
    const GstTensorFilterProperties * prop, void *private_data, event_ops ops,
    GstTensorFilterFrameworkEventData * data)
{
  UNUSED (self);
  UNUSED (private_data);
  UNUSED (ops);
  UNUSED (data);
  UNUSED (prop);

  return 0;
}

static GstTensorFilterFramework NNS_support_custom_easy = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V1,
  .open = custom_open,
  .close = custom_close,
  .invoke = custom_invoke,
  .getFrameworkInfo = custom_getFrameworkInfo,
  .getModelInfo = custom_getModelInfo,
  .eventHandler = custom_eventHandler,
  .subplugin_data = NULL,
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_custom_easy (void)
{
  nnstreamer_filter_probe (&NNS_support_custom_easy);
}

/** @brief Destruct the subplugin */
void
fini_filter_custom_easy (void)
{
  nnstreamer_filter_exit (NNS_support_custom_easy.name);
}

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
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <errno.h>
#include <glib.h>
#include <tensor_filter_custom_easy.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_subplugin.h>

void init_filter_custom_easy (void) __attribute__ ((constructor));
void fini_filter_custom_easy (void) __attribute__ ((destructor));

/**
 * @brief internal_data
 */
typedef struct _internal_data
{
  NNS_custom_invoke func;
  const GstTensorsInfo *in_info;
  const GstTensorsInfo *out_info;
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

  ptr = g_new0 (internal_data, 1);

  if (!ptr)
    return -ENOMEM;

  ptr->func = func;
  ptr->data = data;
  ptr->in_info = in_info;
  ptr->out_info = out_info;

  if (register_subplugin (NNS_EASY_CUSTOM_FILTER, modelname, ptr) == TRUE)
    return 0;

  g_free (ptr);

  return -EINVAL;
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
    g_critical
        ("Cannot find the easy-custom model, \"%s\". You should provide a valid model name of easy-custom.",
        prop->model_files[0]);
    g_free (rd);
    return -EINVAL;
  }

  g_assert (rd->model->func);
  g_assert (rd->model->in_info);
  g_assert (rd->model->out_info);

  *private_data = rd;
  return 0;
}

/**
 * @brief Callback required by tensor_filter subplugin
 */
static int
custom_invoke (const GstTensorFilterProperties * prop,
    void **private_data, const GstTensorMemory * input,
    GstTensorMemory * output)
{
  runtime_data *rd = *private_data;
  g_assert (rd && rd->model && rd->model->func);

  return rd->model->func (rd->model->data, prop, input, output);
}

/**
 * @brief Callback required by tensor_filter subplugin
 */
static int
custom_getInputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  runtime_data *rd = *private_data;
  g_assert (rd && rd->model && rd->model->in_info);
  gst_tensors_info_copy (info, rd->model->in_info);
  return 0;
}

/**
 * @brief Callback required by tensor_filter subplugin
 */
static int
custom_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  runtime_data *rd = *private_data;
  g_assert (rd && rd->model && rd->model->out_info);
  gst_tensors_info_copy (info, rd->model->out_info);
  return 0;
}

/**
 * @brief Callback required by tensor_filter subplugin
 */
static void
custom_close (const GstTensorFilterProperties * prop, void **private_data)
{
  runtime_data *rd = *private_data;
  g_free (rd);
  *private_data = NULL;
}

static char name_str[] = "custom-easy";
static GstTensorFilterFramework NNS_support_custom_easy = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .name = name_str,
  .allow_in_place = FALSE,      /* custom cannot support in-place. */
  .allocate_in_invoke = FALSE,  /* we allocate output buffers for you. */
  .run_without_model = FALSE,   /* we need a func to run. */
  .invoke_NN = custom_invoke,

  .getInputDimension = custom_getInputDim,
  .getOutputDimension = custom_getOutputDim,
  .setInputDimension = NULL,    /* NYI: we don't support flexible dim, yet */
  .open = custom_open,
  .close = custom_close,
  .destroyNotify = NULL,        /* No need. We don't support "allocate_in_invoke." */
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

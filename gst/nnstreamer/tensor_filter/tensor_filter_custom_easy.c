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

static const char fw_name_custom_easy[] = "custom-easy";

/** @brief Guards custom_easy_models and keeps it in step with the registry. Never held while the registry may load a .so. */
static GMutex custom_easy_lock;

/** @brief Registered models by name, looked up without loading a .so */
static GHashTable *custom_easy_models = NULL;

/**
 * @brief internal_data
 */
typedef struct _internal_data
{
  NNS_custom_invoke func;
  GstTensorsInfo in_info;
  GstTensorsInfo out_info;
  void *data; /**< The easy-filter writer's data */
  NNS_custom_invoke_dynamic func_dynamic;
  gint refcount; /**< One for the registry and one for each opened filter */
} internal_data;

/**
 * @brief The easy-filter user's data
 */
typedef struct
{
  internal_data *model;
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
 * @brief Internal function to drop a reference of internal data, releasing it with the last one.
 */
static void
custom_unref_internal_data (internal_data * data)
{
  if (data && g_atomic_int_dec_and_test (&data->refcount))
    custom_free_internal_data (data);
}

/**
 * @brief Internal function to register internal data under the model name.
 * @return 0 if success. -EINVAL if error, releasing the data.
 */
static int
custom_register_internal_data (const char *modelname, internal_data * data)
{
  gboolean registered;

  g_mutex_lock (&custom_easy_lock);
  registered = register_subplugin (NNS_EASY_CUSTOM_FILTER, modelname, data);
  if (registered) {
    if (!custom_easy_models)
      custom_easy_models = g_hash_table_new_full (g_str_hash, g_str_equal,
          g_free, NULL);
    g_hash_table_insert (custom_easy_models, g_strdup (modelname), data);
  }
  g_mutex_unlock (&custom_easy_lock);

  if (registered)
    return 0;

  custom_free_internal_data (data);
  return -EINVAL;
}

/**
 * @brief Internal function to find a registered model and take a reference of it.
 * @note The registry may load a .so that registers the name, and a .so constructor or destructor may call the register APIs. So the lookup that loads runs without the lock, and the lock only covers the table lookup and the reference.
 */
static internal_data *
custom_ref_internal_data (const char *modelname)
{
  internal_data *data = NULL;

  get_subplugin (NNS_EASY_CUSTOM_FILTER, modelname);

  g_mutex_lock (&custom_easy_lock);
  if (custom_easy_models)
    data = g_hash_table_lookup (custom_easy_models, modelname);
  if (data)
    g_atomic_int_inc (&data->refcount);
  g_mutex_unlock (&custom_easy_lock);

  return data;
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
  ptr->refcount = 1;
  gst_tensors_info_copy (&ptr->in_info, in_info);
  gst_tensors_info_copy (&ptr->out_info, out_info);

  return custom_register_internal_data (modelname, ptr);
}


/**
 * @brief Register the custom-easy tensor function. More info in .h
 * @return 0 if success. -ERRNO if error.
 */
int
NNS_custom_easy_dynamic_register (const char *modelname,
    NNS_custom_invoke_dynamic func, void *data, const GstTensorsInfo * in_info)
{
  internal_data *ptr;

  if (!func || !in_info)
    return -EINVAL;

  if (!gst_tensors_info_validate (in_info))
    return -EINVAL;

  ptr = g_new0 (internal_data, 1);

  if (!ptr)
    return -ENOMEM;

  ptr->func_dynamic = func;
  ptr->data = data;
  ptr->refcount = 1;
  gst_tensors_info_copy (&ptr->in_info, in_info);

  return custom_register_internal_data (modelname, ptr);
}

/**
 * @brief Unregister the custom-easy tensor function.
 * @return 0 if success. -EINVAL if invalid model name.
 */
int
NNS_custom_easy_unregister (const char *modelname)
{
  internal_data *ptr = NULL;
  gboolean unregistered = FALSE;

  if (modelname)
    get_subplugin (NNS_EASY_CUSTOM_FILTER, modelname);

  g_mutex_lock (&custom_easy_lock);
  if (custom_easy_models && modelname)
    ptr = g_hash_table_lookup (custom_easy_models, modelname);
  if (ptr) {
    unregistered = unregister_subplugin (NNS_EASY_CUSTOM_FILTER, modelname);
    if (unregistered)
      g_hash_table_remove (custom_easy_models, modelname);
  }
  g_mutex_unlock (&custom_easy_lock);

  if (!unregistered) {
    ml_loge ("Failed to unregister custom filter %s.", modelname);
    return -EINVAL;
  }

  /* opened filters keep the data until they are closed */
  custom_unref_internal_data (ptr);
  return 0;
}

/**
 * @brief Callback required by tensor_filter subplugin
 */
static int
custom_open (const GstTensorFilterProperties * prop, void **private_data)
{
  runtime_data *rd;

  if (!prop->model_files || prop->num_models < 1 || !prop->model_files[0]
      || prop->model_files[0][0] == '\0') {
    ml_loge ("The easy-custom filter requires a registered model name. "
        "Set the 'model' property of tensor_filter.");
    return -EINVAL;
  }

  rd = g_new (runtime_data, 1);
  if (!rd)
    return -ENOMEM;
  rd->model = custom_ref_internal_data (prop->model_files[0]);

  if (NULL == rd->model) {
    ml_loge
        ("Cannot find the easy-custom model, \"%s\". You should provide a valid model name of easy-custom.",
        prop->model_files[0]);
    goto errorreturn;
  }

  if (NULL == rd->model->func && NULL == rd->model->func_dynamic) {
    ml_logf
        ("A custom-easy filter, \"%s\", should provide invoke function body, 'func'. A null-ptr is supplied instead.\n",
        prop->model_files[0]);
    goto errorreturn;
  }

  if (!prop->invoke_dynamic && rd->model->func_dynamic) {
    ml_loge
        ("Not matched easy-custom model, \"%s\". "
        "Dynamic invoke option is disabled but you registered dynamic invoke function. "
        "You should register model using NNS_custom_easy_register.",
        prop->model_files[0]);
    goto errorreturn;
  }

  if (prop->invoke_dynamic && rd->model->func) {
    ml_loge
        ("Not matched easy-custom model, \"%s\". "
        "If want to use dynamic invoke, register model using NNS_custom_easy_dynamic_register.",
        prop->model_files[0]);
    goto errorreturn;
  }

  if (!gst_tensors_info_validate (&rd->model->in_info)) {
    ml_logf
        ("A custom-easy filter, \"%s\", should provide input stream metadata, 'in_info'.\n",
        prop->model_files[0]);
    goto errorreturn;
  }

  if (rd->model->func && !gst_tensors_info_validate (&rd->model->out_info)) {
    ml_logf
        ("A custom-easy filter, \"%s\", should provide output stream metadata, 'out_info'.\n",
        prop->model_files[0]);
    goto errorreturn;
  }

  *private_data = rd;
  return 0;
errorreturn:
  custom_unref_internal_data (rd->model);
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
  if (rd)
    custom_unref_internal_data (rd->model);
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
  int ret = 0;
  runtime_data *rd = (runtime_data *) private_data;
  UNUSED (self);

  /* Internal Logic Error */
  g_assert (rd && rd->model);

  if (!prop->invoke_dynamic) {
    if (!rd->model->func) {
      ml_loge
          ("Custom filter function is not registered. Register the function using `NNS_custom_easy_register`.");
      return -1;
    }
    return rd->model->func (rd->model->data, prop, input, output);
  } else {
    if (!rd->model->func_dynamic) {
      ml_loge
          ("Dynamic invoke is enabled but dynamic custom filter function is not registered. Register the function using `NNS_custom_easy_dynamic_register`.");
      return -1;
    }
    return rd->model->func_dynamic (rd->model->data, &prop->input_meta,
        &prop->output_meta, input, output);
  }

  return ret;
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
  fw_info->name = fw_name_custom_easy;
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

  if (ops == GET_IN_OUT_INFO) {
    gst_tensors_info_copy (in_info, &rd->model->in_info);
    gst_tensors_info_copy (out_info, &rd->model->out_info);
    return 0;
  }

  return -ENOENT;
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

  return -ENOENT;
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

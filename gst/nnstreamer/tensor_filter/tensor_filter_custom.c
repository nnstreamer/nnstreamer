/**
 * GStreamer Tensor_Filter Module
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	tensor_filter_custom.c
 * @date	01 Jun 2018
 * @brief	Custom tensor post-processing interface for NNStreamer suite between NN developer-plugins and NNstreamer.
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (custom) for tensor_filter.
 * Fill in "GstTensorFilterFramework" for tensor_filter.h/c
 *
 */

#include <errno.h>
#include <glib.h>
#include <gmodule.h>

#include "tensor_filter_custom.h"
#include "nnstreamer_plugin_api_filter.h"
#include "nnstreamer_conf.h"
#include <nnstreamer_log.h>

void init_filter_custom (void) __attribute__((constructor));
void fini_filter_custom (void) __attribute__((destructor));

static const gchar *custom_accl_support[] = {
  ACCL_AUTO_STR,
  ACCL_DEFAULT_STR,
  NULL
};

/**
 * @brief internal_data
 */
struct _internal_data
{
  GModule *module;
  NNStreamer_custom_class *methods;

  void *customFW_private_data;
};
typedef struct _internal_data internal_data;

/**
 * @brief Load the custom library. Will skip loading if it's already loaded.
 * @return 0 if successfully loaded. 1 if skipped (already loaded). -1 if error
 */
static int
custom_loadlib (const GstTensorFilterProperties * prop, void **private_data)
{
  internal_data *ptr;
  gpointer custom_cls;

  if (*private_data != NULL) {
    /** @todo : Check the integrity of filter->data and filter->model_file, nnfw */
    ml_loge ("Init is called but it is already initialized.\n");
    return -EINVAL;
  }

  if (!prop->model_files || prop->num_models != 1 ||
      !prop->model_files[0] || prop->model_files[0][0] == '\0') {
    /* The .so file path is not given */
    ml_logw ("Custom filter file is not given.\n");
    return -EINVAL;
  }

  if (!nnsconf_validate_file (NNSCONF_PATH_CUSTOM_FILTERS,
          prop->model_files[0])) {
    /* Cannot load the library */
    ml_logw ("Custom filter file %s is invalid.\n", prop->model_files[0]);
    return -EINVAL;
  }

  ptr = *private_data = g_new0 (internal_data, 1);
  if (ptr == NULL) {
    ml_loge ("Failed to allocate memory for custom filter.\n");
    return -ENOMEM;
  }

  /* Load .so if this is the first time for this instance. */
  ptr->module = g_module_open (prop->model_files[0], 0);
  if (!ptr->module) {
    g_free (ptr);
    *private_data = NULL;
    ml_logw ("Cannot load custom filter file %s.\n", prop->model_files[0]);
    return -EINVAL;
  }

  if (!g_module_symbol (ptr->module, "NNStreamer_custom", &custom_cls)) {
    ml_loge ("tensor_filter_custom:loadlib error: %s\n", g_module_error ());
    g_module_close (ptr->module);
    g_free (ptr);
    *private_data = NULL;
    return -EINVAL;
  }

  ptr->methods = *(NNStreamer_custom_class **) custom_cls;

  if (NULL == ptr->methods->initfunc) {
    ml_loge ("tensor_filter_custom (%s) requires a valid 'initfunc'.",
        prop->model_files[0]);
    return -EINVAL;
  }

  ptr->customFW_private_data = ptr->methods->initfunc (prop);

  /* After init func, (getInput XOR setInput) && (getOutput XOR setInput) must hold! */
  /** @todo Double check if this check is really required and safe */
  if (!ptr->methods->getInputDim != !ptr->methods->setInputDim &&
      !ptr->methods->getOutputDim != !ptr->methods->setInputDim)
    return 0;                   /* it holds! */

  ml_loge
      ("tensor_filter_custom (%s) requires input/output dimension callbacks.",
      prop->model_files[0]);
  return -EINVAL;
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 */
static int
custom_open (const GstTensorFilterProperties * prop, void **private_data)
{
  int retval = custom_loadlib (prop, private_data);
  internal_data *ptr;

  /* This must be called only once */
  if (retval != 0)
    return retval;

  ptr = *private_data;

  /* One of the two is defined. (but not both) */
  if (!ptr->methods->invoke != !ptr->methods->allocate_invoke)
    return 0;

  ml_loge
      ("An invoke callback is not given or both invoke functions are given. Cannot load %s.\n",
      prop->model_files[0]);
  return -EINVAL;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param prop The properties of parent object
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
static int
custom_invoke (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  internal_data *ptr;

  /* Actually, tensor_filter must have called getInput/OutputDim first. */
  g_return_val_if_fail (*private_data != NULL, -EINVAL);
  g_return_val_if_fail (input != NULL, -EINVAL);
  g_return_val_if_fail (output != NULL, -EINVAL);

  ptr = *private_data;
  if (ptr->methods->invoke) {
    return ptr->methods->invoke (ptr->customFW_private_data, prop,
        input, output);
  } else if (ptr->methods->allocate_invoke) {
    return ptr->methods->allocate_invoke (ptr->customFW_private_data,
        prop, input, output);
  } else {
    return -1;
  }
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 */
static int
custom_getInputDim (const GstTensorFilterProperties * prop, void **private_data,
    GstTensorsInfo * info)
{
  internal_data *ptr;

  ptr = *private_data;
  g_return_val_if_fail (ptr != NULL, -EINVAL);
  g_return_val_if_fail (info != NULL, -EINVAL);

  if (ptr->methods->getInputDim == NULL)
    return -ENOENT;

  return ptr->methods->getInputDim (ptr->customFW_private_data, prop, info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 */
static int
custom_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  internal_data *ptr;

  ptr = *private_data;
  g_return_val_if_fail (ptr != NULL, -EINVAL);
  g_return_val_if_fail (info != NULL, -EINVAL);

  if (ptr->methods->getOutputDim == NULL)
    return -ENOENT;

  return ptr->methods->getOutputDim (ptr->customFW_private_data, prop, info);
}

/**
 * @brief The set-input-dim callback for GstTensorFilterFramework
 */
static int
custom_setInputDim (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  internal_data *ptr;

  ptr = *private_data;
  g_return_val_if_fail (ptr != NULL, -EINVAL);
  g_return_val_if_fail (in_info != NULL, -EINVAL);
  g_return_val_if_fail (out_info != NULL, -EINVAL);

  if (ptr->methods->setInputDim == NULL)
    return -ENOENT;

  return ptr->methods->setInputDim (ptr->customFW_private_data,
      prop, in_info, out_info);
}

/**
 * @brief Free privateData and move on.
 */
static void
custom_close (const GstTensorFilterProperties * prop, void **private_data)
{
  internal_data *ptr = *private_data;

  g_return_if_fail (ptr != NULL);

  ptr->methods->exitfunc (ptr->customFW_private_data, prop);
  g_free (ptr);
  *private_data = NULL;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 */
static void
custom_destroyNotify (void **private_data, void *data)
{
  internal_data *ptr = *private_data;

  if (ptr && ptr->methods->allocate_invoke && ptr->methods->destroy_notify &&
      data) {
    ptr->methods->destroy_notify (data);
  } else {
    g_free (data);
  }
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 */
static int
custom_allocateInInvoke (void **private_data)
{
  internal_data *ptr = *private_data;

  if (ptr && ptr->methods->allocate_invoke) {
    return 0;
  }

  return -EINVAL;
}

/**
 * @brief Check support of the backend
 */
static int
custom_checkAvailability (accl_hw hw)
{
  if (g_strv_contains (custom_accl_support, get_accl_hw_str (hw)))
    return 0;

  return -ENOENT;
}

static gchar filter_subplugin_custom[] = "custom";

static GstTensorFilterFramework NNS_support_custom = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .name = filter_subplugin_custom,
  .allow_in_place = FALSE,      /* custom cannot support in-place (output == input). */
  .allocate_in_invoke = TRUE,   /* GstTensorFilter allocates output buffers */
  .run_without_model = FALSE,   /* custom needs a so file */
  .invoke_NN = custom_invoke,
  .handleEvent = NULL,          /* do not set event function for custom filter */
  .getInputDimension = custom_getInputDim,
  .getOutputDimension = custom_getOutputDim,
  .setInputDimension = custom_setInputDim,
  .open = custom_open,
  .close = custom_close,
  .destroyNotify = custom_destroyNotify,        /* if custom filter model supports allocate_in_invoke, this will be set from custom filter. */
  .allocateInInvoke = custom_allocateInInvoke,
  .checkAvailability = custom_checkAvailability,
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_custom (void)
{
  nnstreamer_filter_probe (&NNS_support_custom);
}

/** @brief Destruct the subplugin */
void
fini_filter_custom (void)
{
  nnstreamer_filter_exit (NNS_support_custom.name);
}

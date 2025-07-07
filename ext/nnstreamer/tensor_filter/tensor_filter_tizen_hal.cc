/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer tensor_filter, sub-plugin for Tizen HAL
 * Copyright (C) 2025 Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 */
/**
 * @file      tensor_filter_tizen_hal.cc
 * @date      15 Jan 2025
 * @brief     NNStreamer tensor-filter sub-plugin for Tizen HAL
 * @see       http://github.com/nnstreamer/nnstreamer
 * @author    Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug       No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (Tizen HAL) for tensor_filter.
 */

#include <iostream>
#include <string>

#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

#include <hal-ml.h>

namespace nnstreamer
{
namespace tensor_filter_tizen_hal
{
extern "C" {
void init_filter_tizen_hal (void) __attribute__ ((constructor));
void fini_filter_tizen_hal (void) __attribute__ ((destructor));
}

/** @brief tensor-filter-subplugin concrete class for Tizen HAL */
class tizen_hal_subplugin final : public tensor_filter_subplugin
{
  private:
  static const char *fw_name;
  static tizen_hal_subplugin *registeredRepresentation;
  static const GstTensorFilterFrameworkInfo framework_info;
  hal_ml_h hal_handle;
  gchar *backend_name;

  public:
  static void init_filter_tizen_hal ();
  static void fini_filter_tizen_hal ();

  tizen_hal_subplugin ();
  ~tizen_hal_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

const char *tizen_hal_subplugin::fw_name = "tizen-hal";

/**
 * @brief Constructor for tizen_hal_subplugin.
 */
tizen_hal_subplugin::tizen_hal_subplugin ()
    : tensor_filter_subplugin (), hal_handle (nullptr), backend_name (nullptr)
{
}

/**
 * @brief Destructor for Tizen HAL subplugin.
 */
tizen_hal_subplugin::~tizen_hal_subplugin ()
{
  if (backend_name) {
    g_free (backend_name);
    backend_name = nullptr;
  }

  if (hal_handle) {
    hal_ml_destroy (hal_handle);
    hal_handle = nullptr;
  }
}

/**
 * @brief Method to get empty object.
 */
tensor_filter_subplugin &
tizen_hal_subplugin::getEmptyInstance ()
{
  return *(new tizen_hal_subplugin ());
}

/**
 * @brief Method to prepare/configure Tizen HAL instance.
 */
void
tizen_hal_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  /* parse backend from custom prop */
  if (!prop->custom_properties)
    throw std::invalid_argument ("Custom properties are not given. Provide a valid backend.");

  /* prop->custom_properties has value of "backend:backend_name,key1:val1,key2:val2" */
  gchar **options = g_strsplit (prop->custom_properties, ",", -1);

  for (guint op = 0; op < g_strv_length (options); ++op) {
    gchar **option = g_strsplit (options[op], ":", -1);

    if (g_strv_length (option) >= 2) {
      g_strstrip (option[0]);
      g_strstrip (option[1]);

      if (g_ascii_strcasecmp (option[0], "backend") == 0) {
        backend_name = g_strdup (option[1]);
      }
    }

    g_strfreev (option);
  }

  g_strfreev (options);

  if (backend_name == NULL) {
    throw std::invalid_argument (
        "Invalid custom properties format. 'backend:backend_name' must be specified.");
  }

  nns_logi ("Using backend: %s", backend_name);

  int ret = hal_ml_create (backend_name, &hal_handle);
  if (ret == HAL_ML_ERROR_NOT_SUPPORTED) {
    throw std::invalid_argument ("Given backend is not supported.");
  }

  if (ret == HAL_ML_ERROR_RUNTIME_ERROR) {
    throw std::invalid_argument ("Failed to initialize backend.");
  }

  if (ret == HAL_ML_ERROR_INVALID_PARAMETER) {
    throw std::invalid_argument ("Tizen HAL got invalid arguments.");
  }

  if (ret != HAL_ML_ERROR_NONE) {
    throw std::runtime_error ("Tizen HAL ML unknown error occurred.");
  }

  hal_ml_param_h param = nullptr;
  ret = hal_ml_param_create (&param);
  if (ret != HAL_ML_ERROR_NONE) {
    throw std::runtime_error ("Failed to create hal_ml_param for configuring instance");
  }

  ret = hal_ml_param_set (param, "properties", (void *) prop);
  if (ret != HAL_ML_ERROR_NONE) {
    hal_ml_param_destroy (param);
    throw std::runtime_error ("Failed to set 'properties' parameter for Tizen HAL configuration");
  }

  ret = hal_ml_request (hal_handle, "configure_instance", param);
  if (ret != HAL_ML_ERROR_NONE) {
    hal_ml_param_destroy (param);
    throw std::runtime_error ("Failed to configure Tizen HAL instance");
  }

  hal_ml_param_destroy (param);
}

/**
 * @brief Method to execute the model.
 */
void
tizen_hal_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  if (!input) {
    throw std::runtime_error ("Invalid input buffer, it is NULL.");
  }
  if (!output) {
    throw std::runtime_error ("Invalid output buffer, it is NULL.");
  }

  int ret = hal_ml_request_invoke (hal_handle, input, output);
  if (ret != HAL_ML_ERROR_NONE) {
    throw std::runtime_error ("Failed to invoke Tizen HAL model execution");
  }
}

/**
 * @brief Method to get the information of Tizen HAL subplugin.
 */
void
tizen_hal_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = fw_name;

  if (!hal_handle) {
    nns_logw ("HAL backend is not configured.");
    return;
  }

  hal_ml_param_h param = nullptr;
  int ret = hal_ml_param_create (&param);
  if (ret != HAL_ML_ERROR_NONE) {
    throw std::runtime_error ("Failed to create hal_ml_param for getting framework info");
  }

  ret = hal_ml_param_set (param, "framework_info", (void *) std::addressof (info));
  if (ret != HAL_ML_ERROR_NONE) {
    hal_ml_param_destroy (param);
    throw std::runtime_error ("Failed to set 'framework_info' parameter");
  }

  ret = hal_ml_request (hal_handle, "get_framework_info", param);
  if (ret != HAL_ML_ERROR_NONE) {
    hal_ml_param_destroy (param);
    throw std::runtime_error ("Failed to get framework info");
  }

  hal_ml_param_destroy (param);
}

/**
 * @brief Method to get the model information.
 */
int
tizen_hal_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (!hal_handle) {
    nns_loge ("HAL backend is not configured.");
    return -1;
  }

  hal_ml_param_h param = nullptr;
  if (hal_ml_param_create (&param) != HAL_ML_ERROR_NONE
      || hal_ml_param_set (param, "ops", (void *) &ops) != HAL_ML_ERROR_NONE
      || hal_ml_param_set (param, "in_info", (void *) std::addressof (in_info)) != HAL_ML_ERROR_NONE
      || hal_ml_param_set (param, "out_info", (void *) std::addressof (out_info))
             != HAL_ML_ERROR_NONE) {
    hal_ml_param_destroy (param);
    nns_loge ("Failed to set parameters for getModelInfo");
    return -1;
  }

  int ret = hal_ml_request (hal_handle, "get_model_info", param);
  hal_ml_param_destroy (param);

  if (ret != HAL_ML_ERROR_NONE) {
    return (ret == HAL_ML_ERROR_NOT_SUPPORTED) ? -ENOENT : -1;
  }

  return 0;
}

/**
 * @brief Method to handle events.
 */
int
tizen_hal_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  if (!hal_handle) {
    nns_loge ("HAL backend is not configured.");
    return -1;
  }

  hal_ml_param_h param = nullptr;
  if (hal_ml_param_create (&param) != HAL_ML_ERROR_NONE
      || hal_ml_param_set (param, "ops", (void *) &ops) != HAL_ML_ERROR_NONE
      || hal_ml_param_set (param, "data", (void *) std::addressof (data)) != HAL_ML_ERROR_NONE) {
    hal_ml_param_destroy (param);
    nns_loge ("Failed to set parameters for event handler");
    return -1;
  }

  int ret = hal_ml_request (hal_handle, "event_handler", param);
  hal_ml_param_destroy (param);

  if (ret != HAL_ML_ERROR_NONE) {
    return (ret == HAL_ML_ERROR_NOT_SUPPORTED) ? -ENOENT : -1;
  }

  return 0;
}

tizen_hal_subplugin *tizen_hal_subplugin::registeredRepresentation = nullptr;

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
tizen_hal_subplugin::init_filter_tizen_hal (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<tizen_hal_subplugin> ();
}

/** @brief Destruct the subplugin */
void
tizen_hal_subplugin::fini_filter_tizen_hal (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/**
 * @brief Register the sub-plugin for Tizen HAL.
 */
void
init_filter_tizen_hal ()
{
  tizen_hal_subplugin::init_filter_tizen_hal ();
}

/**
 * @brief Destruct the sub-plugin for Tizen HAL.
 */
void
fini_filter_tizen_hal ()
{
  tizen_hal_subplugin::fini_filter_tizen_hal ();
}

} /* namespace tensor_filter_tizen_hal */
} /* namespace nnstreamer */

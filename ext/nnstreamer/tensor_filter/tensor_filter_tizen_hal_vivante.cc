/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer tensor_filter, sub-plugin for Vivante
 * Copyright (C) 2025 Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 */
/**
 * @file      tensor_filter_tizen_hal_vivante.cc
 * @date      15 Jan 2025
 * @brief     NNStreamer tensor-filter sub-plugin for Tizen HAL Vivante
 * @see       http://github.com/nnstreamer/nnstreamer
 * @author    Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug       No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (Tizen HAL Vivante) for tensor_filter.
 */

#include <iostream>
#include <string>
#include <vector>

#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

#include <hal-ml.h>


namespace nnstreamer
{
namespace tensor_filter_vivante_tizen_hal
{
extern "C" {
void init_filter_vivante_tizen_hal (void) __attribute__ ((constructor));
void fini_filter_vivante_tizen_hal (void) __attribute__ ((destructor));
}

/** @brief tensor-filter-subplugin concrete class for Vivante */
class vivante_tizen_hal_subplugin final : public tensor_filter_subplugin
{
  private:
  static const char *fw_name;
  static vivante_tizen_hal_subplugin *registeredRepresentation;
  static const GstTensorFilterFrameworkInfo framework_info;

  hal_ml_h hal_handle;

  public:
  static void init_filter_vivante_tizen_hal ();
  static void fini_filter_vivante_tizen_hal ();

  vivante_tizen_hal_subplugin ();
  ~vivante_tizen_hal_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

const char *vivante_tizen_hal_subplugin::fw_name = "vivante";

/**
 * @brief Constructor for vivante_tizen_hal_subplugin.
 */
vivante_tizen_hal_subplugin::vivante_tizen_hal_subplugin ()
    : tensor_filter_subplugin (), hal_handle (nullptr)
{
  int ret = hal_ml_create ("vivante", &hal_handle);
  if (ret == HAL_ML_ERROR_NOT_SUPPORTED) {
    throw std::invalid_argument ("Vivante is not supported");
  }
  if (ret != HAL_ML_ERROR_NONE) {
    throw std::runtime_error ("Failed to initialize Vivante HAL ML");
  }
}

/**
 * @brief Destructor for vivante subplugin.
 */
vivante_tizen_hal_subplugin::~vivante_tizen_hal_subplugin ()
{
  if (hal_handle)
    hal_ml_destroy (hal_handle);
}

/**
 * @brief Method to get empty object.
 */
tensor_filter_subplugin &
vivante_tizen_hal_subplugin::getEmptyInstance ()
{
  return *(new vivante_tizen_hal_subplugin ());
}


/**
 * @brief Method to prepare/configure Vivante instance.
 */
void
vivante_tizen_hal_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  hal_ml_param_h param = nullptr;

  int ret = hal_ml_param_create (&param);
  if (ret != HAL_ML_ERROR_NONE) {
    throw std::runtime_error ("Failed to create hal_ml_param for configuring instance");
  }

  ret = hal_ml_param_set (param, "properties", (void *) prop);
  if (ret != HAL_ML_ERROR_NONE) {
    hal_ml_param_destroy (param);
    throw std::runtime_error ("Failed to set 'properties' parameter for Vivante configuration");
  }

  ret = hal_ml_request (hal_handle, "configure_instance", param);
  if (ret != HAL_ML_ERROR_NONE) {
    hal_ml_param_destroy (param);
    throw std::runtime_error ("Failed to configure Vivante instance");
  }

  hal_ml_param_destroy (param);
}

/**
 * @brief Method to execute the model.
 */
void
vivante_tizen_hal_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  if (!input)
    throw std::runtime_error ("Invalid input buffer, it is NULL.");
  if (!output)
    throw std::runtime_error ("Invalid output buffer, it is NULL.");

  int ret = hal_ml_request_invoke (hal_handle, input, output);
  if (ret != HAL_ML_ERROR_NONE) {
    throw std::runtime_error ("Failed to invoke Vivante model");
  }
}

/**
 * @brief Method to get the information of Vivante subplugin.
 */
void
vivante_tizen_hal_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
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
  info.name = fw_name;
}

/**
 * @brief Method to get the model information.
 */
int
vivante_tizen_hal_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  hal_ml_param_h param = nullptr;
  int ret = hal_ml_param_create (&param);

  if (ret != HAL_ML_ERROR_NONE) {
    nns_loge ("Failed to create hal_ml_param");
    return -1;
  }

  if (hal_ml_param_set (param, "ops", (void *) &ops) != HAL_ML_ERROR_NONE
      || hal_ml_param_set (param, "in_info", (void *) std::addressof (in_info)) != HAL_ML_ERROR_NONE
      || hal_ml_param_set (param, "out_info", (void *) std::addressof (out_info))
             != HAL_ML_ERROR_NONE) {
    hal_ml_param_destroy (param);
    nns_loge ("Failed to set parameters for getModelInfo");
    return -1;
  }

  ret = hal_ml_request (hal_handle, "get_model_info", param);
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
vivante_tizen_hal_subplugin::eventHandler (
    event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  hal_ml_param_h param = nullptr;
  int ret = hal_ml_param_create (&param);

  if (ret != HAL_ML_ERROR_NONE) {
    nns_loge ("Failed to create hal_ml_param");
    return -1;
  }

  if (hal_ml_param_set (param, "ops", (void *) &ops) != HAL_ML_ERROR_NONE
      || hal_ml_param_set (param, "data", (void *) std::addressof (data)) != HAL_ML_ERROR_NONE) {
    hal_ml_param_destroy (param);
    nns_loge ("Failed to set parameters for event handler");
    return -1;
  }

  ret = hal_ml_request (hal_handle, "event_handler", param);
  hal_ml_param_destroy (param);

  if (ret != HAL_ML_ERROR_NONE) {
    return (ret == HAL_ML_ERROR_NOT_SUPPORTED) ? -ENOENT : -1;
  }

  return 0;
}

vivante_tizen_hal_subplugin *vivante_tizen_hal_subplugin::registeredRepresentation = nullptr;

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
vivante_tizen_hal_subplugin::init_filter_vivante_tizen_hal (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<vivante_tizen_hal_subplugin> ();
}

/** @brief Destruct the subplugin */
void
vivante_tizen_hal_subplugin::fini_filter_vivante_tizen_hal (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/**
 * @brief Register the sub-plugin for Vivante.
 */
void
init_filter_vivante_tizen_hal ()
{
  vivante_tizen_hal_subplugin::init_filter_vivante_tizen_hal ();
}

/**
 * @brief Destruct the sub-plugin for Vivante.
 */
void
fini_filter_vivante_tizen_hal ()
{
  vivante_tizen_hal_subplugin::fini_filter_vivante_tizen_hal ();
}

} /* namespace tensor_filter_vivante_tizen_hal */
} /* namespace nnstreamer */

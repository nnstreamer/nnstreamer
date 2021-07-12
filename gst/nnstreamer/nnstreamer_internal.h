/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	nnstreamer_internal.h
 * @date	28 Jan 2021
 * @brief	Internal header for NNStreamer plugins and native APIs.
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __NNSTREAMER_INTERNAL_H__
#define __NNSTREAMER_INTERNAL_H__

#include <glib.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>

G_BEGIN_DECLS

/**
 * @brief Get the custom configuration value from .ini and envvar.
 * @detail For predefined configurations defined in this header,
 *         use the given enum for faster configuration processing.
 *         For custom configurations not defined in this header,
 *         you may use this API to access your own custom configurations.
 *         Configuration values may be loaded only once during runtime,
 *         thus, if the values are changed in run-time, the changes are
 *         not guaranteed to be reflected.
 *         The ENVVAR is supposed to be NNSTREAMER_${group}_${key}, which
 *         has higher priority than the .ini configuration.
 *         Be careful not to use special characters in group name ([, ], _).
 * @param[in] group The group name, [group], in .ini file.
 * @param[in] key The key name, key = value, in .ini file.
 * @return The newly allocated string. A caller must free it. NULL if it's not available.
 */
extern gchar *
nnsconf_get_custom_value_string (const gchar * group, const gchar * key);

/**
 * @brief Get the custom configuration value from .ini and envvar.
 * @detail For predefined configurations defined in this header,
 *         use the given enum for faster configuration processing.
 *         For custom configurations not defined in this header,
 *         you may use this API to access your own custom configurations.
 *         Configuration values may be loaded only once during runtime,
 *         thus, if the values are changed in run-time, the changes are
 *         not guaranteed to be reflected.
 *         The ENVVAR is supposed to be NNSTREAMER_${group}_${key}, which
 *         has higher priority than the .ini configuration.
 *         Be careful not to use special characters in group name ([, ], _).
 * @param[in] group The group name, [group], in .ini file.
 * @param[in] key The key name, key = value, in .ini file.
 * @param[in] def The default return value in case there is no value available.
 * @return The value interpreted as TRUE/FALSE.
 */
extern gboolean
nnsconf_get_custom_value_bool (const gchar * group, const gchar * key, gboolean def);

/**
 * @brief Get neural network framework name from given model file. This does not guarantee the framework is available on the target device.
 * @param[in] model_files the prediction model paths
 * @param[in] num_models the number of model files
 * @param[in] load_conf flag to load configuration for the priority of framework
 * @return Possible framework name (NULL if it fails to detect automatically). Caller should free returned value using g_free().
 */
extern gchar *
gst_tensor_filter_detect_framework (const gchar * const *model_files, const guint num_models, const gboolean load_conf);

/**
 * @brief Check if the given hw is supported by the framework.
 * @param[in] name The name of filter sub-plugin.
 * @param[in] hw Backend accelerator hardware.
 * @return TRUE if given hw is available.
 */
extern gboolean
gst_tensor_filter_check_hw_availability (const gchar * name, const accl_hw hw);

/**
 * @brief Get pad caps from tensors config and caps of the peer connected to the pad.
 * @param pad GstPad to get possible caps
 * @param config tensors config structure
 * @return caps for given config. Caller is responsible for unreffing the returned caps.
 */
extern GstCaps *
gst_tensor_pad_caps_from_config (GstPad * pad, const GstTensorsConfig * config);

/**
 * @brief Get all possible caps from tensors config. Unlike gst_tensor_pad_caps_from_config(), this function does not check peer caps.
 * @param pad GstPad to get possible caps
 * @param config tensors config structure
 * @return caps for given config. Caller is responsible for unreffing the returned caps.
 */
extern GstCaps *
gst_tensor_pad_possible_caps_from_config (GstPad * pad, const GstTensorsConfig * config);

/**
 * @brief Check current pad caps is flexible tensor.
 * @param pad GstPad to check current caps
 * @return TRUE if pad has flexible tensor caps.
 */
extern gboolean
gst_tensor_pad_caps_is_flexible (GstPad * pad);

G_END_DECLS
#endif /* __NNSTREAMER_INTERNAL_H__ */

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
#include "nnstreamer_internal_single.h"

G_BEGIN_DECLS

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

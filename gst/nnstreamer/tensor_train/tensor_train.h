/* GStreamer
 *
 * Copyright (C) 2018 Samsung Electronics Co., Ltd.
 * 
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

/**
 * @file	gsttensor_train.h
 * @date	11 October 2022
 * @brief	Function for train tensor data using NN Frameworks
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	nnfw <nnfw@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_TRAIN_H__
#define __GST_TENSOR_TRAIN_H__

#include <gst/gst.h>
#include "gsttensor_trainsink.h"
#include <nnstreamer_subplugin.h>

#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_plugin_api_filter.h>

void gst_tensor_trainsink_find_framework (GstTensorTrainSink * sink,
    const char *name);
void gst_tensor_trainsink_create_framework (GstTensorTrainSink * sink);
gsize gst_tensor_trainsink_get_tensor_size (GstTensorTrainSink * sink,
    guint index, gboolean is_input);
#endif /* __GST_TENSOR_TRAIN_H__ */

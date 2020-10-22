/**
 * nnstreamer registerer for tizen sensor plugin
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
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
 */

/**
 * @file	tizensensor.c
 * @date	22 Oct 2020
 * @brief	Registers nnstreamer extension plugin for tizen sensor
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Dongju Chae <dongju.chae@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gst.h>

#include <tensor_source/tensor_src_tizensensor.h>

#define NNSTREAMER_TIZEN_SENSOR_INIT(plugin,name,type) \
  do { \
    if (!gst_element_register (plugin, "tensor_" # name, GST_RANK_NONE, GST_TYPE_TENSOR_ ## type)) { \
      GST_ERROR ("Failed to register nnstreamer plugin : tensor_" # name); \
      return FALSE; \
    } \
  } while (0)

/**
 * @brief Function to initialize all nnstreamer elements
 */
static gboolean
gst_nnstreamer_tizen_sensor_init (GstPlugin * plugin)
{
  NNSTREAMER_TIZEN_SENSOR_INIT (plugin, src_tizensensor, SRC_TIZENSENSOR);
  return TRUE;
}

#ifndef PACKAGE
#define PACKAGE "nnstreamer_tizen_sensor"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nnstreamer_tizen_sensor,
    "nnstreamer Tizen sensor framework extension",
    gst_nnstreamer_tizen_sensor_init, VERSION, "LGPL", "nnstreamer",
    "https://github.com/nnstreamer/nnstreamer");

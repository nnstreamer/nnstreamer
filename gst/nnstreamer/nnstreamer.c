/**
 * nnstreamer registerer
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 */

/**
 * @file	nnstreamer.c
 * @date	11 Oct 2018
 * @brief	Registers all nnstreamer plugins for gstreamer so that we can have a single big binary
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <gst/gst.h>
#include <gst/gstplugin.h>

#define NNSTREAMER_PLUGIN(name) \
  extern gboolean G_PASTE(nnstreamer_export_, name) (GstPlugin *plugin)

NNSTREAMER_PLUGIN (tensor_converter);
NNSTREAMER_PLUGIN (tensor_aggregator);
NNSTREAMER_PLUGIN (tensor_decoder);
NNSTREAMER_PLUGIN (tensor_demux);

#define NNSTREAMER_INIT(name, plugin) \
  do { \
    if (!G_PASTE(nnstreamer_export_, name)(plugin)) \
      return FALSE; \
  } while (0);

/**
 * @brief Function to initialize all nnstreamer elements
 */
static gboolean
gst_nnstreamer_init (GstPlugin * plugin)
{
  NNSTREAMER_INIT (tensor_converter, plugin);
  NNSTREAMER_INIT (tensor_aggregator, plugin);
  NNSTREAMER_INIT (tensor_decoder, plugin);
  NNSTREAMER_INIT (tensor_demux, plugin);

  return TRUE;
}

#ifndef SINGLE_BINARY
#error SINGLE_BINARY must be defined
#endif

#ifndef PACKAGE
#define PACKAGE "nnstreamer"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nnstreamer,
    "nnstreamer plugin library",
    gst_nnstreamer_init, VERSION, "LGPL", "nnstreamer",
    "https://github.com/nnsuite/nnstreamer");

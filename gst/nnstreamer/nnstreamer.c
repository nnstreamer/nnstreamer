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
 *  @mainpage nnstreamer
 *  @section  intro         Introduction
 *  - Introduction      :   Neural Network Streamer for AI Projects
 *  @section   Program      Program Name
 *  - Program Name      :   nnstreamer
 *  - Program Details   :   It provides a neural network framework connectivities (e.g., tensorflow, caffe) for gstreamer streams.
 *    Efficient Streaming for AI Projects: Neural network models wanted to use efficient and flexible streaming management as well.
 *    Intelligent Media Filters!: Use a neural network model as a media filter / converter.
 *    Composite Models!: Allow to use multiple neural network models in a single stream instance.
 *    Multi Model Intelligence!: Allow to use multiple sources for neural network models.
 *  @section  INOUTPUT      Input/output data
 *  - INPUT             :   None
 *  - OUTPUT            :   None
 *  @section  CREATEINFO    Code information
 *  - Initial date      :   2018/06/14
 *  - Version           :   0.1
 */

/**
 * @file	nnstreamer.c
 * @date	11 Oct 2018
 * @brief	Registers all nnstreamer plugins for gstreamer so that we can have a single big binary
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gst.h>
#include <gst/gstplugin.h>

#define NNSTREAMER_PLUGIN(name) \
  extern gboolean G_PASTE(nnstreamer_export_, name) (GstPlugin *plugin)

NNSTREAMER_PLUGIN (tensor_converter);
NNSTREAMER_PLUGIN (tensor_aggregator);
NNSTREAMER_PLUGIN (tensor_decoder);
NNSTREAMER_PLUGIN (tensor_demux);
NNSTREAMER_PLUGIN (tensor_merge);
NNSTREAMER_PLUGIN (tensor_mux);
NNSTREAMER_PLUGIN (tensor_sink);
NNSTREAMER_PLUGIN (tensor_src_iio);
NNSTREAMER_PLUGIN (tensor_split);
NNSTREAMER_PLUGIN (tensor_transform);
NNSTREAMER_PLUGIN (tensor_filter);
NNSTREAMER_PLUGIN (tensor_reposink);
NNSTREAMER_PLUGIN (tensor_reposrc);

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
  NNSTREAMER_INIT (tensor_merge, plugin);
  NNSTREAMER_INIT (tensor_mux, plugin);
  NNSTREAMER_INIT (tensor_sink, plugin);
  NNSTREAMER_INIT (tensor_src_iio, plugin);
  NNSTREAMER_INIT (tensor_split, plugin);
  NNSTREAMER_INIT (tensor_transform, plugin);
  NNSTREAMER_INIT (tensor_filter, plugin);
  NNSTREAMER_INIT (tensor_reposink, plugin);
  NNSTREAMER_INIT (tensor_reposrc, plugin);

  return TRUE;
}

#ifndef PACKAGE
#define PACKAGE "nnstreamer"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nnstreamer,
    "nnstreamer plugin library",
    gst_nnstreamer_init, VERSION, "LGPL", "nnstreamer",
    "https://github.com/nnsuite/nnstreamer");

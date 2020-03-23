/**
 * nnstreamer registerer
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

#include "tensor_aggregator/tensor_aggregator.h"
#include "tensor_converter/tensor_converter.h"
#include "tensor_decoder/tensordec.h"
#include "tensor_demux/gsttensordemux.h"
#include "tensor_filter/tensor_filter.h"
#include "tensor_merge/gsttensormerge.h"
#include "tensor_mux/gsttensormux.h"
#include "tensor_repo/tensor_reposink.h"
#include "tensor_repo/tensor_reposrc.h"
#include "tensor_sink/tensor_sink.h"
#if defined(__gnu_linux__) && !defined(__ANDROID__)
#include "tensor_source/tensor_src_iio.h"
#endif /* __gnu_linux__ && !__ANDROID__ */
#include "tensor_split/gsttensorsplit.h"
#include "tensor_transform/tensor_transform.h"

#define NNSTREAMER_INIT(plugin,name,type) \
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
gst_nnstreamer_init (GstPlugin * plugin)
{
  NNSTREAMER_INIT (plugin, aggregator, AGGREGATOR);
  NNSTREAMER_INIT (plugin, converter, CONVERTER);
  NNSTREAMER_INIT (plugin, decoder, DECODER);
  NNSTREAMER_INIT (plugin, demux, DEMUX);
  NNSTREAMER_INIT (plugin, filter, FILTER);
  NNSTREAMER_INIT (plugin, merge, MERGE);
  NNSTREAMER_INIT (plugin, mux, MUX);
  NNSTREAMER_INIT (plugin, reposink, REPOSINK);
  NNSTREAMER_INIT (plugin, reposrc, REPOSRC);
  NNSTREAMER_INIT (plugin, sink, SINK);
  NNSTREAMER_INIT (plugin, split, SPLIT);
  NNSTREAMER_INIT (plugin, transform, TRANSFORM);
#if defined(__gnu_linux__) && !defined(__ANDROID__)
  NNSTREAMER_INIT (plugin, src_iio, SRC_IIO);
#endif /* __gnu_linux__ && !__ANDROID__ */
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

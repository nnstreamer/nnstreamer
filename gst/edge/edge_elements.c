/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file    edge_sink.h
 * @date    02 Aug 2022
 * @brief   Register edge plugins
 * @author  Yechan Choi <yechan9.choi@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#include <gst/gst.h>

#include "edge_common.h"
#include "edge_sink.h"
#include "edge_src.h"

/**
 * @brief The entry point of the Gstreamer Edge plugin
 */
static gboolean
plugin_init (GstPlugin * plugin)
{
  if (!gst_element_register (plugin, GST_EDGE_ELEM_NAME_SINK, GST_RANK_NONE,
          GST_TYPE_EDGESINK)) {
    return FALSE;
  }

  if (!gst_element_register (plugin, GST_EDGE_ELEM_NAME_SRC, GST_RANK_NONE,
          GST_TYPE_EDGESRC)) {
    return FALSE;
  }

  return TRUE;
}

#ifndef PACKAGE
#define PACKAGE GST_EDGE_PACKAGE
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR, edge,
    "A collcetion of GStreamer plugins to support Edge",
    plugin_init, VERSION, "LGPL", PACKAGE,
    "https://github.com/nnstreamer/nnstreamer")

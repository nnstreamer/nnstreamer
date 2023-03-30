/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file	datarepo_elements.c
 * @date	31 January 2023
 * @brief	Register datarepo plugins
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Hyunil Park <hyunil46.park@samsung.com>
 * @bug		No known bugs except for NYI items
 */
#include <gst/gst.h>
#include "gstdatareposrc.h"
#include "gstdatareposink.h"

/**
 * @brief The entry point of the Gstreamer datarepo plugin
 */
static gboolean
plugin_init (GstPlugin * plugin)
{
  if (!gst_element_register (plugin, "datareposrc", GST_RANK_NONE,
          GST_TYPE_DATA_REPO_SRC))
    return FALSE;

  if (!gst_element_register (plugin, "datareposink", GST_RANK_NONE,
          GST_TYPE_DATA_REPO_SINK))
    return FALSE;

  return TRUE;
}

#ifndef PACKAGE
#define PACKAGE "NNStreamer MLOps Data Repository Plugins"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    datarepo,
    "NNStreamer MLOps Data Repository plugin library",
    plugin_init, VERSION, "LGPL", PACKAGE,
    "https://github.com/nnstreamer/nnstreamer")

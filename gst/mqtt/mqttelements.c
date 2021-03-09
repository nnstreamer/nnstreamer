/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Wook Song <wook16.song@samsung.com>
 */
/**
 * @file    mqttsink.c
 * @date    09 Mar 2021
 * @brief   Register sub-plugins included in libgstmqtt
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Wook Song <wook16.song@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include <gst/gst.h>

#include "mqttcommon.h"
#include "mqttsink.h"
#include "mqttsrc.h"

/**
 * @brief The entry point of the GStreamer MQTT plugin
 */
static gboolean
plugin_init (GstPlugin * plugin)
{
  if (!gst_element_register (plugin, GST_MQTT_ELEM_NAME_SINK, GST_RANK_NONE,
          GST_TYPE_MQTT_SINK)) {
    return FALSE;
  }

  if (!gst_element_register (plugin, GST_MQTT_ELEM_NAME_SRC, GST_RANK_NONE,
          GST_TYPE_MQTT_SRC)) {
    return FALSE;
  }

  return TRUE;
}

#ifndef PACKAGE
#define PACKAGE GST_MQTT_PACKAGE
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR, mqtt,
    "A collection of GStreamer plugins to support MQTT",
    plugin_init, VERSION, "LGPL", PACKAGE,
    "https://github.com/nnstreamer/nnstreamer")

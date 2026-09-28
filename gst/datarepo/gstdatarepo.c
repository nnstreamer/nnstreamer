/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file	gstdatarepo.c
 * @date	31 January 2023
 * @brief	Register datarepo plugins
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Hyunil Park <hyunil46.park@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include "gstdatarepo.h"
#include "gstdatareposrc.h"
#include "gstdatareposink.h"
#include <string.h>

/**
 * @brief Get data type from caps.
 */
GstDataRepoDataType
gst_data_repo_get_data_type_from_caps (const GstCaps * caps)
{
  const gchar *name;

  g_return_val_if_fail (GST_IS_CAPS (caps), GST_DATA_REPO_DATA_UNKNOWN);

  name = gst_structure_get_name (gst_caps_get_structure (caps, 0));
  g_return_val_if_fail (name != NULL, GST_DATA_REPO_DATA_UNKNOWN);

  if (g_ascii_strcasecmp (name, "other/tensors") == 0) {
    return GST_DATA_REPO_DATA_TENSOR;
  } else if (g_ascii_strcasecmp (name, "video/x-raw") == 0) {
    return GST_DATA_REPO_DATA_VIDEO;
  } else if (g_ascii_strcasecmp (name, "audio/x-raw") == 0) {
    return GST_DATA_REPO_DATA_AUDIO;
  } else if (g_ascii_strcasecmp (name, "text/x-raw") == 0) {
    return GST_DATA_REPO_DATA_TEXT;
  } else if (g_ascii_strcasecmp (name, "application/octet-stream") == 0) {
    return GST_DATA_REPO_DATA_OCTET;
  } else if (g_ascii_strcasecmp (name, "image/png") == 0
      || g_ascii_strcasecmp (name, "image/jpeg") == 0
      || g_ascii_strcasecmp (name, "image/tiff") == 0
      || g_ascii_strcasecmp (name, "image/gif") == 0) {
    return GST_DATA_REPO_DATA_IMAGE;
  }

  GST_ERROR ("Could not get a data type from caps.");
  return GST_DATA_REPO_DATA_UNKNOWN;
}

/**
 * @brief Check the location of image files before it is used as a printf format.
 * @param location The file name pattern, formatted with one unsigned int index.
 * @return TRUE if its d, i, o, u, x or X conversions besides "%%" read only
 *         the index, either as one conversion or as "%1$" conversions, with
 *         a width and precision of at most three digits.
 */
gboolean
gst_data_repo_is_valid_image_location (const gchar * location)
{
  const gchar *p = location;
  guint plain = 0, positional = 0;
  gsize digits;

  g_return_val_if_fail (location != NULL, FALSE);

  while ((p = strchr (p, '%')) != NULL) {
    p++;
    if (*p == '%') {
      p++;
      continue;
    }

    if (g_str_has_prefix (p, "1$")) {
      p += 2;
      positional++;
    } else {
      plain++;
    }
    p += strspn (p, "-+ #0'");
    if ((digits = strspn (p, "0123456789")) > 3)
      return FALSE;
    p += digits;
    if (*p == '.') {
      p++;
      p += strspn (p, "0");
      if ((digits = strspn (p, "0123456789")) > 3)
        return FALSE;
      p += digits;
    }

    /* Wider modifiers are kept for working locations such as "%04ld". */
    if (g_str_has_prefix (p, "hh") || g_str_has_prefix (p, "ll"))
      p += 2;
    else if (*p != '\0' && strchr ("hljzt", *p) != NULL)
      p++;

    if (*p == '\0' || strchr ("diouxX", *p) == NULL)
      return FALSE;
    if (plain > 1 || (plain > 0 && positional > 0))
      return FALSE;
  }

  return TRUE;
}

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

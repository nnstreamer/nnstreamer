/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Pipeline from/to PBTxt Converter Parser
 * Copyright (C) 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * @file  types.c
 * @date  12 Nov 2020
 * @brief Simplified Gstreamer's internal functions for gst2pbtxt parser (nnstreamer parser)
 * @see https://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <glib.h>
#include "types.h"

/**
 * @brief Get the error quark used by the parsing subsystem.
 *
 * Returns: the quark of the parse errors.
 */
GQuark
gst2pbtxt_parse_error_quark (void)
{
  static GQuark quark = 0;

  if (!quark)
    quark = g_quark_from_static_string ("gst_parse_error");
  return quark;
}

/**
 * @brief Replacement of gst_parse_element_make
 */
_Element *
nnstparser_element_make (const gchar * element, const gchar * name)
{
  _Element ret = g_malloc (sizeof (_Element));
  ret->element = g_strdup (element);
  ret->name = g_strdup (name);

  return ret;
}

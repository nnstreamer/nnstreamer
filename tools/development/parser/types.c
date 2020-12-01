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
  _Element ret = g_new0 (_Element, 1);
  ret->specialType = eST_normal;
  ret->element = g_strdup (element);
  ret->name = g_strdup (name);
  ret->refcount = 1;
  return ret;
}

/**
 * @brief Unref an element
 */
_Element *
nnstparser_element_unref (_Element * element)
{
  g_assert (element);

  if (element->refcount <= 0) {
    g_printerr ("ERROR! Refcounter is broken: %s.", __func__);
  }
  g_assert (element->refcount > 0);

  element->refcount--;
  if (element->refcount <= 0) {
    g_free (element);
    return NULL;
  }

  return element;
}

/**
 * @brief Ref an element
 */
void
nnstparser_element_ref (_Element * element)
{
  g_assert (element);

  if (element->refcount <= 0) {
    g_printerr ("ERROR! Refcounter is broken: %s.", __func__);
  }
  g_assert (element->refcount > 0);

  element->refcount++;
}

/**
 * @brief Create URL dummy element instead of gst_element_make_from_uri
 */
_Element *
nnstparser_element_from_uri (_URIType type, const gchar * uri,
    const gchar * elementname, void **error)
{
  _Element ret = g_malloc (sizeof (_Element));

  g_assert (type == GST_URI_SINK || type == GST_URI_SRC);

  ret->specialType = (type == GST_URI_SINK) ? eST_URI_SINK : eST_URI_SRC;
  ret->element = g_strdup (url);
  ret->name = g_strdup (elementname);
  ret->refcount = 1;
  return ret;
}

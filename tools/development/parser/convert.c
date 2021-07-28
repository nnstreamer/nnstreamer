/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Pipeline from/to PBTxt Converter Parser
 * Copyright (C) 2021 MyungJoo Ham <myungjoo.ham@samsung.com>
 * Copyright (C) 2021 Dongju Chae <dongju.chae@samsung.com>
 */
/**
 * @file    convert.c
 * @date    13 May 2021
 * @brief   GStreamer pipeline from/to pbtxt converter
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "convert.h"
#include <glib/gprintf.h>

static GData *datalist;

#define GET_NODE_INDEX(elem)\
  g_slist_index (g_datalist_get_data (&datalist, elem->element), elem)

/**
 * @brief Get a node name (ignore index 0)
 */
static gchar *
pbtxt_get_node_name (_Element * elem)
{
  gint index = GET_NODE_INDEX (elem);

  /* Internal error: the parsing wasn't successful due to unknown errors */
  g_assert (index >= 0);

  if (index == 0)
    return g_strdup_printf ("%s", elem->element);
  else
    return g_strdup_printf ("%s_%d", elem->element, index + 1);
}

/**
 * @brief Print input stream of node
 */
static void
pbtxt_print_node_input_stream (gpointer data, gpointer user_data)
{
  _Pad *pad = (_Pad *) data;
  gchar *pad_name;
  (void) user_data;

  g_return_if_fail (pad != NULL);

  if (g_slist_length (pad->peer->parent->sink_pads) == 0) {
    /* assume that any src has only one pad */
    pad_name = pbtxt_get_node_name (pad->peer->parent);
  } else {
    pad_name = g_strdup_printf ("%s_%d_%d",
        pad->peer->parent->element,
        GET_NODE_INDEX (pad->peer->parent),
        g_slist_index (pad->peer->parent->src_pads, pad->peer));
  }

  g_printf ("\tinput_stream: \"%s\"\n", pad_name);
  g_free (pad_name);
}

/**
 * @brief Print output stream of node
 */
static void
pbtxt_print_node_output_stream (gpointer data, gpointer user_data)
{
  _Pad *pad = (_Pad *) data;
  gchar *pad_name;
  (void) user_data;

  g_return_if_fail (pad != NULL);

  if (g_slist_length (pad->peer->parent->src_pads) == 0) {
    /* assume that any sink has only one pad */
    pad_name = pbtxt_get_node_name (pad->peer->parent);
  } else {
    pad_name = g_strdup_printf ("%s_%d_%d",
        pad->parent->element,
        GET_NODE_INDEX (pad->parent),
        g_slist_index (pad->parent->src_pads, pad));
  }

  g_printf ("\toutput_stream: \"%s\"\n", pad_name);
  g_free (pad_name);
}

/** @brief Print pbtxt nodes */
static void
pbtxt_print_node (gpointer data, gpointer user_data)
{
  _Element *elem = (_Element *) data;
  (void) user_data;

  g_return_if_fail (elem != NULL);

  if (g_slist_length (elem->src_pads) == 0 ||
      g_slist_length (elem->sink_pads) == 0)
    return;

  g_printf ("\nnode: {\n\tcalculator: \"%sCalculator\"\n", elem->element);
  g_slist_foreach (elem->sink_pads, pbtxt_print_node_input_stream, NULL);
  g_slist_foreach (elem->src_pads, pbtxt_print_node_output_stream, NULL);
  g_printf ("}\n");

  /* TODO: Filling 'node_options' for detail element info. */
}

/**
 * @brief Prepare conversion checking nodes
 */
static void
pbtxt_prepare (gpointer data, gpointer user_data)
{
  _Element *elem = (_Element *) data;
  GSList *list;
  (void) user_data;

  g_return_if_fail (elem != NULL);

  list = g_datalist_get_data (&datalist, elem->element);
  list = g_slist_append (list, elem);
  g_datalist_set_data (&datalist, elem->element, list);

  if (g_slist_length (elem->sink_pads) == 0) {
    gchar *name = pbtxt_get_node_name (elem);
    g_printf ("input_stream: \"%s\"\n", name);
    g_free (name);
  }

  if (g_slist_length (elem->src_pads) == 0) {
    gchar *name = pbtxt_get_node_name (elem);
    g_printf ("output_stream: \"%s\"\n", name);
    g_free (name);
  }
}

/** @brief Convert gst pipeline to pbtxt */
void
convert_to_pbtxt (_Element * pipeline)
{
  g_return_if_fail (pipeline != NULL);

  g_datalist_init (&datalist);

  g_slist_foreach (pipeline->elements, pbtxt_prepare, NULL);
  g_slist_foreach (pipeline->elements, pbtxt_print_node, NULL);

  g_datalist_clear (&datalist);
}

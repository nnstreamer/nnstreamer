/**
 * GStreamer / NNStreamer tensor_decoder plugin support
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
 *
 */
/**
 * @file	tensordec-plugins.c
 * @date	05 Nov 2018
 * @brief	Tensor-decoder plugin support logic
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "tensordec.h"
#include <nnstreamer_subplugin.h>
#include <gst/gstinfo.h>

typedef struct _TensorDecDefList TensorDecDefList;

/**
 * @brief Linked list having all registered decoder subplugins
 */
struct _TensorDecDefList
{
  TensorDecDefList *next; /**< "Next" in the list */
  TensorDecDef *body; /**< "Data" in this list element */
};
static TensorDecDef unknown = {
  .modename = "unknown",
  .type = OUTPUT_UNKNOWN,
};
static TensorDecDefList listhead = {.next = NULL,.body = &unknown };

/**
 * @brief decoder's subplugins should call this function to register
 * @param[in] decoder The decoder subplugin instance
 */
gboolean
tensordec_probe (TensorDecDef * decoder)
{
  TensorDecDefList *list;

  if (!decoder || !decoder->modename || !decoder->modename[0]) {
    GST_ERROR ("Cannot register invalid decoder.\n");
    return FALSE;
  }

  /**
   * Check if there is no duplicated entry (modename).
   * This is Linked List Traversal.
   * If we hit the "tail", add decoder
   */
  list = &listhead;
  do {
    if (0 == g_strcmp0 (list->body->modename, decoder->modename)) {
      /* Duplicated! */
      GST_ERROR ("Duplicated decoder name found: %s\n", decoder->modename);
      return FALSE;
    }
    if (list->next == NULL) {
      TensorDecDefList *next = g_malloc (sizeof (TensorDecDefList));
      next->next = NULL;
      next->body = decoder;
      list->next = next;
      break;
    }
    list = list->next;
  } while (list != NULL);

  GST_INFO ("A new subplugin, \"%s\" is registered for tensor_decoder.\n",
      decoder->modename);

  return TRUE;
}

/**
 * @brief decoder's subplugin may call this to unregister
 * @param[in] name the name of decoder (modename)
 */
void
tensordec_exit (const gchar * name)
{
  TensorDecDefList *list = &listhead;

  if (!name || !name[0]) {
    GST_ERROR ("Cannot unregister without proper name.\n");
    return;
  }

  /**
   * Check if there is no duplicated entry (modename).
   * This is Linked List Traversal.
   * If we hit the "tail", add decoder
   */
  list = &listhead;
  do {
    if (list->next != NULL && 0 == g_strcmp0 (list->next->body->modename, name)) {
      TensorDecDefList *found = list->next;
      list->next = found->next;
      g_free (found);
      GST_INFO ("A subplugin, \"%s\" is removed from tensor_decoder.\n", name);
      return;
    }
    list = list->next;
  } while (list != NULL);

  GST_ERROR ("A subplugin, \"%s\" was not found.\n", name);
  return;
}

/**
 * @brief Find decoders subplugin with the name
 * @param[in] name the name of decoder (modename)
 */
const TensorDecDef *
tensordec_find (const gchar * name)
{
  TensorDecDefList *list = &listhead;

  if (!name || !name[0]) {
    GST_ERROR ("Cannot find without proper name.\n");
    return NULL;
  }

  do {
    g_assert (list->body);

    if (0 == g_strcmp0 (list->body->modename, name)) {
      return list->body;
    }
    list = list->next;
  } while (list != NULL);

  /* If not found, try to search with nnstreamer_subplugin APIs */
  return get_subplugin (NNS_SUBPLUGIN_DECODER, name);
}

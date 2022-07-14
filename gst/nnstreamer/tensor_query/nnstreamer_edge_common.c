/* SPDX-License-Identifier: Apache-2.0 */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nnstreamer-edge-common.c
 * @date   6 April 2022
 * @brief  Common util functions for nnstreamer edge.
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#define _GNU_SOURCE
#include <stdio.h>

#include "nnstreamer-edge-common.h"

/**
 * @brief Free allocated memory.
 */
void
nns_edge_free (void *data)
{
  if (data)
    free (data);
}

/**
 * @brief Allocate new memory and copy bytes.
 * @note Caller should release newly allocated memory using nns_edge_free().
 */
void *
nns_edge_memdup (const void *data, size_t size)
{
  void *mem = NULL;

  if (data && size > 0) {
    mem = malloc (size);

    if (mem) {
      memcpy (mem, data, size);
    } else {
      nns_edge_loge ("Failed to allocate memory (%zd).", size);
    }
  }

  return mem;
}

/**
 * @brief Allocate new memory and copy string.
 * @note Caller should release newly allocated string using nns_edge_free().
 */
char *
nns_edge_strdup (const char *str)
{
  char *new_str = NULL;
  size_t len;

  if (str) {
    len = strlen (str);

    new_str = (char *) malloc (len + 1);
    if (new_str) {
      memcpy (new_str, str, len);
      new_str[len] = '\0';
    } else {
      nns_edge_loge ("Failed to allocate memory (%zd).", len + 1);
    }
  }

  return new_str;
}

/**
 * @brief Allocate new memory and print formatted string.
 * @note Caller should release newly allocated string using nns_edge_free().
 */
char *
nns_edge_strdup_printf (const char *format, ...)
{
  char *new_str = NULL;
  va_list args;
  int len;

  va_start (args, format);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
  len = vasprintf (&new_str, format, args);
#pragma GCC diagnostic pop
  if (len < 0)
    new_str = NULL;
  va_end (args);

  return new_str;
}

/**
 * @brief Create nnstreamer edge event.
 */
int
nns_edge_event_create (nns_edge_event_e event, nns_edge_event_h * event_h)
{
  nns_edge_event_s *ee;

  if (!event_h) {
    nns_edge_loge ("Invalid param, event_h should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (event <= NNS_EDGE_EVENT_UNKNOWN) {
    nns_edge_loge ("Invalid param, given event type is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  ee = (nns_edge_event_s *) malloc (sizeof (nns_edge_event_s));
  if (!ee) {
    nns_edge_loge ("Failed to allocate memory for edge event.");
    return NNS_EDGE_ERROR_OUT_OF_MEMORY;
  }

  memset (ee, 0, sizeof (nns_edge_event_s));
  ee->magic = NNS_EDGE_MAGIC;
  ee->event = event;

  *event_h = ee;
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Destroy nnstreamer edge event.
 */
int
nns_edge_event_destroy (nns_edge_event_h event_h)
{
  nns_edge_event_s *ee;

  ee = (nns_edge_event_s *) event_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (ee)) {
    nns_edge_loge ("Invalid param, given edge event is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  ee->magic = NNS_EDGE_MAGIC_DEAD;

  if (ee->data.destroy_cb)
    ee->data.destroy_cb (ee->data.data);

  SAFE_FREE (ee);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Set event data.
 */
int
nns_edge_event_set_data (nns_edge_event_h event_h, void *data, size_t data_len,
    nns_edge_data_destroy_cb destroy_cb)
{
  nns_edge_event_s *ee;

  ee = (nns_edge_event_s *) event_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (ee)) {
    nns_edge_loge ("Invalid param, given edge event is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!data || data_len <= 0) {
    nns_edge_loge ("Invalid param, data should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  /* Clear old data and set new one. */
  if (ee->data.destroy_cb)
    ee->data.destroy_cb (ee->data.data);

  ee->data.data = data;
  ee->data.data_len = data_len;
  ee->data.destroy_cb = destroy_cb;

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Get the nnstreamer edge event type.
 */
int
nns_edge_event_get_type (nns_edge_event_h event_h, nns_edge_event_e * event)
{
  nns_edge_event_s *ee;

  ee = (nns_edge_event_s *) event_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (ee)) {
    nns_edge_loge ("Invalid param, given edge event is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!event) {
    nns_edge_loge ("Invalid param, event should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  *event = ee->event;
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Parse edge event (NNS_EDGE_EVENT_NEW_DATA_RECEIVED) and get received data.
 * @note Caller should release returned edge data using nns_edge_data_destroy().
 */
int
nns_edge_event_parse_new_data (nns_edge_event_h event_h,
    nns_edge_data_h * data_h)
{
  nns_edge_event_s *ee;

  ee = (nns_edge_event_s *) event_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (ee)) {
    nns_edge_loge ("Invalid param, given edge event is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!data_h) {
    nns_edge_loge ("Invalid param, data_h should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (ee->event != NNS_EDGE_EVENT_NEW_DATA_RECEIVED) {
    nns_edge_loge ("The edge event has invalid event type.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  return nns_edge_data_copy ((nns_edge_data_h) ee->data.data, data_h);
}

/**
 * @brief Parse edge event (NNS_EDGE_EVENT_CAPABILITY) and get capability string.
 * @note Caller should release returned string using free().
 */
int
nns_edge_event_parse_capability (nns_edge_event_h event_h, char **capability)
{
  nns_edge_event_s *ee;

  ee = (nns_edge_event_s *) event_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (ee)) {
    nns_edge_loge ("Invalid param, given edge event is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!capability) {
    nns_edge_loge ("Invalid param, capability should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (ee->event != NNS_EDGE_EVENT_CAPABILITY) {
    nns_edge_loge ("The edge event has invalid event type.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  *capability = nns_edge_strdup (ee->data.data);

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Create nnstreamer edge data.
 */
int
nns_edge_data_create (nns_edge_data_h * data_h)
{
  nns_edge_data_s *ed;

  if (!data_h) {
    nns_edge_loge ("Invalid param, data_h should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  ed = (nns_edge_data_s *) malloc (sizeof (nns_edge_data_s));
  if (!ed) {
    nns_edge_loge ("Failed to allocate memory for edge data.");
    return NNS_EDGE_ERROR_OUT_OF_MEMORY;
  }

  memset (ed, 0, sizeof (nns_edge_data_s));
  ed->magic = NNS_EDGE_MAGIC;
  ed->info_table = g_hash_table_new_full (g_str_hash, g_str_equal,
      nns_edge_free, nns_edge_free);

  *data_h = ed;
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Destroy nnstreamer edge data.
 */
int
nns_edge_data_destroy (nns_edge_data_h data_h)
{
  nns_edge_data_s *ed;
  unsigned int i;

  ed = (nns_edge_data_s *) data_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (ed)) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  ed->magic = NNS_EDGE_MAGIC_DEAD;

  for (i = 0; i < ed->num; i++) {
    if (ed->data[i].destroy_cb)
      ed->data[i].destroy_cb (ed->data[i].data);
  }

  g_hash_table_destroy (ed->info_table);

  SAFE_FREE (ed);
  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Validate edge data handle.
 */
int
nns_edge_data_is_valid (nns_edge_data_h data_h)
{
  nns_edge_data_s *ed;

  ed = (nns_edge_data_s *) data_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (ed)) {
    nns_edge_loge ("Invalid param, edge data handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Copy edge data and return new handle.
 */
int
nns_edge_data_copy (nns_edge_data_h data_h, nns_edge_data_h * new_data_h)
{
  nns_edge_data_s *ed;
  nns_edge_data_s *copied;
  GHashTableIter iter;
  gpointer key, value;
  unsigned int i;
  int ret;

  ed = (nns_edge_data_s *) data_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (ed)) {
    nns_edge_loge ("Invalid param, edge data handle is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!new_data_h) {
    nns_edge_loge ("Invalid param, new_data_h should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  ret = nns_edge_data_create (new_data_h);
  if (ret != NNS_EDGE_ERROR_NONE) {
    nns_edge_loge ("Failed to create new data handle.");
    return ret;
  }

  copied = (nns_edge_data_s *) (*new_data_h);

  copied->num = ed->num;
  for (i = 0; i < ed->num; i++) {
    copied->data[i].data = nns_edge_memdup (ed->data[i].data,
        ed->data[i].data_len);
    copied->data[i].data_len = ed->data[i].data_len;
    copied->data[i].destroy_cb = nns_edge_free;
  }

  g_hash_table_iter_init (&iter, ed->info_table);
  while (g_hash_table_iter_next (&iter, &key, &value)) {
    g_hash_table_insert (copied->info_table, nns_edge_strdup (key),
        nns_edge_strdup (value));
  }

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Add raw data into nnstreamer edge data.
 */
int
nns_edge_data_add (nns_edge_data_h data_h, void *data, size_t data_len,
    nns_edge_data_destroy_cb destroy_cb)
{
  nns_edge_data_s *ed;

  ed = (nns_edge_data_s *) data_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (ed)) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (ed->num >= NNS_EDGE_DATA_LIMIT) {
    nns_edge_loge ("Cannot add data, the maximum number of edge data is %d.",
        NNS_EDGE_DATA_LIMIT);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!data || data_len <= 0) {
    nns_edge_loge ("Invalid param, data should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  ed->data[ed->num].data = data;
  ed->data[ed->num].data_len = data_len;
  ed->data[ed->num].destroy_cb = destroy_cb;
  ed->num++;

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Get the nnstreamer edge data.
 * @note DO NOT release returned data. You should copy the data to another buffer if the returned data is necessary.
 */
int
nns_edge_data_get (nns_edge_data_h data_h, unsigned int index, void **data,
    size_t *data_len)
{
  nns_edge_data_s *ed;

  ed = (nns_edge_data_s *) data_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (ed)) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!data || !data_len) {
    nns_edge_loge ("Invalid param, data and len should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (index >= ed->num) {
    nns_edge_loge
        ("Invalid param, the number of edge data is %u but requested %uth data.",
        ed->num, index);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  *data = ed->data[index].data;
  *data_len = ed->data[index].data_len;

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Get the number of nnstreamer edge data.
 */
int
nns_edge_data_get_count (nns_edge_data_h data_h, unsigned int *count)
{
  nns_edge_data_s *ed;

  ed = (nns_edge_data_s *) data_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (ed)) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!count) {
    nns_edge_loge ("Invalid param, count should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  *count = ed->num;

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Set the information of edge data.
 */
int
nns_edge_data_set_info (nns_edge_data_h data_h, const char *key,
    const char *value)
{
  nns_edge_data_s *ed;

  ed = (nns_edge_data_s *) data_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (ed)) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!STR_IS_VALID (key)) {
    nns_edge_loge ("Invalid param, given key is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!STR_IS_VALID (value)) {
    nns_edge_loge ("Invalid param, given value is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  g_hash_table_insert (ed->info_table, nns_edge_strdup (key),
      nns_edge_strdup (value));

  return NNS_EDGE_ERROR_NONE;
}

/**
 * @brief Get the information of edge data. Caller should release the returned value using free().
 */
int
nns_edge_data_get_info (nns_edge_data_h data_h, const char *key, char **value)
{
  nns_edge_data_s *ed;
  char *val;

  ed = (nns_edge_data_s *) data_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (ed)) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!STR_IS_VALID (key)) {
    nns_edge_loge ("Invalid param, given key is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (!value) {
    nns_edge_loge ("Invalid param, value should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  val = g_hash_table_lookup (ed->info_table, key);
  if (!val) {
    nns_edge_loge ("Invalid param, cannot find info about '%s'.", key);
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  *value = nns_edge_strdup (val);

  return NNS_EDGE_ERROR_NONE;
}

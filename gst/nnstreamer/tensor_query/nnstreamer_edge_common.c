/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nnstreamer_edge_internal.c
 * @date   6 April 2022
 * @brief  Common util functions for nnstreamer edge.
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "nnstreamer_edge_common.h"

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

  g_free (ee);
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
 * @brief Validate data handle.
 */
bool
nns_edge_data_is_valid (nns_edge_data_h data_h)
{
  nns_edge_data_s *ed;

  ed = (nns_edge_data_s *) data_h;

  if (!NNS_EDGE_MAGIC_IS_VALID (ed))
    return false;

  return true;
}

/**
 * @brief Create nnstreamer edge data.
 */
int
nns_edge_data_create (nns_edge_data_type_e dtype, nns_edge_data_h * data_h)
{
  nns_edge_data_s *ed;

  if (!data_h) {
    nns_edge_loge ("Invalid param, data_h should not be null.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  if (dtype < 0 || dtype >= NNS_EDGE_DATA_TYPE_MAX) {
    nns_edge_loge ("Invalid param, given data type is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  ed = (nns_edge_data_s *) malloc (sizeof (nns_edge_data_s));
  if (!ed) {
    nns_edge_loge ("Failed to allocate memory for edge data.");
    return NNS_EDGE_ERROR_OUT_OF_MEMORY;
  }

  memset (ed, 0, sizeof (nns_edge_data_s));
  ed->magic = NNS_EDGE_MAGIC;
  ed->dtype = dtype;

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

  if (!nns_edge_data_is_valid (data_h)) {
    nns_edge_loge ("Invalid param, given edge data is invalid.");
    return NNS_EDGE_ERROR_INVALID_PARAMETER;
  }

  ed->magic = NNS_EDGE_MAGIC_DEAD;

  for (i = 0; i < ed->num; i++) {
    if (ed->data[i].destroy_cb)
      ed->data[i].destroy_cb (ed->data[i].data);
  }

  g_free (ed);
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

  if (!nns_edge_data_is_valid (data_h)) {
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

/**
 * NNStreamer Common Header
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
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
 * @file	tensor_repo.c
 * @date	17 Nov 2018
 * @brief	tensor repo header file for NNStreamer, the GStreamer plugin for neural networks
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include<tensor_repo.h>

/**
 * @brief tensor repo global variable with init.
 */
GstTensorRepo _repo = {.num_data = 0,.tensorsdata = NULL,.initialized = FALSE };

/**
 * @brief getter to get nth GstTensorData
 */
GstTensorData *
gst_tensor_repo_get_tensor (guint nth)
{
  return g_slist_nth_data (_repo.tensorsdata, nth);
}

/**
 * @brief add GstTensorData into repo
 */
guint
gst_tensor_repo_add_data (GstTensorData * data)
{
  guint id = _repo.num_data;
  GST_REPO_LOCK ();
  _repo.tensorsdata = g_slist_append (_repo.tensorsdata, data);
  _repo.num_data++;
  GST_REPO_UNLOCK ();
  return id;
}

/**
 * @brief push GstBuffer into repo
 */
void
gst_tensor_repo_push_buffer (guint nth, GstBuffer * buffer)
{
  GST_TENSOR_REPO_LOCK (nth);

  GstTensorData *data = gst_tensor_repo_get_tensor (nth);
  data->buffer = buffer;
  GST_TENSOR_REPO_BROADCAST (nth);
  GST_TENSOR_REPO_UNLOCK (nth);
}

/**
 * @brief pop GstTensorData from repo
 */
GstTensorData *
gst_tensor_repopop_buffer (guint nth)
{
  GST_TENSOR_REPO_LOCK (nth);
  GstTensorData *current_data, *data;

  current_data = g_slist_nth_data (_repo.tensorsdata, nth);

  while (!current_data)
    GST_TENSOR_REPO_WAIT (nth);
  data = current_data;
  current_data = NULL;
  GST_TENSOR_REPO_UNLOCK (nth);

  return data;
}

/**
 * @brief remove nth GstTensorData from GstTensorRepo
 */
void
gst_tensor_repo_remove_data (guint nth)
{
  GST_REPO_LOCK ();
  GSList *data = g_slist_nth (_repo.tensorsdata, nth);
  g_mutex_clear (GST_TENSOR_REPO_GET_LOCK (nth));
  g_cond_clear (GST_TENSOR_REPO_GET_COND (nth));
  _repo.tensorsdata = g_slist_delete_link (_repo.tensorsdata, data);
  _repo.num_data--;
  GST_REPO_UNLOCK ();
}

/**
 * @brief GstTensorRepo initialization
 */
void
gst_tensor_repo_init ()
{
  _repo.num_data = 0;
  g_mutex_init (&_repo.repo_lock);
  _repo.tensorsdata = NULL;
  _repo.initialized = TRUE;
}

/**
 * NNStreamer Tensor Repo Header's Contents
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
 * @brief	tensor repo file for NNStreamer, the GStreamer plugin for neural networks
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include<tensor_repo.h>
#include <stdio.h>

#ifndef DBG
#define DBG TRUE
#endif

#define _print_log(...) if (DBG) g_message(__VA_ARGS__)

/**
 * @brief tensor repo global variable with init.
 */
GstTensorRepo _repo = {.num_data = 0,.initialized = FALSE };

/**
 * @brief getter to get nth GstTensorData
 */
GstTensorData *
gst_tensor_repo_get_tensor (guint nth)
{
  GstTensorData *data;
  gpointer *p = g_hash_table_lookup (_repo.hash, GINT_TO_POINTER (nth));
  data = (GstTensorData *) p;
  g_return_val_if_fail (data != NULL, NULL);
  return data;
}

/**
 * @brief add GstTensorData into repo
 */
gboolean
gst_tensor_repo_add_data (GstTensorData * data, guint myid)
{
  gboolean ret = FALSE;

  if (!_repo.initialized)
    gst_tensor_repo_init ();

  GST_REPO_LOCK ();
  ret = g_hash_table_insert (_repo.hash, GINT_TO_POINTER (myid), data);
  g_assert (ret);

  _print_log ("Successfully added in hash table with key[%d]", myid);

  _repo.num_data++;
  GST_REPO_UNLOCK ();
  return ret;
}

/**
 * @brief push GstBuffer into repo
 */
gboolean
gst_tensor_repo_push_buffer (guint nth, GstBuffer * buffer)
{
  GST_TENSOR_REPO_LOCK (nth);

  GstTensorData *data = gst_tensor_repo_get_tensor (nth);
  g_return_val_if_fail (data != NULL, FALSE);

  data->buffer = buffer;
  GST_TENSOR_REPO_BROADCAST (nth);
  GST_TENSOR_REPO_UNLOCK (nth);
  return TRUE;
}

/**
 * @brief pop GstTensorData from repo
 */
GstTensorData *
gst_tensor_repopop_buffer (guint nth)
{
  GST_TENSOR_REPO_LOCK (nth);
  GstTensorData *current_data, *data;

  current_data = gst_tensor_repo_get_tensor (nth);

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
gboolean
gst_tensor_repo_remove_data (guint nth)
{
  gboolean ret;
  GST_REPO_LOCK ();
  g_mutex_clear (GST_TENSOR_REPO_GET_LOCK (nth));
  g_cond_clear (GST_TENSOR_REPO_GET_COND (nth));

  ret = g_hash_table_remove (_repo.hash, GINT_TO_POINTER (nth));

  if (ret) {
    _repo.num_data--;
    _print_log ("key[%d] is removed\n", nth);
  }

  GST_REPO_UNLOCK ();
  return ret;
}

/**
 * @brief GstTensorRepo initialization
 */
void
gst_tensor_repo_init ()
{
  _repo.num_data = 0;
  g_mutex_init (&_repo.repo_lock);
  _repo.hash = g_hash_table_new (g_direct_hash, g_direct_equal);
  _repo.initialized = TRUE;
}

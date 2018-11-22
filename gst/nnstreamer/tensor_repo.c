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
#define DBG FALSE
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
gst_tensor_repo_add_data (guint myid)
{
  gboolean ret = FALSE;
  GstTensorData *new;

  GST_REPO_LOCK ();
  gpointer *check = g_hash_table_lookup (_repo.hash, GINT_TO_POINTER (myid));

  if (check != NULL) {
    return TRUE;
  }

  new = g_new (GstTensorData, 1);
  new->eos = FALSE;
  new->buffer = NULL;
  g_cond_init (&new->cond);
  g_mutex_init (&new->lock);

  ret = g_hash_table_insert (_repo.hash, GINT_TO_POINTER (myid), new);
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
  _print_log ("Pushed [%d] (size : %lu)\n", nth,
      gst_buffer_get_size (data->buffer));

  GST_TENSOR_REPO_SIGNAL (nth);
  GST_TENSOR_REPO_UNLOCK (nth);
  return TRUE;
}

/**
 * @brief check eos of slot
 */
gboolean
gst_tensor_repo_check_eos (guint nth)
{
  GST_TENSOR_REPO_LOCK (nth);
  GstTensorData *data = gst_tensor_repo_get_tensor (nth);
  GST_TENSOR_REPO_UNLOCK (nth);
  return data->eos;
}

/**
 * @brief set eos of slot
 */
gboolean
gst_tensor_repo_set_eos (guint nth)
{
  GST_TENSOR_REPO_LOCK (nth);
  GstTensorData *data = gst_tensor_repo_get_tensor (nth);
  data->eos = TRUE;
  GST_TENSOR_REPO_UNLOCK (nth);
  return data->eos;
}


/**
 * @brief pop GstTensorData from repo
 */
GstBuffer *
gst_tensor_repopop_buffer (guint nth)
{
  GstTensorData *current_data;
  GstBuffer *buf;

  current_data = gst_tensor_repo_get_tensor (nth);

  GST_TENSOR_REPO_LOCK (nth);
  while (!current_data->buffer)
    GST_TENSOR_REPO_WAIT (nth);
  buf = gst_buffer_copy_deep (current_data->buffer);
  _print_log ("Popped [ %d ] (size: %lu)\n", nth, gst_buffer_get_size (buf));
  current_data->buffer = NULL;
  GST_TENSOR_REPO_UNLOCK (nth);

  return buf;
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
  if (_repo.initialized)
    return;

  _repo.num_data = 0;
  g_mutex_init (&_repo.repo_lock);
  g_cond_init (&_repo.repo_cond);
  GST_REPO_LOCK ();
  _repo.hash = g_hash_table_new (g_direct_hash, g_direct_equal);
  _repo.initialized = TRUE;
  GST_REPO_BROADCAST ();
  GST_REPO_UNLOCK ();
}

/**
 * @brief wait for finish of initialization
 */
gboolean
gst_tensor_repo_wait ()
{
  GST_REPO_LOCK ();
  while (!_repo.initialized)
    GST_REPO_WAIT ();
  GST_REPO_UNLOCK ();
  return TRUE;
}

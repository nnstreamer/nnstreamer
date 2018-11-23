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

/**
 * @brief tensor repo global variable with init.
 */
GstTensorRepo _repo = {.num_data = 0,.initialized = FALSE };

/**
 * @brief getter to get nth GstTensorRepoData
 */
GstTensorRepoData *
gst_tensor_repo_get_repodata (guint nth)
{
  GstTensorRepoData *data;
  gpointer p = g_hash_table_lookup (_repo.hash, GINT_TO_POINTER (nth));
  data = (GstTensorRepoData *) p;
  return data;
}

/**
 * @brief add GstTensorRepoData into repo
 */
gboolean
gst_tensor_repo_add_repodata (guint nth)
{
  gboolean ret = FALSE;
  GstTensorRepoData *new;

  GST_REPO_LOCK ();
  gpointer check = g_hash_table_lookup (_repo.hash, GINT_TO_POINTER (nth));

  if (check != NULL) {
    return TRUE;
  }

  new = g_new (GstTensorRepoData, 1);
  new->eos = FALSE;
  new->buffer = NULL;
  g_cond_init (&new->cond);
  g_mutex_init (&new->lock);

  ret = g_hash_table_insert (_repo.hash, GINT_TO_POINTER (nth), new);
  g_assert (ret);

  if (DBG)
    GST_DEBUG ("Successfully added in hash table with key[%d]", nth);

  _repo.num_data++;
  GST_REPO_UNLOCK ();
  return ret;
}

/**
 * @brief push GstBuffer into repo
 */
gboolean
gst_tensor_repo_set_buffer (guint nth, GstBuffer * buffer)
{
  GST_TENSOR_REPO_LOCK (nth);

  GstTensorRepoData *data = gst_tensor_repo_get_repodata (nth);
  g_return_val_if_fail (data != NULL, FALSE);

  data->buffer = buffer;
  if (DBG) {
    unsigned long size = gst_buffer_get_size (data->buffer);
    GST_DEBUG ("Pushed [%d] (size : %lu)\n", nth, size);
  }

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
  GstTensorRepoData *data = gst_tensor_repo_get_repodata (nth);
  if (data->eos) {
    if (DBG)
      GST_DEBUG ("check eos done [%s]\n", data->eos ? "TRUE" : "FALSE");
  }
  return data->eos;
}

/**
 * @brief set eos of slot
 */
gboolean
gst_tensor_repo_set_eos (guint nth)
{
  GST_TENSOR_REPO_LOCK (nth);
  GstTensorRepoData *data = gst_tensor_repo_get_repodata (nth);
  data->eos = TRUE;
  GST_TENSOR_REPO_SIGNAL (nth);
  GST_TENSOR_REPO_UNLOCK (nth);
  return data->eos;
}


/**
 * @brief get GstTensorRepoData from repo
 */
GstBuffer *
gst_tensor_repo_get_buffer (guint nth)
{
  GstTensorRepoData *current_data;
  GstBuffer *buf;
  gboolean eos = FALSE;

  current_data = gst_tensor_repo_get_repodata (nth);
  GST_TENSOR_REPO_LOCK (nth);
  while (!current_data->buffer) {
    if (gst_tensor_repo_check_eos (nth)) {
      eos = TRUE;
      break;
    }
    GST_TENSOR_REPO_WAIT (nth);
  }
  if (eos) {
    if (DBG)
      GST_DEBUG ("Get EOS Signal while waiting\n");
    buf = NULL;
  } else {
    buf = gst_buffer_copy_deep (current_data->buffer);
    if (DBG) {
      unsigned long size = gst_buffer_get_size (buf);
      GST_DEBUG ("Popped [ %d ] (size: %lu)\n", nth, size);
    }
  }
  current_data->buffer = NULL;
  GST_TENSOR_REPO_UNLOCK (nth);
  return buf;
}

/**
 * @brief remove nth GstTensorRepoData from GstTensorRepo
 */
gboolean
gst_tensor_repo_remove_repodata (guint nth)
{
  gboolean ret;
  GST_REPO_LOCK ();
  g_mutex_clear (GST_TENSOR_REPO_GET_LOCK (nth));
  g_cond_clear (GST_TENSOR_REPO_GET_COND (nth));

  ret = g_hash_table_remove (_repo.hash, GINT_TO_POINTER (nth));

  if (ret) {
    _repo.num_data--;
    if (DBG)
      GST_DEBUG ("key[%d] is removed\n", nth);
  }

  if (!_repo.num_data) {
    g_hash_table_destroy (_repo.hash);
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

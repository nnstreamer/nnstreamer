/**
 * NNStreamer Tensor Repo Header's Contents
 * Copyright (C) 2018 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file	gsttensor_repo.c
 * @date	17 Nov 2018
 * @brief	tensor repo file for NNStreamer, the GStreamer plugin for neural networks
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "gsttensor_repo.h"

#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief tensor repo global variable with init.
 * @note The static repo_lock and repo_cond need no g_mutex_init() or g_cond_init().
 */
static GstTensorRepo _repo = {.num_data = 0,.initialized = FALSE };

/**
 * @brief Macro for Lock & Cond
 */
#define GST_REPO_LOCK() (g_mutex_lock(&_repo.repo_lock))
#define GST_REPO_UNLOCK() (g_mutex_unlock(&_repo.repo_lock))
#define GST_REPO_WAIT() (g_cond_wait(&_repo.repo_cond, &_repo.repo_lock))
#define GST_REPO_BROADCAST() (g_cond_broadcast (&_repo.repo_cond))

/**
 * @brief Internal function to drop a reference of repo data, releasing it with the last one.
 */
static void
gst_tensor_repo_unref_repodata (gpointer data)
{
  GstTensorRepoData *_data = (GstTensorRepoData *) data;
  g_return_if_fail (_data != NULL);

  if (!g_atomic_int_dec_and_test (&_data->refcount))
    return;

  if (_data->buffer)
    gst_buffer_unref (_data->buffer);
  if (_data->caps)
    gst_caps_unref (_data->caps);

  g_mutex_clear (&_data->lock);
  g_cond_clear (&_data->cond_pull);
  g_cond_clear (&_data->cond_push);

  g_free (_data);
}

/**
 * @brief Internal function to get a reference of nth GstTensorRepoData.
 * @return The slot data, to be released with gst_tensor_repo_unref_repodata(), or NULL.
 */
static GstTensorRepoData *
gst_tensor_repo_ref_repodata (guint nth)
{
  GstTensorRepoData *data;

  g_return_val_if_fail (_repo.initialized, NULL);

  GST_REPO_LOCK ();
  data = g_hash_table_lookup (_repo.hash, GINT_TO_POINTER (nth));
  if (data)
    g_atomic_int_inc (&data->refcount);
  GST_REPO_UNLOCK ();

  return data;
}

/**
 * @brief Set the changing status of repo.
 */
gboolean
gst_tensor_repo_set_changed (guint o_nth, guint nth, gboolean is_sink)
{
  GstTensorRepoData *data;

  data = gst_tensor_repo_ref_repodata (o_nth);

  if (data) {
    g_mutex_lock (&data->lock);

    if (is_sink) {
      data->sink_changed = TRUE;
      data->sink_id = nth;
      if (DBG)
        GST_DEBUG ("SET sink_changed! @id %d \n", o_nth);

      /* signal pull */
      g_cond_signal (&data->cond_pull);
    } else {
      data->src_changed = TRUE;
      data->src_id = nth;
      if (DBG)
        GST_DEBUG ("SET src_changed! @id %d\n", o_nth);

      /* signal push */
      g_cond_signal (&data->cond_push);
    }

    g_mutex_unlock (&data->lock);
    gst_tensor_repo_unref_repodata (data);
    return TRUE;
  }

  return FALSE;
}

/**
 * @brief Add GstTensorRepoData into repo.
 */
gboolean
gst_tensor_repo_add_repodata (guint nth, gboolean is_sink)
{
  GstTensorRepoData *data;

  g_return_val_if_fail (_repo.initialized, FALSE);

  GST_REPO_LOCK ();
  data = g_hash_table_lookup (_repo.hash, GINT_TO_POINTER (nth));

  if (data != NULL) {
    g_atomic_int_inc (&data->refcount);
    GST_REPO_UNLOCK ();

    g_mutex_lock (&data->lock);

    if (is_sink)
      data->sink_changed = FALSE;
    else
      data->src_changed = FALSE;

    data->pushed = FALSE;

    g_mutex_unlock (&data->lock);
    gst_tensor_repo_unref_repodata (data);

    if (DBG)
      GST_DEBUG ("SET SINK & SRC Changed FALSE!! @%d\n", nth);
    return TRUE;
  }

  data = g_new0 (GstTensorRepoData, 1);
  data->refcount = 1;
  g_cond_init (&data->cond_push);
  g_cond_init (&data->cond_pull);
  g_mutex_init (&data->lock);

  g_hash_table_insert (_repo.hash, GINT_TO_POINTER (nth), data);
  _repo.num_data++;

  if (DBG)
    GST_DEBUG ("Successfully added in hash table with key[%d]", nth);

  GST_REPO_UNLOCK ();
  return TRUE;
}

/**
 * @brief Push GstBuffer into repo.
 */
gboolean
gst_tensor_repo_set_buffer (guint nth, GstBuffer * buffer, GstCaps * caps)
{
  GstTensorRepoData *data;

  data = gst_tensor_repo_ref_repodata (nth);

  g_return_val_if_fail (data != NULL, FALSE);

  g_mutex_lock (&data->lock);

  while (data->buffer != NULL && !data->eos) {
    /* wait pull */
    g_cond_wait (&data->cond_pull, &data->lock);
  }

  if (data->eos) {
    g_mutex_unlock (&data->lock);
    gst_tensor_repo_unref_repodata (data);
    return FALSE;
  }

  data->buffer = gst_buffer_copy_deep (buffer);
  if (!data->caps || !gst_caps_is_equal (data->caps, caps)) {
    if (data->caps)
      gst_caps_unref (data->caps);
    data->caps = gst_caps_copy (caps);
  }

  if (DBG) {
    unsigned long size = gst_buffer_get_size (data->buffer);
    GST_DEBUG ("Pushed [%d] (size : %lu)\n", nth, size);
  }

  /* signal push */
  g_cond_signal (&data->cond_push);

  g_mutex_unlock (&data->lock);
  gst_tensor_repo_unref_repodata (data);
  return TRUE;
}

/**
 * @brief Check EOS (End-of-Stream) of slot.
 */
gboolean
gst_tensor_repo_check_eos (guint nth)
{
  GstTensorRepoData *data;
  gboolean eos = FALSE;

  data = gst_tensor_repo_ref_repodata (nth);

  if (data) {
    eos = data->eos;
    if (DBG)
      GST_DEBUG ("check eos done [%s]\n", eos ? "TRUE" : "FALSE");
    gst_tensor_repo_unref_repodata (data);
  }

  return eos;
}

/**
 * @brief Check repo data is changed.
 */
gboolean
gst_tensor_repo_check_changed (guint nth, guint * newid, gboolean is_sink)
{
  gboolean ret = FALSE;
  GstTensorRepoData *data;

  data = gst_tensor_repo_ref_repodata (nth);

  g_return_val_if_fail (data != NULL, FALSE);

  if (DBG)
    GST_DEBUG ("%dth RepoData : sink_changed %d, src_changed %d\n", nth,
        data->sink_changed, data->src_changed);

  if (is_sink) {
    if (data->sink_changed) {
      *newid = data->sink_id;
      ret = TRUE;
    }
  } else {
    if (data->src_changed) {
      *newid = data->src_id;
      ret = TRUE;
    }
  }

  gst_tensor_repo_unref_repodata (data);
  return ret;
}

/**
 * @brief Set EOS (End-of-Stream) of slot.
 */
gboolean
gst_tensor_repo_set_eos (guint nth)
{
  GstTensorRepoData *data;

  data = gst_tensor_repo_ref_repodata (nth);

  g_return_val_if_fail (data != NULL, FALSE);

  g_mutex_lock (&data->lock);

  data->eos = TRUE;
  g_cond_signal (&data->cond_push);
  g_cond_signal (&data->cond_pull);

  g_mutex_unlock (&data->lock);
  gst_tensor_repo_unref_repodata (data);
  return TRUE;
}

/**
 * @brief Get GstTensorRepoData from repo.
 */
GstBuffer *
gst_tensor_repo_get_buffer (guint nth, gboolean * eos, guint * newid,
    GstCaps ** caps)
{
  GstTensorRepoData *data;
  GstBuffer *buf = NULL;

  data = gst_tensor_repo_ref_repodata (nth);

  g_return_val_if_fail (data != NULL, NULL);

  g_mutex_lock (&data->lock);

  while (!data->buffer) {
    if (data->src_changed) {
      *newid = data->src_id;
      buf = NULL;
      goto done;
    }

    if (data->eos) {
      *eos = TRUE;
      buf = NULL;
      goto done;
    }

    /* wait push */
    g_cond_wait (&data->cond_push, &data->lock);
  }

  /* Current buffer will be wasted. */
  buf = data->buffer;
  *caps = gst_caps_ref (data->caps);
  if (DBG) {
    unsigned long size = gst_buffer_get_size (buf);
    GST_DEBUG ("Popped [ %d ] (size: %lu)\n", nth, size);
  }

done:
  data->buffer = NULL;
  /* signal pull */
  g_cond_signal (&data->cond_pull);
  g_mutex_unlock (&data->lock);
  gst_tensor_repo_unref_repodata (data);
  return buf;
}

/**
 * @brief Remove nth GstTensorRepoData from GstTensorRepo.
 */
gboolean
gst_tensor_repo_remove_repodata (guint nth)
{
  GstTensorRepoData *data;

  g_return_val_if_fail (_repo.initialized, FALSE);

  GST_REPO_LOCK ();
  data = g_hash_table_lookup (_repo.hash, GINT_TO_POINTER (nth));
  if (data) {
    g_hash_table_steal (_repo.hash, GINT_TO_POINTER (nth));
    _repo.num_data--;
  }
  GST_REPO_UNLOCK ();

  if (!data)
    return FALSE;

  /* wake up the elements still waiting on the slot, the last reference releases it */
  g_mutex_lock (&data->lock);
  data->eos = TRUE;
  g_cond_broadcast (&data->cond_push);
  g_cond_broadcast (&data->cond_pull);
  g_mutex_unlock (&data->lock);
  gst_tensor_repo_unref_repodata (data);

  if (DBG)
    GST_DEBUG ("key[%d] is removed\n", nth);
  return TRUE;
}

/**
 * @brief GstTensorRepo initialization.
 */
void
gst_tensor_repo_init (void)
{
  GST_REPO_LOCK ();
  if (!_repo.initialized) {
    _repo.num_data = 0;
    _repo.hash = g_hash_table_new_full (g_direct_hash, g_direct_equal, NULL,
        gst_tensor_repo_unref_repodata);
    _repo.initialized = TRUE;
    GST_REPO_BROADCAST ();
  }
  GST_REPO_UNLOCK ();
}

/**
 * @brief Wait for finish of initialization.
 */
gboolean
gst_tensor_repo_wait (void)
{
  GST_REPO_LOCK ();
  while (!_repo.initialized)
    GST_REPO_WAIT ();
  GST_REPO_UNLOCK ();
  return TRUE;
}

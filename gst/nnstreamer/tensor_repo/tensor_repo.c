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

#include "tensor_repo.h"

#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief tensor repo global variable with init.
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
 * @brief Define tensor_repo meta data type to register.
 */
GType
gst_meta_repo_api_get_type (void)
{
  static volatile GType type;
  static const gchar *tags[] = { "tensor", "tensors", NULL };
  if (g_once_init_enter (&type)) {
    GType _type;
    const GstMetaInfo *meta_info = gst_meta_get_info ("GstMetaRepo");
    if (meta_info) {
      _type = meta_info->api;
    } else {
      _type = gst_meta_api_type_register ("GstMetaRepoAPI", tags);
    }
    g_once_init_leave (&type, _type);
  }
  return type;
}

/**
 * @brief tensor_repo meta data init
 */
static gboolean
gst_meta_repo_init (GstMeta * meta, gpointer params, GstBuffer * buffer)
{
  GstMetaRepo *emeta = (GstMetaRepo *) meta;
  emeta->caps = NULL;
  return TRUE;
}

/**
 * @brief tensor_repo meta data transform (source to dest)
 */
static gboolean
gst_meta_repo_transform (GstBuffer * transbuf, GstMeta * meta,
    GstBuffer * buffer, GQuark type, gpointer data)
{
  GstMetaRepo *dest_meta = GST_META_REPO_ADD (transbuf);
  GstMetaRepo *src_meta = (GstMetaRepo *) meta;
  dest_meta->caps = src_meta->caps;
  return TRUE;
}

/**
 * @brief tensor_repo meta data free
 */
static void
gst_meta_repo_free (GstMeta * meta, GstBuffer * buffer)
{
  GstMetaRepo *emeta = (GstMetaRepo *) meta;
  emeta->caps = NULL;
}

/**
 * @brief Get tensor_repo meta data info.
 * @return GstMetaInfo
 */
const GstMetaInfo *
gst_meta_repo_get_info (void)
{
  static const GstMetaInfo *meta_info = NULL;
  if (g_once_init_enter (&meta_info)) {
    const GstMetaInfo *mi = gst_meta_register (GST_META_REPO_API_TYPE,
        "GstMetaRepo",
        sizeof (GstMetaRepo),
        (GstMetaInitFunction) gst_meta_repo_init,
        (GstMetaFreeFunction) gst_meta_repo_free,
        (GstMetaTransformFunction) gst_meta_repo_transform);
    g_once_init_leave (&meta_info, mi);
  }
  return meta_info;
}

/**
 * @brief Add get_tensor_repo meta data in buffer.
 * @return GstMetaRepo
 */
GstMetaRepo *
gst_buffer_add_meta_repo (GstBuffer * buffer)
{
  GstMetaRepo *meta;
  g_return_val_if_fail (GST_IS_BUFFER (buffer), NULL);
  meta = (GstMetaRepo *) gst_buffer_add_meta (buffer, GST_META_REPO_INFO, NULL);
  return meta;
}

/**
 * @brief Getter to get nth GstTensorRepoData.
 */
GstTensorRepoData *
gst_tensor_repo_get_repodata (guint nth)
{
  gpointer p;

  g_return_val_if_fail (_repo.initialized, NULL);

  p = g_hash_table_lookup (_repo.hash, GINT_TO_POINTER (nth));

  return (GstTensorRepoData *) p;
}

/**
 * @brief Set the changing status of repo.
 */
gboolean
gst_tensor_repo_set_changed (guint o_nth, guint nth, gboolean is_sink)
{
  GstTensorRepoData *data;

  data = gst_tensor_repo_get_repodata (o_nth);

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
  gboolean ret = FALSE;
  GstTensorRepoData *data;

  data = gst_tensor_repo_get_repodata (nth);

  if (data != NULL) {
    g_mutex_lock (&data->lock);

    if (is_sink)
      data->sink_changed = FALSE;
    else
      data->src_changed = FALSE;

    data->pushed = FALSE;

    g_mutex_unlock (&data->lock);

    if (DBG)
      GST_DEBUG ("SET SINK & SRC Changed FALSE!! @%d\n", nth);
    return TRUE;
  }

  data = g_new (GstTensorRepoData, 1);
  data->eos = FALSE;
  data->buffer = NULL;
  g_cond_init (&data->cond_push);
  g_cond_init (&data->cond_pull);
  g_mutex_init (&data->lock);
  data->sink_changed = FALSE;
  data->src_changed = FALSE;
  data->pushed = FALSE;

  GST_REPO_LOCK ();
  ret = g_hash_table_insert (_repo.hash, GINT_TO_POINTER (nth), data);
  g_assert (ret);

  if (ret) {
    _repo.num_data++;

    if (DBG)
      GST_DEBUG ("Successfully added in hash table with key[%d]", nth);
  }

  GST_REPO_UNLOCK ();
  return ret;
}

/**
 * @brief Push GstBuffer into repo.
 */
gboolean
gst_tensor_repo_set_buffer (guint nth, guint o_nth, GstBuffer * buffer,
    GstCaps * caps)
{
  GstTensorRepoData *data;
  GstMetaRepo *meta;

  data = gst_tensor_repo_get_repodata (nth);

  g_return_val_if_fail (data != NULL, FALSE);

  g_mutex_lock (&data->lock);

  while (data->buffer != NULL && !data->eos) {
    /* wait pull */
    g_cond_wait (&data->cond_pull, &data->lock);
  }

  if (data->eos) {
    g_mutex_unlock (&data->lock);
    return FALSE;
  }

  data->buffer = gst_buffer_copy (buffer);

  meta = GST_META_REPO_ADD (data->buffer);

  gst_caps_replace (&meta->caps, caps);

  if (DBG) {
    unsigned long size = gst_buffer_get_size (data->buffer);
    GST_DEBUG ("Pushed [%d] (size : %lu)\n", nth, size);
  }

  /* signal push */
  g_cond_signal (&data->cond_push);

  g_mutex_unlock (&data->lock);
  return TRUE;
}

/**
 * @brief Check EOS (End-of-Stream) of slot.
 */
gboolean
gst_tensor_repo_check_eos (guint nth)
{
  GstTensorRepoData *data;

  data = gst_tensor_repo_get_repodata (nth);

  if (data) {
    if (DBG)
      GST_DEBUG ("check eos done [%s]\n", data->eos ? "TRUE" : "FALSE");
    return data->eos;
  }

  return FALSE;
}

/**
 * @brief Check repo data is changed.
 */
gboolean
gst_tensor_repo_check_changed (guint nth, guint * newid, gboolean is_sink)
{
  gboolean ret = FALSE;
  GstTensorRepoData *data;

  data = gst_tensor_repo_get_repodata (nth);

  g_return_val_if_fail (data != NULL, FALSE);

  if (DBG)
    GST_DEBUG ("%dth RepoData : sink_chaned %d, src_changed %d\n", nth,
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

  return ret;
}

/**
 * @brief Set EOS (End-of-Stream) of slot.
 */
gboolean
gst_tensor_repo_set_eos (guint nth)
{
  GstTensorRepoData *data;

  data = gst_tensor_repo_get_repodata (nth);

  g_return_val_if_fail (data != NULL, FALSE);

  g_mutex_lock (&data->lock);

  data->eos = TRUE;
  g_cond_signal (&data->cond_push);
  g_cond_signal (&data->cond_pull);

  g_mutex_unlock (&data->lock);
  return TRUE;
}

/**
 * @brief Get GstTensorRepoData from repo.
 */
GstBuffer *
gst_tensor_repo_get_buffer (guint nth, guint o_nth, gboolean * eos,
    guint * newid)
{
  GstTensorRepoData *data;
  GstBuffer *buf = NULL;

  data = gst_tensor_repo_get_repodata (nth);

  g_return_val_if_fail (data != NULL, NULL);

  g_mutex_lock (&data->lock);

  while (!data->buffer) {
    if (gst_tensor_repo_check_changed (nth, newid, FALSE)) {
      buf = NULL;
      goto done;
    }

    if (gst_tensor_repo_check_eos (nth)) {
      *eos = TRUE;
      buf = NULL;
      goto done;
    }

    /* wait push */
    g_cond_wait (&data->cond_push, &data->lock);
  }

  buf = gst_buffer_copy_deep (data->buffer);
  gst_buffer_unref (data->buffer);
  if (DBG) {
    unsigned long size = gst_buffer_get_size (buf);
    GST_DEBUG ("Popped [ %d ] (size: %lu)\n", nth, size);
  }

done:
  data->buffer = NULL;
  /* signal pull */
  g_cond_signal (&data->cond_pull);
  g_mutex_unlock (&data->lock);
  return buf;
}

/**
 * @brief Remove nth GstTensorRepoData from GstTensorRepo.
 */
gboolean
gst_tensor_repo_remove_repodata (guint nth)
{
  gboolean ret = FALSE;
  GstTensorRepoData *data;

  g_return_val_if_fail (_repo.initialized, FALSE);

  GST_REPO_LOCK ();
  data = gst_tensor_repo_get_repodata (nth);

  if (data) {
    g_mutex_clear (&data->lock);
    g_cond_clear (&data->cond_pull);
    g_cond_clear (&data->cond_push);

    ret = g_hash_table_remove (_repo.hash, GINT_TO_POINTER (nth));

    if (ret) {
      _repo.num_data--;
      if (DBG)
        GST_DEBUG ("key[%d] is removed\n", nth);
    }
  }

  GST_REPO_UNLOCK ();
  return ret;
}

/**
 * @brief GstTensorRepo initialization.
 */
void
gst_tensor_repo_init (void)
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

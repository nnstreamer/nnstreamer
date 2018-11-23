/**
 * NNStreamer Tensor Repo Header
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
 * @file	tensor_repo.h
 * @date	17 Nov 2018
 * @brief	tensor repo header file for NNStreamer, the GStreamer plugin for neural networks
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __GST_TENSOR_REPO_H__
#define __GST_TENSOR_REPO_H__

#include <glib.h>
#include <stdint.h>
#include <tensor_common.h>
#include "tensor_typedef.h"
#include <gst/gst.h>

G_BEGIN_DECLS

/**
 * @brief GstTensorRepo internal data structure.
 *
 * GstTensorRepo has GSlist of GstTensorRepoData.
 *
 */

typedef struct
{
  GstBuffer *buffer;
  GCond cond;
  GMutex lock;
  gboolean eos;
} GstTensorRepoData;

/**
 * @brief GstTensorRepo data structure.
 */
typedef struct
{
  gint num_data;
  GMutex repo_lock;
  GCond repo_cond;
  GHashTable* hash;
  gboolean initialized;
} GstTensorRepo;

/**
 * @brief getter to get nth GstTensorRepoData
 */
GstTensorRepoData *
gst_tensor_repo_get_repodata (guint nth);

/**
 * @brief add GstTensorRepoData into repo
 */
/* guint */
gboolean
gst_tensor_repo_add_repodata (guint myid);

/**
 * @brief push GstBuffer into repo
 */
gboolean
gst_tensor_repo_set_buffer (guint nth, GstBuffer * buffer);

/**
 * @brief get EOS
 */
gboolean
gst_tensor_repo_check_eos(guint nth);

/**
 * @brief set EOS
 */
gboolean
gst_tensor_repo_set_eos(guint nth);

/**
 * @brief Get GstTensorRepoData from repo
 */
GstBuffer *
gst_tensor_repo_get_buffer (guint nth);

/**
 * @brief remove nth GstTensorRepoData from GstTensorRepo
 */
gboolean
gst_tensor_repo_remove_repodata (guint nth);

/**
 * @brief GstTensorRepo initialization
 */
void
gst_tensor_repo_init ();

/**
 * @brief Wait for the repo initialization
 */
gboolean
gst_tensor_repo_wait();


/**
 * @brief Macro for Lock & Cond
 */
#define GST_TENSOR_REPO_GET_LOCK(id) (&((GstTensorRepoData*)(gst_tensor_repo_get_repodata(id)))->lock)
#define GST_TENSOR_REPO_GET_COND(id) (&((GstTensorRepoData*)(gst_tensor_repo_get_repodata(id)))->cond)
#define GST_TENSOR_REPO_LOCK(id) (g_mutex_lock(GST_TENSOR_REPO_GET_LOCK(id)))
#define GST_TENSOR_REPO_UNLOCK(id) (g_mutex_unlock(GST_TENSOR_REPO_GET_LOCK(id)))
#define GST_TENSOR_REPO_WAIT(id) (g_cond_wait(GST_TENSOR_REPO_GET_COND(id), GST_TENSOR_REPO_GET_LOCK(id)))
#define GST_TENSOR_REPO_SIGNAL(id) (g_cond_signal (GST_TENSOR_REPO_GET_COND(id)))
#define GST_REPO_LOCK()(g_mutex_lock(&_repo.repo_lock))
#define GST_REPO_UNLOCK()(g_mutex_unlock(&_repo.repo_lock))
#define GST_REPO_WAIT() (g_cond_wait(&_repo.repo_cond, &_repo.repo_lock))
#define GST_REPO_BROADCAST() (g_cond_broadcast (&_repo.repo_cond))

G_END_DECLS
#endif /* __GST_TENSOR_REPO_H__ */

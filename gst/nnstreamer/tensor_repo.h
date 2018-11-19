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
 * GstTensorRepo has GSlist of GstTensorData.
 *
 */
typedef struct
{
  GstTensorConfig *config;
  GstBuffer *buffer;
  GCond cond;
  GMutex lock;
} GstTensorData;

/**
 * @brief GstTensorRepo data structure.
 */
struct GstTensorRepo_s
{
  gint num_buffer;
  GMutex repo_lock;
  GSList *tensorsdata;
  gboolean initialized;
} GstTensorRepo_default={.num_buffer=0, .tensorsdata=NULL, .initialized=FALSE};

typedef struct GstTensorRepo_s GstTensorRepo;

/**
 * @brief extern variable for GstTensorRepo
 */
extern GstTensorRepo _repo;

/**
 * @brief getter to get nth GstTensorData
 */
GstTensorData *
gst_tensor_repo_get_tensor (guint nth)
{
  return g_slist_nth_data (_repo.tensorsdata, nth);
}

/**
 * @brief Macro for Lock & Cond
 */
#define GST_TENSOR_REPO_GET_LOCK(id) (&((GstTensorData*)(gst_tensor_repo_get_tensor(id)))->lock)
#define GST_TENSOR_REPO_GET_COND(id) (&((GstTensorData*)(gst_tensor_repo_get_tensor(id)))->cond)
#define GST_TENSOR_REPO_LOCK(id) (g_mutex_lock(GST_TENSOR_REPO_GET_LOCK(id)))
#define GST_TENSOR_REPO_UNLOCK(id) (g_mutex_unlock(GST_TENSOR_REPO_GET_LOCK(id)))
#define GST_TENSOR_REPO_WAIT(id) (g_cond_wait(GST_TENSOR_REPO_GET_COND(id), GET_TENSOR_REPO_GET_LOCK(id)))
#define GST_TENSOR_REPO_BROADCAST(id) (g_cond_broadcast (GST_TENSOR_REPO_GET_COND(id)))

/**
 * @brief remove nth GstTensorData from GstTensorRepo
 */
void
gst_tensor_repo_remove_data (guint nth)
{
  g_mutex_lock (&_repo.repo_lock);
  GSList *data = g_slist_nth (_repo.tensorsdata, nth);
  _repo.tensorsdata = g_slist_delete_link (_repo.tensorsdata, data);
  g_mutex_unlock (&_repo.repo_lock);
}

/**
 * @brief GstTensorRepo initialization
 */
void
gst_tensor_repo_init()
{
  _repo.num_buffer=0;
  g_mutex_init (&_repo.repo_lock);
  _repo.tensorsdata=NULL;
  _repo.initialized=TRUE;
}

G_END_DECLS
#endif /* __GST_TENSOR_REPO_H__ */

/**
 * NNStreamer Tensor Repo Header
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
 * @file	tensor_repo.h
 * @date	17 Nov 2018
 * @brief	tensor repo header file for NNStreamer, the GStreamer plugin for neural networks
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __GST_TENSOR_REPO_H__
#define __GST_TENSOR_REPO_H__

#include <glib.h>
#include <gst/gst.h>

#include "tensor_common.h"

G_BEGIN_DECLS

/**
 * @brief GstTensorRepo meta structure.
 */
typedef struct
{
  GstMeta meta;
  GstCaps *caps;
} GstMetaRepo;

/**
 * @brief Define tensor_repo meta data type to register.
 */
GType gst_meta_repo_api_get_type (void);
#define GST_META_REPO_API_TYPE (gst_meta_repo_api_get_type())

/**
 * @brief Get tensor_repo meta data info.
 */
const GstMetaInfo *gst_meta_repo_get_info (void);
#define GST_META_REPO_INFO ((GstMetaInfo*) gst_meta_repo_get_info())

/**
 * @brief Macro of get_tensor_repo meta data.
 */
#define gst_buffer_get_meta_repo(b) \
  ((GstMetaRepo*) gst_buffer_get_meta((b), GST_META_REPO_API_TYPE))

/**
 * @brief Add get_tensor_repo meta data in buffer.
 */
GstMetaRepo *gst_buffer_add_meta_repo (GstBuffer * buffer);

/**
 * @brief Macro of get & add meta
 */
#define GST_META_REPO_GET(buf) ((GstMetaRepo*) gst_buffer_get_meta_repo(buf))
#define GST_META_REPO_ADD(buf) ((GstMetaRepo*) gst_buffer_add_meta_repo(buf))

/**
 * @brief GstTensorRepo internal data structure.
 *
 * GstTensorRepo has GSlist of GstTensorRepoData.
 */
typedef struct
{
  GstBuffer *buffer;
  GCond cond_push;
  GCond cond_pull;
  GMutex lock;
  gboolean eos;
  gboolean src_changed;
  guint src_id;
  gboolean sink_changed;
  guint sink_id;
  gboolean pushed;
} GstTensorRepoData;

/**
 * @brief GstTensorRepo data structure.
 */
typedef struct
{
  guint num_data;
  GMutex repo_lock;
  GCond repo_cond;
  GHashTable* hash;
  gboolean initialized;
} GstTensorRepo;

/**
 * @brief Getter to get nth GstTensorRepoData.
 */
GstTensorRepoData *
gst_tensor_repo_get_repodata (guint nth);

/**
 * @brief Add GstTensorRepoData into repo.
 */
gboolean
gst_tensor_repo_add_repodata (guint myid, gboolean is_sink);

/**
 * @brief Push GstBuffer into repo.
 */
gboolean
gst_tensor_repo_set_buffer (guint nth, guint o_nth, GstBuffer * buffer, GstCaps * caps);

/**
 * @brief Check EOS (End-of-Stream) of slot.
 */
gboolean
gst_tensor_repo_check_eos (guint nth);

/**
 * @brief Set EOS (End-of-Stream) of slot.
 */
gboolean
gst_tensor_repo_set_eos (guint nth);

/**
 * @brief Set the changing status of repo.
 */
gboolean
gst_tensor_repo_set_changed (guint o_nth, guint nth, gboolean is_sink);

/**
 * @brief Get GstTensorRepoData from repo.
 */
GstBuffer *
gst_tensor_repo_get_buffer (guint nth, guint o_nth, gboolean *eos, guint *newid);

/**
 * @brief Check repo data is changed.
 */
gboolean
gst_tensor_repo_check_changed (guint nth, guint *newid, gboolean is_sink);

/**
 * @brief Remove nth GstTensorRepoData from GstTensorRepo.
 */
gboolean
gst_tensor_repo_remove_repodata (guint nth);

/**
 * @brief GstTensorRepo initialization.
 */
void
gst_tensor_repo_init (void);

/**
 * @brief Wait for the repo initialization.
 */
gboolean
gst_tensor_repo_wait (void);

G_END_DECLS

#endif /* __GST_TENSOR_REPO_H__ */

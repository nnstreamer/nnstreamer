/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file	gstdatarepo.h
 * @date	27 June 2023
 * @brief	GStreamer plugin to read file in MLOps Data repository into buffers
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Hyunil Park <hyunil46.park@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_DATA_REPO_H__
#define __GST_DATA_REPO_H__

#include <gst/gst.h>

G_BEGIN_DECLS

/**
 * @brief Data type of incoming buffer.
 */
typedef enum
{
  GST_DATA_REPO_DATA_UNKNOWN = 0,
  GST_DATA_REPO_DATA_VIDEO,
  GST_DATA_REPO_DATA_AUDIO,
  GST_DATA_REPO_DATA_TEXT,
  GST_DATA_REPO_DATA_OCTET,
  GST_DATA_REPO_DATA_TENSOR,
  GST_DATA_REPO_DATA_IMAGE,

  GST_DATA_REPO_DATA_MAX
} GstDataRepoDataType;

/**
 * @brief Get data type from caps.
 */
GstDataRepoDataType
gst_data_repo_get_data_type_from_caps (const GstCaps * caps);

G_END_DECLS
#endif /* __GST_DATA_REPO_H__ */

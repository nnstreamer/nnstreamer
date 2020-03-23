/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 Samsung Electronics Co., Ltd.
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
 */

/**
 * @file	tensor_reposrc.h
 * @date	19 Nov 2018
 * @brief	GStreamer plugin to handle tensor repository
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_REPOSRC_H__
#define __GST_TENSOR_REPOSRC_H__

#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_REPOSRC \
  (gst_tensor_reposrc_get_type())
#define GST_TENSOR_REPOSRC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_REPOSRC,GstTensorRepoSrc))
#define GST_TENSOR_REPOSRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_REPOSRC,GstTensorRepoSrcClass))
#define GST_IS_TENSOR_REPOSRC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_REPOSRC))
#define GST_IS_TENSOR_REPOSRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_REPOSRC))

typedef struct _GstTensorRepoSrc GstTensorRepoSrc;
typedef struct _GstTensorRepoSrcClass GstTensorRepoSrcClass;

/**
 * @brief GstTensorRepoSrc data structure.
 *
 * GstTensorRepoSrc inherits GstPushSrc
 */
struct _GstTensorRepoSrc
{
  GstPushSrc parent;
  GstTensorsConfig config;
  gboolean silent;
  guint myid;
  guint o_myid;
  GstCaps *caps;
  gboolean ini;
  gint fps_n;
  gint fps_d;
  gboolean negotiation;
  gboolean set_startid;
};

/**
 * @brief GstTensorRepoSrcClass data structure.
 *
 * GstTensorRepoSrc inherits GstPushSrc
 */
struct _GstTensorRepoSrcClass
{
  GstPushSrcClass parent_class;
};

/**
 * @brief Function to get type of tensor_reposrc.
 */
GType gst_tensor_reposrc_get_type (void);

G_END_DECLS

#endif /* __GST_TENSOR_REPOSRC_H__ */

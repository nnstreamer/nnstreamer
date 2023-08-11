/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file    edge_src.h
 * @date    02 Aug 2022
 * @brief   Subscribe and push incoming data to the GStreamer pipeline
 * @author  Yechan Choi <yechan9.choi@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifndef __GST_EDGE_SRC_H__
#define __GST_EDGE_SRC_H__

#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>
#include "edge_common.h"
#include <nnstreamer-edge.h>
#include "nnstreamer_util.h"
#include "../nnstreamer/nnstreamer_log.h"

G_BEGIN_DECLS
#define GST_TYPE_EDGESRC \
    (gst_edgesrc_get_type())
#define GST_EDGESRC(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_EDGESRC,GstEdgeSrc))
#define GST_EDGESRC_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_EDGESRC,GstEdgeSrcClass))
#define GST_IS_EDGESRC(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_EDGESRC))
#define GST_IS_EDGESRC_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_EDGESRC))
#define GST_EDGESRC_CAST(obj) ((GstEdgeSrc *) (obj))
typedef struct _GstEdgeSrc GstEdgeSrc;
typedef struct _GstEdgeSrcClass GstEdgeSrcClass;

/**
 * @brief GstEdgeSrc data structure.
 */
struct _GstEdgeSrc
{
  GstBaseSrc element;

  gchar *host;
  guint16 port;
  gchar *dest_host;
  guint16 dest_port;
  gchar *topic;

  nns_edge_connect_type_e connect_type;
  nns_edge_h edge_h;
  GAsyncQueue *msg_queue;
};

/**
 * @brief GstEdgeSrcClass data structure.
 */
struct _GstEdgeSrcClass
{
  GstBaseSrcClass parent_class;
};

GType gst_edgesrc_get_type (void);

G_END_DECLS
#endif /* __GST_EDGE_SRC_H__ */

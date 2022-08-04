/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2022 Samsung Electronics Co., Ltd.
 *
 * @file    edge_sink.h
 * @date    01 Aug 2022
 * @brief   Publish incoming streams
 * @author  Yechan Choi <yechan9.choi@samsung.com>
 * @see     http://github.com/nnstreamer/nnstreamer
 * @bug     No known bugs
 *
 */
#ifndef __GST_EDGE_SINK_H__
#define __GST_EDGE_SINK_H__

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>

G_BEGIN_DECLS
#define GST_TYPE_EDGESINK \
    (gst_edgesink_get_type())
#define GST_EDGESINK(obj) \
    (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_EDGESINK,GstEdgeSink))
#define GST_EDGESINK_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_EDGESINK,GstEdgeSinkClass))
#define GST_IS_EDGESINK(obj) \
    (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_EDGESINK))
#define GST_IS_EDGESINK_CLASS(klass) \
    (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_EDGESINK))
#define GST_EDGESINK_CAST(obj) ((GstEdgeSink *)(obj))
typedef struct _GstEdgeSink GstEdgeSink;
typedef struct _GstEdgeSinkClass GstEdgeSinkClass;

/**
 * @brief GstEdgeSink data structure.
 */
struct _GstEdgeSink
{
  GstBaseSink element;

};

/**
 * @brief GstEdgeSinkClass data structure.
 */
struct _GstEdgeSinkClass
{
  GstBaseSinkClass parent_class;   /**<parent class */
};

GType gst_edgesink_get_type (void);

G_END_DECLS
#endif /* __GST_EDGE_SINK_H__ */

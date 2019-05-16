/**
 * GStreamer Android MediaCodec (AMC) Source
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All rights reserved.
 * Copyright (C) 2019 Dongju Chae <dongju.chae@samsung.com>
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
 * @file	  gstamcsrc.h
 * @date	  19 May 2019
 * @brief	  GStreamer source element for Android MediaCodec (AMC)
 * @see		  http://github.com/nnsuite/nnstreamer
 * @author	Dongju Chae <dongju.chae@samsung.com>
 * @bug		  No known bugs except for NYI items
 */

#ifndef __GST_AMC_SRC_H__
#define __GST_AMC_SRC_H__

#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>

G_BEGIN_DECLS

#define GST_TYPE_AMC_SRC              (gst_amc_src_get_type ())
#define GST_IS_AMC_SRC(obj)           (G_TYPE_CHECK_INSTANCE_TYPE ((obj), GST_TYPE_AMC_SRC))
#define GST_IS_AMC_SRC_CLASS(klass)   (G_TYPE_CHECK_CLASS_TYPE ((klass), GST_TYPE_AMC_SRC))
#define GST_AMC_SRC_GET_CLASS(obj)    (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_AMC_SRC, GstAMCSrcClass))
#define GST_AMC_SRC(obj)              (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_AMC_SRC, GstAMCSrc))
#define GST_AMC_SRC_CLASS(klass)      (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_AMC_SRC, GstAMCSrcClass))
#define GST_AMC_SRC_CAST(obj)         ((GstAMCSrc*)(obj))
#define GST_AMC_SRC_CLASS_CAST(klass) ((GstAMCSrcClass*)(klass))

typedef struct _GstAMCSrc GstAMCSrc;
typedef struct _GstAMCSrcClass GstAMCSrcClass;
typedef struct _GstAMCSrcPrivate GstAMCSrcPrivate;

/**
 * @brief GstAMCSrc data structure.
 *
 * GstAMCSrc inherits GstPushSrc.
 */
struct _GstAMCSrc
{
  GstPushSrc parent; /**< parent class object */
};

/**
 * @brief GstAMCSrcClass data structure.
 *
 * GstAMCSrcClass inherits GstPushSrcClass.
 */
struct _GstAMCSrcClass
{
  GstPushSrcClass parent_class; /**< inherits class object */
};

GST_EXPORT
GType gst_amc_src_get_type (void);

G_END_DECLS

#endif /** __GST_AMC_SRC_H__ */

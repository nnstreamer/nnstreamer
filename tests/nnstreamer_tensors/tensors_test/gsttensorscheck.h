/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
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
 */

/**
 * @file	gsttensorscheck.h
 * @date	26 June 2018
 * @brief	test element to check tensors
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSORSCHECK_H__
#define __GST_TENSORSCHECK_H__

#include <gst/gst.h>
#include <tensor_common.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSORSCHECK \
  (gst_tensorscheck_get_type())
#define GST_TENSORSCHECK(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSORSCHECK,Gsttensorscheck))
#define GST_TENSORSCHECK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSORSCHECK,GsttensorscheckClass))
#define GST_IS_TENSORSCHECK(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSORSCHECK))
#define GST_IS_TENSORSCHECK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSORSCHECK))

typedef struct _Gsttensorscheck Gsttensorscheck;
typedef struct _GsttensorscheckClass GsttensorscheckClass;

/**
 * @brief Internal data structure for tensorscheck instances.
 */
struct _Gsttensorscheck
{
  GstElement element;
  GstPad *sinkpad, *srcpad;

  gboolean silent;
  gboolean passthrough;

  /* For Tensor */
  GstTensorsConfig in_config;
  GstTensorConfig out_config;
};

/**
 * @brief GsttensorscheckClass inherits GstElementClass
 */
struct _GsttensorscheckClass
{
  GstElementClass parent_class;
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensorscheck_get_type (void);

G_END_DECLS

#endif /* __GST_TENSORSCHECK_H__ */

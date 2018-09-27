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
 * @file	gsttesttensors.h
 * @date	26 June 2018
 * @brief	test element to generate tensors
 * @see		https://github.com/nnsuite/nnstreamer
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TESTTENSORS_H__
#define __GST_TESTTENSORS_H__

#include <gst/gst.h>
#include <tensor_common.h>

G_BEGIN_DECLS

#define GST_TYPE_TESTTENSORS \
  (gst_testtensors_get_type())
#define GST_TESTTENSORS(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TESTTENSORS,Gsttesttensors))
#define GST_TESTTENSORS_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TESTTENSORS,GsttesttensorsClass))
#define GST_IS_TESTTENSORS(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TESTTENSORS))
#define GST_IS_TESTTENSORS_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TESTTENSORS))

typedef struct _Gsttesttensors Gsttesttensors;
typedef struct _GsttesttensorsClass GsttesttensorsClass;

/**
 * @brief Internal data structure for testtensors instances.
 */
struct _Gsttesttensors
{
  GstElement element;

  GstPad *sinkpad, *srcpad;

  gboolean silent;
  gboolean passthrough;

  /* For Tensor */
  GstTensorConfig in_config;
  GstTensorsConfig out_config;
};

/**
 * @brief Gsttesttensors inherits GstElementClass
 */
struct _GsttesttensorsClass
{
  GstElementClass parent_class;
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_testtensors_get_type (void);

G_END_DECLS

#endif /* __GST_TESTTENSORS_H__ */

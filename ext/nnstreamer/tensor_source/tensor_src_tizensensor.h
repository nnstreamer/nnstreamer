/**
 * GStreamer Tensor_Src_TizenSensor
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	tensor_src_tizensensor.h
 * @date	07 Nov 2019
 * @brief	GStreamer plugin to support Tizen sensor framework (sensord)
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_SRC_TIZENSENSOR_H__
#define __GST_TENSOR_SRC_TIZENSENSOR_H__

#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>

#include <tensor_typedef.h> /* GstTensorInfo */

#ifndef __TIZEN__
#error This plugin requires TIZEN packages.
#endif
/* Tizen Sensor Framework */
#include <sensor.h>

G_BEGIN_DECLS
#define GST_TYPE_TENSOR_SRC_TIZENSENSOR \
  (gst_tensor_src_tizensensor_get_type())
#define GST_TENSOR_SRC_TIZENSENSOR(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_SRC_TIZENSENSOR,GstTensorSrcTIZENSENSOR))
#define GST_TENSOR_SRC_TIZENSENSOR_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_SRC_TIZENSENSOR,GstTensorSrcTIZENSENSORClass))
#define GST_IS_TENSOR_SRC_TIZENSENSOR(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_SRC_TIZENSENSOR))
#define GST_IS_TENSOR_SRC_TIZENSENSOR_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_SRC_TIZENSENSOR))
#define GST_TENSOR_SRC_TIZENSENSOR_CAST(obj)  ((GstTensorSrcTIZENSENSOR *)(obj))
typedef struct _GstTensorSrcTIZENSENSOR GstTensorSrcTIZENSENSOR;
typedef struct _GstTensorSrcTIZENSENSORClass GstTensorSrcTIZENSENSORClass;

/**
 * @brief Sensor data retrieval modes
 * @details More entries coming soon!
 */
typedef enum
{
  TZN_SENSOR_MODE_POLLING = 0, /**< GST polls Tizen Sensor FW */
  /** @todo TZN_SENSOR_MODE_ACTIVE_POLLING ; Let Tizen poll */
  /** @todo TZN_SENSOR_MODE_WAIT_UPDATES ; wait for events from Tizen */
} sensor_op_modes;

/**
 * @brief GstTensorSrcTIZENSENSOR data structure.
 *
 * GstTensorSrcTIZENSENSOR inherits GstBaseSrcTIZENSENSOR.
 */
struct _GstTensorSrcTIZENSENSOR
{
  GstBaseSrc element; /**< parent class object */

  /** gstreamer related properties */
  gboolean silent; /**< true to print minimized log */
  gboolean configured; /**< true if device is configured and ready */
  gboolean running; /**< true if src is active and data is flowing */

  /** For managing critical sections */
  GMutex lock;

  /** Properties saved */
  sensor_type_e type; /**< Sensor type. "ALL" for unspecified. */
  gint sequence; /**< Sequence number. 0 for the first sensor of the type */
  sensor_op_modes mode; /**< Sensor data retrieval mode */
  gint freq_n; /**< Operating frequency of N/d */
  gint freq_d; /**< Operating frequency of n/D */

  /**
   * Sensor node info (handle, context)
   * These are temporary values valid during a session of "configured"
   * values should be cleared when confiured becomes FALSE
   */
  const GstTensorInfo *src_spec;
  unsigned int interval_ms;
  sensor_listener_h listener;
  sensor_h sensor;
};

/**
 * @brief GstTensorSrcTIZENSENSORClass data structure.
 *
 * GstTensorSrcTIZENSENSOR inherits GstBaseSrc.
 */
struct _GstTensorSrcTIZENSENSORClass
{
  GstBaseSrcClass parent_class; /**< inherits class object */
};

/**
 * @brief Function to get type of tensor_src_tizensensor.
 */
GType gst_tensor_src_tizensensor_get_type (void);

G_END_DECLS
#endif /** __GST_TENSOR_SRC_TIZENSENSOR_H__ */

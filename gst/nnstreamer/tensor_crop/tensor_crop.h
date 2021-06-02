/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file	tensor_crop.c
 * @date	10 May 2021
 * @brief	GStreamer element to crop the regions of incoming tensor
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_CROP_H__
#define __GST_TENSOR_CROP_H__

#include <gst/gst.h>
#include <gst/base/gstcollectpads.h>
#include <tensor_common.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_CROP \
  (gst_tensor_crop_get_type())
#define GST_TENSOR_CROP(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_CROP,GstTensorCrop))
#define GST_TENSOR_CROP_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_CROP,GstTensorCropClass))
#define GST_IS_TENSOR_CROP(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_CROP))
#define GST_IS_TENSOR_CROP_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_CROP))

typedef struct _GstTensorCrop GstTensorCrop;
typedef struct _GstTensorCropClass GstTensorCropClass;

/**
 * @brief GstTensorCrop pad data.
 */
typedef struct
{
  GstCollectData data;

  GstTensorsConfig config;
} GstTensorCropPadData;

/**
 * @brief GstTensorCrop data structure.
 */
struct _GstTensorCrop
{
  GstElement element; /**< parent object */

  GstPad *sinkpad_raw; /**< sink pad (raw data) */
  GstPad *sinkpad_info; /**< sink pad (crop info) */
  GstPad *srcpad; /**< src pad */

  /* <private> */
  gint lateness; /**< time-diff of raw and info buffer */
  gboolean silent; /**< true to print minimized log */
  gboolean send_stream_start; /**< flag to send STREAM_START event */
  GstCollectPads *collect; /**< sink pads */
};

/**
 * @brief GstTensorCropClass data structure.
 */
struct _GstTensorCropClass
{
  GstElementClass parent_class; /**< parent class */
};

/**
 * @brief Function to get type of tensor_crop.
 */
GType gst_tensor_crop_get_type (void);

G_END_DECLS

#endif /* __GST_TENSOR_CROP_H__ */

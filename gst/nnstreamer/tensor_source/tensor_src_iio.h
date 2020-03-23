/**
 * GStreamer Tensor_Src_IIO
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2019 Parichay Kapoor <pk.kapoor@samsung.com>
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
 * @file	tensor_src_iio.h
 * @date	26 Feb 2019
 * @brief	GStreamer plugin to support linux IIO as tensor(s)
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_SRC_IIO_H__
#define __GST_TENSOR_SRC_IIO_H__

#include <gst/gst.h>
#include <gst/base/gstbasesrc.h>
#include <glib/gprintf.h>
#include <tensor_common.h>
#include <poll.h>

G_BEGIN_DECLS
#define GST_TYPE_TENSOR_SRC_IIO \
  (gst_tensor_src_iio_get_type())
#define GST_TENSOR_SRC_IIO(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_SRC_IIO,GstTensorSrcIIO))
#define GST_TENSOR_SRC_IIO_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_SRC_IIO,GstTensorSrcIIOClass))
#define GST_IS_TENSOR_SRC_IIO(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_SRC_IIO))
#define GST_IS_TENSOR_SRC_IIO_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_SRC_IIO))
#define GST_TENSOR_SRC_IIO_CAST(obj)  ((GstTensorSrcIIO *)(obj))
typedef struct _GstTensorSrcIIO GstTensorSrcIIO;
typedef struct _GstTensorSrcIIOClass GstTensorSrcIIOClass;

/**
 * @brief iio device channel enabled mode
 */
typedef enum
{
  CHANNELS_ENABLED_ALL,
  CHANNELS_ENABLED_AUTO,
  CHANNELS_ENABLED_CUSTOM
} channels_enabled_options;

/**
 * @brief GstTensorSrcIIO devices's properties (internal data structure)
 *
 * This data structure is used for both device/triggers,
 * as triggers are also iio devices
 */
typedef struct _GstTensorSrcIIODeviceProperties
{
  gchar *name; /**< The name of the device */
  gchar *base_dir; /**< The base directory for the device */
  gint id; /**< The id of the device */
} GstTensorSrcIIODeviceProperties;

/**
 * @brief GstTensorSrcIIO channel's properties (internal data structure)
 */
typedef struct _GstTensorSrcIIOChannelProperties
{
  gboolean enabled; /**< currently state enabled/disabled */
  gboolean pre_enabled; /**< already in enabled/disabled state */
  gchar *name; /**< The name of the channel */
  gchar *generic_name; /**< The generic name of the channel */
  gchar *base_dir; /**< The base directory for the channel */
  gchar *base_file; /**< The base filename for the channel */
  gint index; /**< index of the channel in the buffer */

  gboolean big_endian; /**< endian-ness of the data in buffer */
  gboolean is_signed; /**< sign property of the data*/
  guint used_bits; /**< size of the bits used for the data */
  guint64 mask; /**< size of the bits used for the data */
  guint storage_bytes; /**< total storage size for the data */
  guint storage_bits; /**< exact bit size for the data */
  guint shift; /**< shift to be applied on the read data */
  guint location; /**< location of channel data in buffer */
  gfloat offset; /**< offset applied on raw data read from device */
  gfloat scale; /**< scale applied on offset-ed data read from device */
} GstTensorSrcIIOChannelProperties;

/**
 * @brief GstTensorSrcIIO data structure.
 *
 * GstTensorSrcIIO inherits GstBaseSrcIIO.
 */
struct _GstTensorSrcIIO
{
  GstBaseSrc element; /**< parent class object */

  /** gstreamer related properties */
  gboolean silent; /**< true to print minimized log */
  gboolean configured; /**< true if device is configured and ready */

  /** linux IIO related properties */
  gchar *mode; /**< IIO device operating mode */
  gchar *base_dir; /**< Base directory for IIO devices */
  gchar *dev_dir; /**< Directory for device files */
  GstTensorSrcIIODeviceProperties device; /**< IIO device */
  GstTensorSrcIIODeviceProperties trigger; /**< IIO trigger */
  GList *channels; /**< list of enabled channels */
  GHashTable *custom_channel_table; /**< table of idx of channels to be enabled */
  channels_enabled_options channels_enabled; /**< enabling which channels */
  guint scan_size; /**< size for a single scan of buffer length 1 */
  struct pollfd *buffer_data_fp; /**< pollfd for reading data buffer */
  guint num_channels_enabled; /**< channels to be enabled */
  gboolean merge_channels_data; /**< merge channel data with same type/size */
  gboolean is_tensor; /**< False if tensors is used for data */
  guint buffer_capacity; /**< size of the buffer */
  guint64 sampling_frequency; /**< sampling frequncy for the device */

  guint64 default_sampling_frequency; /**< default set value of sampling frequency */
  guint default_buffer_capacity; /**< size of the buffer */
  gchar *default_trigger; /**< default set value of sampling frequency */
  gint poll_timeout; /**< timeout for polling the fifo file */

  /** Only first element is filled when is_tensor is true */
  GstTensorsConfig *tensors_config; /**< tensors for storing data config */
};

/**
 * @brief GstTensorSrcIIOClass data structure.
 *
 * GstTensorSrcIIO inherits GstBaseSrc.
 */
struct _GstTensorSrcIIOClass
{
  GstBaseSrcClass parent_class; /**< inherits class object */
};

/**
 * @brief Function to get type of tensor_src_iio.
 */
GType gst_tensor_src_iio_get_type (void);

G_END_DECLS
#endif /** __GST_TENSOR_SRC_IIO_H__ */

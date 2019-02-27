/**
 * GStreamer Tensor_Source_IIO
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2019 Parichay Kapoor <pk.kapoor@samsung.com>
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
 * @file	tensor_src_iio.c
 * @date	27 Feb 2019
 * @brief	GStreamer plugin to capture sensor data as tensor(s)
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @todo  fill in empty functions
 * @todo  create a sample example and unit tests
 * @todo  set limit on triggers, channels, buffer capacity, frequency
 * @todo  verify if multiple triggers can be supported
 *
 * This is the plugin to capture data from sensors
 * and convert them to tensor format.
 * Current implementation will support accelerators, light and gyro sensors.
 *
 */

/**
 * SECTION:element-tensor_src_iio
 *
 * Source element to handle sensors as input.
 * The output are always in the format of other/tensor or other/tensors.
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gstinfo.h>
#include <gst/gst.h>
#include <glib.h>
#include <string.h>

#include "tensor_src_iio.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

/**
 * @brief Macro for debug message.
 */
#define silent_debug(...) do { \
    if (DBG) { \
      GST_DEBUG_OBJECT (self, __VA_ARGS__); \
    } \
  } while (0)

GST_DEBUG_CATEGORY_STATIC (gst_tensor_src_iio_debug);
#define GST_CAT_DEFAULT gst_tensor_src_iio_debug

/**
 * @brief tensor_src_iio properties.
 */
enum
{
  PROP_0,
  PROP_SILENT,
  PROP_DEVICE,
  PROP_TRIGGERS,
  PROP_VOLTAGE,
  PROP_CHANNELS,
  PROP_BUFFER_CAPACITY,
  PROP_FREQUENCY
};

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_SILENT TRUE

/**
 * @brief Flag for general default value of string
 */
#define DEFAULT_PROP_STRING NULL

/**
 * @brief Minimum and maximum buffer length for iio
 */
#define MIN_BUFFER_CAPACITY 1
#define MAX_BUFFER_CAPACITY 100
#define DEFAULT_BUFFER_CAPACITY 1

/**
 * @brief Minimum and maximum operating frequency for the device
 * Default frequency chooses the first available frequency supported by device
 */
#define MIN_FREQUENCY 1
#define MAX_FREQUENCY 999999999
#define DEFAULT_FREQUENCY 0

/**
 * @brief Template for src pad.
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT));

/** GObject method implementation */
static void gst_tensor_src_iio_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_src_iio_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_src_iio_finalize (GObject * object);

/** GstBaseSrc method implementation */
static gboolean gst_tensor_src_iio_start (GstBaseSrc * src);
static gboolean gst_tensor_src_iio_stop (GstBaseSrc * src);
static gboolean gst_tensor_src_iio_event (GstBaseSrc * src, GstEvent * event);
static gboolean gst_tensor_src_iio_query (GstBaseSrc * src, GstQuery * query);
static gboolean gst_tensor_src_iio_set_caps (GstBaseSrc * src, GstCaps * caps);
static GstCaps *gst_tensor_src_iio_get_caps (GstBaseSrc * src,
    GstCaps * filter);
static GstCaps *gst_tensor_src_iio_fixate (GstBaseSrc * src, GstCaps * caps);
static gboolean gst_tensor_src_iio_is_seekable (GstBaseSrc * src);
static GstFlowReturn gst_tensor_src_iio_create (GstBaseSrc * src,
    guint64 offset, guint size, GstBuffer ** buf);
static GstFlowReturn gst_tensor_src_iio_fill (GstBaseSrc * src, guint64 offset,
    guint size, GstBuffer * buf);

/** internal functions */

#define gst_tensor_src_iio_parent_class parent_class
G_DEFINE_TYPE (GstTensorSrcIIO, gst_tensor_src_iio, GST_TYPE_BASE_SRC);

/**
 * @brief initialize the tensor_src_iio class.
 */
static void
gst_tensor_src_iio_class_init (GstTensorSrcIIOClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSrcClass *bsrc_class;

  gobject_class = G_OBJECT_CLASS (klass);
  gstelement_class = GST_ELEMENT_CLASS (klass);
  bsrc_class = GST_BASE_SRC_CLASS (klass);

  /** GObject methods */
  gobject_class->set_property = gst_tensor_src_iio_set_property;
  gobject_class->get_property = gst_tensor_src_iio_get_property;
  gobject_class->finalize = gst_tensor_src_iio_finalize;

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent",
          "Produce verbose output", DEFAULT_SILENT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_DEVICE,
      g_param_spec_string ("device", "Device Name",
          "Name of the device to be opened", DEFAULT_PROP_STRING,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_TRIGGERS,
      g_param_spec_string ("triggers", "Triggers Name",
          "Name of the triggers to be used", DEFAULT_PROP_STRING,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_CHANNELS,
      g_param_spec_string ("channels", "Enabled Channels",
          "Channels to be enabled", DEFAULT_PROP_STRING, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_BUFFER_CAPACITY,
      g_param_spec_uint ("buffer_capacity", "Buffer Capacity",
          "Capacity of the data buffer", MIN_BUFFER_CAPACITY,
          MAX_BUFFER_CAPACITY, DEFAULT_BUFFER_CAPACITY,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_FREQUENCY,
      g_param_spec_long ("frequency", "Frequency",
          "Operating frequency of the device", MIN_FREQUENCY, MAX_FREQUENCY,
          DEFAULT_FREQUENCY, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata (gstelement_class,
      "TensorSrcIIO",
      "SrcIIO/Tensor",
      "Src element to support linux IIO",
      "Parichay Kapoor <pk.kapoor@samsung.com>");

  /** pad template */
  gst_element_class_add_static_pad_template (gstelement_class, &src_factory);

  /** GstBaseSrcIIO methods */
  bsrc_class->start = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_start);
  bsrc_class->stop = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_stop);
  bsrc_class->event = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_event);
  bsrc_class->query = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_query);
  bsrc_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_set_caps);
  bsrc_class->get_caps = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_get_caps);
  bsrc_class->fixate = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_fixate);
  bsrc_class->is_seekable = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_is_seekable);
  bsrc_class->create = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_create);
  bsrc_class->fill = GST_DEBUG_FUNCPTR (gst_tensor_src_iio_fill);
}

/**
 * @brief initialize tensor_src_iio element.
 */
static void
gst_tensor_src_iio_init (GstTensorSrcIIO * self)
{
  g_mutex_init (&self->mutex);

  /** init properties */
  /** already set using default values */

  return;
}

/**
 * @brief set properties of tensor_src_iio
 */
static void
gst_tensor_src_iio_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  //FIXME: fill this function
}

/**
 * @brief get tensor_src_iio properties
 */
static void
gst_tensor_src_iio_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  //FIXME: fill this function
}

/**
 * @brief finalize the instance
 */
static void
gst_tensor_src_iio_finalize (GObject * object)
{
  //FIXME: fill this function
}

/**
 * @brief start function, called when state changed null to ready.
 */
static gboolean
gst_tensor_src_iio_start (GstBaseSrc * src)
{
  /** load and init resources */
  //FIXME: fill this function
  return TRUE;
}

/**
 * @brief stop function, called when state changed ready to null.
 */
static gboolean
gst_tensor_src_iio_stop (GstBaseSrc * src)
{
  /** free resources */
  //FIXME: fill this function
  return TRUE;
}

/**
 * @brief handle events
 */
static gboolean
gst_tensor_src_iio_event (GstBaseSrc * src, GstEvent * event)
{
  //FIXME: fill this function
  return GST_BASE_SRC_CLASS (parent_class)->event (src, event);
}

/**
 * @brief handle queries
 */
static gboolean
gst_tensor_src_iio_query (GstBaseSrc * src, GstQuery * query)
{
  //FIXME: fill this function
  return GST_BASE_SRC_CLASS (parent_class)->query (src, query);
}

/**
 * @brief set new caps
 */
static gboolean
gst_tensor_src_iio_set_caps (GstBaseSrc * src, GstCaps * caps)
{
  //FIXME: fill this function
  return TRUE;
}

/**
 * @brief get caps of subclass
 */
static GstCaps *
gst_tensor_src_iio_get_caps (GstBaseSrc * src, GstCaps * filter)
{
  //FIXME: fill this function
  GstCaps *caps = NULL;
  return caps;
}

/**
 * @brief fixate the caps when needed during negotiation
 */
static GstCaps *
gst_tensor_src_iio_fixate (GstBaseSrc * src, GstCaps * caps)
{
  //FIXME: fill this function
  GstCaps *ret_caps = NULL;
  return ret_caps;
}

/**
 * @brief check if source supports seeking
 */
static gboolean
gst_tensor_src_iio_is_seekable (GstBaseSrc * src)
{
  /* iio sensors are live source without any support for seeking */
  return FALSE;
}

/**
 * @brief create a buffer with requested size and offset
 */
static GstFlowReturn
gst_tensor_src_iio_create (GstBaseSrc * src, guint64 offset,
    guint size, GstBuffer ** buf)
{
  //FIXME: fill this function
  return GST_FLOW_ERROR;
}

/**
 * @brief fill the buffer with data
 */
static GstFlowReturn
gst_tensor_src_iio_fill (GstBaseSrc * src, guint64 offset,
    guint size, GstBuffer * buf)
{
  //FIXME: fill this function
  return GST_FLOW_ERROR;
}

/**
 * @brief entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
NNSTREAMER_PLUGIN_INIT (tensor_src_iio)
{
  /**
   * debug category for filtering log messages
   */
  GST_DEBUG_CATEGORY_INIT (gst_tensor_src_iio_debug, "tensor_src_iio",
      0, "tensor_src_iio element");

  return gst_element_register (plugin, "tensor_src_iio", GST_RANK_NONE,
      GST_TYPE_TENSOR_SRC_IIO);
}

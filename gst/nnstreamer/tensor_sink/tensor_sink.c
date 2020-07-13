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
 * SECTION:element-tensor_sink
 *
 * Sink element to handle tensor stream
 *
 * @file	tensor_sink.c
 * @date	15 June 2018
 * @brief	GStreamer plugin to handle tensor stream
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "tensor_sink.h"

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

#define silent_debug_timestamp(buf) do { \
  if (DBG) { \
    GST_DEBUG_OBJECT (self, "pts = %" GST_TIME_FORMAT, GST_TIME_ARGS (GST_BUFFER_PTS (buf))); \
    GST_DEBUG_OBJECT (self, "dts = %" GST_TIME_FORMAT, GST_TIME_ARGS (GST_BUFFER_DTS (buf))); \
    GST_DEBUG_OBJECT (self, "duration = %" GST_TIME_FORMAT "\n", GST_TIME_ARGS (GST_BUFFER_DURATION (buf))); \
  } \
} while (0)

GST_DEBUG_CATEGORY_STATIC (gst_tensor_sink_debug);
#define GST_CAT_DEFAULT gst_tensor_sink_debug

/**
 * @brief tensor_sink signals.
 */
enum
{
  SIGNAL_NEW_DATA,
  SIGNAL_STREAM_START,
  SIGNAL_EOS,
  LAST_SIGNAL
};

/**
 * @brief tensor_sink properties.
 */
enum
{
  PROP_0,
  PROP_SIGNAL_RATE,
  PROP_EMIT_SIGNAL,
  PROP_SILENT
};

/**
 * @brief Flag to emit signals.
 */
#define DEFAULT_EMIT_SIGNAL TRUE

/**
 * @brief New data signals per second.
 */
#define DEFAULT_SIGNAL_RATE 0

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_SILENT TRUE

/**
 * @brief Flag for qos event.
 *
 * See GstBaseSink::qos property for more details.
 */
#define DEFAULT_QOS TRUE

/**
 * @brief Flag to synchronize on the clock (Default FALSE).
 * It may be delayed with tensor_filter element, to invoke neural network model.
 * See GstBaseSink::sync property for more details.
 */
#define DEFAULT_SYNC FALSE

/**
 * @brief Variable for signal ids.
 */
static guint _tensor_sink_signals[LAST_SIGNAL] = { 0 };

/** GObject method implementation */
static void gst_tensor_sink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_sink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_sink_finalize (GObject * object);

/** GstBaseSink method implementation */
static gboolean gst_tensor_sink_event (GstBaseSink * sink, GstEvent * event);
static gboolean gst_tensor_sink_query (GstBaseSink * sink, GstQuery * query);
static GstFlowReturn gst_tensor_sink_render (GstBaseSink * sink,
    GstBuffer * buffer);
static GstFlowReturn gst_tensor_sink_render_list (GstBaseSink * sink,
    GstBufferList * buffer_list);

/** internal functions */
static void gst_tensor_sink_render_buffer (GstTensorSink * self,
    GstBuffer * buffer);
static void gst_tensor_sink_set_last_render_time (GstTensorSink * self,
    GstClockTime now);
static GstClockTime gst_tensor_sink_get_last_render_time (GstTensorSink * self);
static void gst_tensor_sink_set_signal_rate (GstTensorSink * self, guint rate);
static guint gst_tensor_sink_get_signal_rate (GstTensorSink * self);
static void gst_tensor_sink_set_emit_signal (GstTensorSink * self,
    gboolean emit);
static gboolean gst_tensor_sink_get_emit_signal (GstTensorSink * self);
static void gst_tensor_sink_set_silent (GstTensorSink * self, gboolean silent);
static gboolean gst_tensor_sink_get_silent (GstTensorSink * self);

#define gst_tensor_sink_parent_class parent_class
G_DEFINE_TYPE (GstTensorSink, gst_tensor_sink, GST_TYPE_BASE_SINK);

/**
 * @brief Initialize tensor_sink class.
 */
static void
gst_tensor_sink_class_init (GstTensorSinkClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *element_class;
  GstBaseSinkClass *bsink_class;
  GstPadTemplate *pad_template;
  GstCaps *pad_caps;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_sink_debug, "tensor_sink", 0,
      "Sink element to handle tensor stream");

  gobject_class = G_OBJECT_CLASS (klass);
  element_class = GST_ELEMENT_CLASS (klass);
  bsink_class = GST_BASE_SINK_CLASS (klass);

  /** GObject methods */
  gobject_class->set_property = gst_tensor_sink_set_property;
  gobject_class->get_property = gst_tensor_sink_get_property;
  gobject_class->finalize = gst_tensor_sink_finalize;

  /**
   * GstTensorSink::signal-rate:
   *
   * The number of new data signals per second (Default 0 for unlimited, MAX 500)
   * If signal-rate is larger than 0, GstTensorSink calculates the time to emit a signal with this property.
   * If set 0 (default value), all the received buffers will be passed to the application.
   *
   * Please note that this property does not guarantee the periodic signals.
   * This means if GstTensorSink cannot get the buffers in time, it will pass all the buffers. (working like default 0)
   */
  g_object_class_install_property (gobject_class, PROP_SIGNAL_RATE,
      g_param_spec_uint ("signal-rate", "Signal rate",
          "New data signals per second (0 for unlimited, max 500)", 0, 500,
          DEFAULT_SIGNAL_RATE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorSink::emit-signal:
   *
   * The flag to emit the signals for new data, stream start, and eos.
   */
  g_object_class_install_property (gobject_class, PROP_EMIT_SIGNAL,
      g_param_spec_boolean ("emit-signal", "Emit signal",
          "Emit signal for new data, stream start, eos", DEFAULT_EMIT_SIGNAL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorSink::silent:
   *
   * The flag to enable/disable debugging messages.
   */
  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorSink::new-data:
   *
   * Signal to get the buffer from GstTensorSink.
   */
  _tensor_sink_signals[SIGNAL_NEW_DATA] =
      g_signal_new ("new-data", G_TYPE_FROM_CLASS (klass), G_SIGNAL_RUN_LAST,
      G_STRUCT_OFFSET (GstTensorSinkClass, new_data), NULL, NULL, NULL,
      G_TYPE_NONE, 1, GST_TYPE_BUFFER | G_SIGNAL_TYPE_STATIC_SCOPE);

  /**
   * GstTensorSink::stream-start:
   *
   * Signal to indicate the start of a new stream.
   * Optional. An application can use this signal to detect the start of a new stream, instead of the message GST_MESSAGE_STREAM_START from pipeline.
   */
  _tensor_sink_signals[SIGNAL_STREAM_START] =
      g_signal_new ("stream-start", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST, G_STRUCT_OFFSET (GstTensorSinkClass, stream_start),
      NULL, NULL, NULL, G_TYPE_NONE, 0, G_TYPE_NONE);

  /**
   * GstTensorSink::eos:
   *
   * Signal to indicate the end-of-stream.
   * Optional. An application can use this signal to detect the EOS (end-of-stream), instead of the message GST_MESSAGE_EOS from pipeline.
   */
  _tensor_sink_signals[SIGNAL_EOS] =
      g_signal_new ("eos", G_TYPE_FROM_CLASS (klass), G_SIGNAL_RUN_LAST,
      G_STRUCT_OFFSET (GstTensorSinkClass, eos), NULL, NULL, NULL,
      G_TYPE_NONE, 0, G_TYPE_NONE);

  gst_element_class_set_static_metadata (element_class,
      "TensorSink",
      "Sink/Tensor",
      "Sink element to handle tensor stream", "Samsung Electronics Co., Ltd.");

  /** pad template */
  pad_caps = gst_caps_from_string (GST_TENSOR_CAP_DEFAULT "; "
      GST_TENSORS_CAP_DEFAULT);
  pad_template = gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
      pad_caps);
  gst_element_class_add_pad_template (element_class, pad_template);
  gst_caps_unref (pad_caps);

  /** GstBaseSink methods */
  bsink_class->event = GST_DEBUG_FUNCPTR (gst_tensor_sink_event);
  bsink_class->query = GST_DEBUG_FUNCPTR (gst_tensor_sink_query);
  bsink_class->render = GST_DEBUG_FUNCPTR (gst_tensor_sink_render);
  bsink_class->render_list = GST_DEBUG_FUNCPTR (gst_tensor_sink_render_list);
}

/**
 * @brief Initialize tensor_sink element.
 */
static void
gst_tensor_sink_init (GstTensorSink * self)
{
  GstBaseSink *bsink;

  bsink = GST_BASE_SINK (self);

  g_mutex_init (&self->mutex);

  /** init properties */
  self->silent = DEFAULT_SILENT;
  self->emit_signal = DEFAULT_EMIT_SIGNAL;
  self->signal_rate = DEFAULT_SIGNAL_RATE;
  self->last_render_time = GST_CLOCK_TIME_NONE;

  /** enable qos */
  gst_base_sink_set_qos_enabled (bsink, DEFAULT_QOS);

  /* set 'sync' to synchronize on the clock or not */
  gst_base_sink_set_sync (bsink, DEFAULT_SYNC);
}

/**
 * @brief Setter for tensor_sink properties.
 *
 * GObject method implementation.
 */
static void
gst_tensor_sink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorSink *self;

  self = GST_TENSOR_SINK (object);

  switch (prop_id) {
    case PROP_SIGNAL_RATE:
      gst_tensor_sink_set_signal_rate (self, g_value_get_uint (value));
      break;

    case PROP_EMIT_SIGNAL:
      gst_tensor_sink_set_emit_signal (self, g_value_get_boolean (value));
      break;

    case PROP_SILENT:
      gst_tensor_sink_set_silent (self, g_value_get_boolean (value));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter for tensor_sink properties.
 *
 * GObject method implementation.
 */
static void
gst_tensor_sink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorSink *self;

  self = GST_TENSOR_SINK (object);

  switch (prop_id) {
    case PROP_SIGNAL_RATE:
      g_value_set_uint (value, gst_tensor_sink_get_signal_rate (self));
      break;

    case PROP_EMIT_SIGNAL:
      g_value_set_boolean (value, gst_tensor_sink_get_emit_signal (self));
      break;

    case PROP_SILENT:
      g_value_set_boolean (value, gst_tensor_sink_get_silent (self));
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Function to finalize instance.
 *
 * GObject method implementation.
 */
static void
gst_tensor_sink_finalize (GObject * object)
{
  GstTensorSink *self;

  self = GST_TENSOR_SINK (object);

  g_mutex_clear (&self->mutex);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Handle events.
 *
 * GstBaseSink method implementation.
 */
static gboolean
gst_tensor_sink_event (GstBaseSink * sink, GstEvent * event)
{
  GstTensorSink *self;
  GstEventType type;

  self = GST_TENSOR_SINK (sink);
  type = GST_EVENT_TYPE (event);

  GST_DEBUG_OBJECT (self, "Received %s event: %" GST_PTR_FORMAT,
      GST_EVENT_TYPE_NAME (event), event);

  switch (type) {
    case GST_EVENT_STREAM_START:
      if (gst_tensor_sink_get_emit_signal (self)) {
        silent_debug ("Emit signal for stream start");

        g_signal_emit (self, _tensor_sink_signals[SIGNAL_STREAM_START], 0);
      }
      break;

    case GST_EVENT_EOS:
      if (gst_tensor_sink_get_emit_signal (self)) {
        silent_debug ("Emit signal for eos");

        g_signal_emit (self, _tensor_sink_signals[SIGNAL_EOS], 0);
      }
      break;

    default:
      break;
  }

  return GST_BASE_SINK_CLASS (parent_class)->event (sink, event);
}

/**
 * @brief Handle queries.
 *
 * GstBaseSink method implementation.
 */
static gboolean
gst_tensor_sink_query (GstBaseSink * sink, GstQuery * query)
{
  GstTensorSink *self;
  GstQueryType type;
  GstFormat format;

  self = GST_TENSOR_SINK (sink);
  type = GST_QUERY_TYPE (query);

  GST_DEBUG_OBJECT (self, "Received %s query: %" GST_PTR_FORMAT,
      GST_QUERY_TYPE_NAME (query), query);

  switch (type) {
    case GST_QUERY_SEEKING:
      /** tensor sink does not support seeking */
      gst_query_parse_seeking (query, &format, NULL, NULL, NULL);
      gst_query_set_seeking (query, format, FALSE, 0, -1);
      return TRUE;

    default:
      break;
  }

  return GST_BASE_SINK_CLASS (parent_class)->query (sink, query);
}

/**
 * @brief Handle buffer.
 *
 * GstBaseSink method implementation.
 */
static GstFlowReturn
gst_tensor_sink_render (GstBaseSink * sink, GstBuffer * buffer)
{
  GstTensorSink *self;

  self = GST_TENSOR_SINK (sink);
  gst_tensor_sink_render_buffer (self, buffer);

  return GST_FLOW_OK;
}

/**
 * @brief Handle list of buffers.
 *
 * GstBaseSink method implementation.
 */
static GstFlowReturn
gst_tensor_sink_render_list (GstBaseSink * sink, GstBufferList * buffer_list)
{
  GstTensorSink *self;
  GstBuffer *buffer;
  guint i;
  guint num_buffers;

  self = GST_TENSOR_SINK (sink);
  num_buffers = gst_buffer_list_length (buffer_list);

  for (i = 0; i < num_buffers; i++) {
    buffer = gst_buffer_list_get (buffer_list, i);
    gst_tensor_sink_render_buffer (self, buffer);
  }

  return GST_FLOW_OK;
}

/**
 * @brief Handle buffer data.
 * @return None
 * @param self pointer to GstTensorSink
 * @param buffer pointer to GstBuffer to be handled
 */
static void
gst_tensor_sink_render_buffer (GstTensorSink * self, GstBuffer * buffer)
{
  GstClockTime now = GST_CLOCK_TIME_NONE;
  guint signal_rate;
  gboolean notify = FALSE;

  g_return_if_fail (GST_IS_TENSOR_SINK (self));

  signal_rate = gst_tensor_sink_get_signal_rate (self);

  if (signal_rate) {
    GstClock *clock;
    GstClockTime render_time;
    GstClockTime last_render_time;

    clock = gst_element_get_clock (GST_ELEMENT (self));

    if (clock) {
      now = gst_clock_get_time (clock);
      last_render_time = gst_tensor_sink_get_last_render_time (self);

      /** time for next signal */
      render_time = (1000 / signal_rate) * GST_MSECOND + last_render_time;

      if (!GST_CLOCK_TIME_IS_VALID (last_render_time) ||
          GST_CLOCK_DIFF (now, render_time) <= 0) {
        /** send data after render time, or firstly received buffer */
        notify = TRUE;
      }

      gst_object_unref (clock);
    }
  } else {
    /** send data if signal rate is 0 */
    notify = TRUE;
  }

  if (notify) {
    gst_tensor_sink_set_last_render_time (self, now);

    if (gst_tensor_sink_get_emit_signal (self)) {
      silent_debug ("Emit signal for new data [%" GST_TIME_FORMAT "] rate [%d]",
          GST_TIME_ARGS (now), signal_rate);

      g_signal_emit (self, _tensor_sink_signals[SIGNAL_NEW_DATA], 0, buffer);
    }
  }

  silent_debug_timestamp (buffer);
}

/**
 * @brief Setter for value last_render_time.
 */
static void
gst_tensor_sink_set_last_render_time (GstTensorSink * self, GstClockTime now)
{
  g_return_if_fail (GST_IS_TENSOR_SINK (self));

  g_mutex_lock (&self->mutex);
  self->last_render_time = now;
  g_mutex_unlock (&self->mutex);
}

/**
 * @brief Getter for value last_render_time.
 */
static GstClockTime
gst_tensor_sink_get_last_render_time (GstTensorSink * self)
{
  GstClockTime last_render_time;

  g_return_val_if_fail (GST_IS_TENSOR_SINK (self), GST_CLOCK_TIME_NONE);

  g_mutex_lock (&self->mutex);
  last_render_time = self->last_render_time;
  g_mutex_unlock (&self->mutex);

  return last_render_time;
}

/**
 * @brief Setter for value signal_rate.
 */
static void
gst_tensor_sink_set_signal_rate (GstTensorSink * self, guint rate)
{
  g_return_if_fail (GST_IS_TENSOR_SINK (self));

  GST_INFO_OBJECT (self, "set signal_rate to %d", rate);
  g_mutex_lock (&self->mutex);
  self->signal_rate = rate;
  g_mutex_unlock (&self->mutex);
}

/**
 * @brief Getter for value signal_rate.
 */
static guint
gst_tensor_sink_get_signal_rate (GstTensorSink * self)
{
  guint rate;

  g_return_val_if_fail (GST_IS_TENSOR_SINK (self), 0);

  g_mutex_lock (&self->mutex);
  rate = self->signal_rate;
  g_mutex_unlock (&self->mutex);

  return rate;
}

/**
 * @brief Setter for flag emit_signal.
 */
static void
gst_tensor_sink_set_emit_signal (GstTensorSink * self, gboolean emit)
{
  g_return_if_fail (GST_IS_TENSOR_SINK (self));

  GST_INFO_OBJECT (self, "set emit_signal to %d", emit);
  g_mutex_lock (&self->mutex);
  self->emit_signal = emit;
  g_mutex_unlock (&self->mutex);
}

/**
 * @brief Getter for flag emit_signal.
 */
static gboolean
gst_tensor_sink_get_emit_signal (GstTensorSink * self)
{
  gboolean res;

  g_return_val_if_fail (GST_IS_TENSOR_SINK (self), FALSE);

  g_mutex_lock (&self->mutex);
  res = self->emit_signal;
  g_mutex_unlock (&self->mutex);

  return res;
}

/**
 * @brief Setter for flag silent.
 */
static void
gst_tensor_sink_set_silent (GstTensorSink * self, gboolean silent)
{
  g_return_if_fail (GST_IS_TENSOR_SINK (self));

  GST_INFO_OBJECT (self, "set silent to %d", silent);
  self->silent = silent;
}

/**
 * @brief Getter for flag silent.
 */
static gboolean
gst_tensor_sink_get_silent (GstTensorSink * self)
{
  g_return_val_if_fail (GST_IS_TENSOR_SINK (self), TRUE);

  return self->silent;
}

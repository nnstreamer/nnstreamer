/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	tensor_converter.c
 * @date	26 Mar 2018
 * @brief	GStreamer plugin to convert media types to tensors (as a filter for other general neural network filters)
 * @see		https://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 */

/**
 * SECTION:element-tensor_converter
 *
 * A filter that converts media stream to tensor stream for NN frameworks.
 * The output is always in the format of other/tensor.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 videotestsrc ! video/x-raw,format=RGB,width=640,height=480 ! tensor_converter ! tensor_sink
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include "tensor_converter.h"
#include <gst/video/video-info.h>
#ifndef NO_AUDIO
#include <gst/audio/audio-info.h>
#endif

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

#define silent_debug_caps(caps,msg) do { \
  if (DBG) { \
    if (caps) { \
      GstStructure *caps_s; \
      gchar *caps_s_string; \
      guint caps_size, caps_idx; \
      caps_size = gst_caps_get_size (caps);\
      for (caps_idx = 0; caps_idx < caps_size; caps_idx++) { \
        caps_s = gst_caps_get_structure (caps, caps_idx); \
        caps_s_string = gst_structure_to_string (caps_s); \
        GST_DEBUG_OBJECT (self, msg " = %s\n", caps_s_string); \
        g_free (caps_s_string); \
      } \
    } \
  } \
} while (0)

#define silent_debug_timestamp(buf) do { \
  if (DBG) { \
    GST_DEBUG_OBJECT (self, "pts = %" GST_TIME_FORMAT, GST_TIME_ARGS (GST_BUFFER_PTS (buf))); \
    GST_DEBUG_OBJECT (self, "dts = %" GST_TIME_FORMAT, GST_TIME_ARGS (GST_BUFFER_DTS (buf))); \
    GST_DEBUG_OBJECT (self, "duration = %" GST_TIME_FORMAT "\n", GST_TIME_ARGS (GST_BUFFER_DURATION (buf))); \
  } \
} while (0)

GST_DEBUG_CATEGORY_STATIC (gst_tensor_converter_debug);
#define GST_CAT_DEFAULT gst_tensor_converter_debug

/**
 * @brief tensor_converter properties
 */
enum
{
  PROP_0,
  PROP_INPUT_DIMENSION,
  PROP_INPUT_TYPE,
  PROP_FRAMES_PER_TENSOR,
  PROP_SET_TIMESTAMP,
  PROP_SILENT
};

/**
 * @brief Flag to set timestamp when received a buffer with invalid timestamp.
 */
#define DEFAULT_SET_TIMESTAMP TRUE

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_SILENT TRUE

/**
 * @brief Frames in output tensor.
 */
#define DEFAULT_FRAMES_PER_TENSOR 1

/**
 * @brief Template for sink pad.
 */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_MEDIA_CAPS_STR));

/**
 * @brief Template for src pad.
 */
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSOR_CAP_DEFAULT));

#define gst_tensor_converter_parent_class parent_class
G_DEFINE_TYPE (GstTensorConverter, gst_tensor_converter, GST_TYPE_ELEMENT);

static void gst_tensor_converter_finalize (GObject * object);
static void gst_tensor_converter_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_tensor_converter_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

static gboolean gst_tensor_converter_sink_event (GstPad * pad,
    GstObject * parent, GstEvent * event);
static gboolean gst_tensor_converter_sink_query (GstPad * pad,
    GstObject * parent, GstQuery * query);
static gboolean gst_tensor_converter_src_query (GstPad * pad,
    GstObject * parent, GstQuery * query);
static GstFlowReturn gst_tensor_converter_chain (GstPad * pad,
    GstObject * parent, GstBuffer * buf);
static GstStateChangeReturn
gst_tensor_converter_change_state (GstElement * element,
    GstStateChange transition);

static void gst_tensor_converter_reset (GstTensorConverter * self);
static GstCaps *gst_tensor_converter_query_caps (GstTensorConverter * self,
    GstPad * pad, GstCaps * filter);
static gboolean gst_tensor_converter_parse_caps (GstTensorConverter * self,
    const GstCaps * caps);

/**
 * @brief Initialize the tensor_converter's class.
 */
static void
gst_tensor_converter_class_init (GstTensorConverterClass * klass)
{
  GObjectClass *object_class;
  GstElementClass *element_class;

  object_class = (GObjectClass *) klass;
  element_class = (GstElementClass *) klass;

  object_class->set_property = gst_tensor_converter_set_property;
  object_class->get_property = gst_tensor_converter_get_property;
  object_class->finalize = gst_tensor_converter_finalize;

  /**
   * GstTensorConverter::input-dim:
   *
   * Input tensor dimension from inner array.
   * Generally this property is used to set tensor configuration for byte-stream (application/octet-stream).
   * When setting this property and input media type is video or audio stream, GstTensorConverter will compare the media info with this.
   * (If it is different, it will be failed.)
   */
  g_object_class_install_property (object_class, PROP_INPUT_DIMENSION,
      g_param_spec_string ("input-dim", "Input tensor dimension",
          "Input tensor dimension from inner array", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorConverter::input-type:
   *
   * Type of each element of the input tensor.
   * Generally this property is used to set tensor configuration for byte-stream (application/octet-stream).
   * When setting this property and input media type is video or audio stream, GstTensorConverter will compare the media info with this.
   * (If it is different, it will be failed.)
   */
  g_object_class_install_property (object_class, PROP_INPUT_TYPE,
      g_param_spec_string ("input-type", "Input tensor type",
          "Type of each element of the input tensor", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorConverter::frames-per-tensor:
   *
   * The number of frames in outgoing buffer. (buffer is a sinle tensor instance)
   * GstTensorConverter can push a buffer with multiple media frames.
   */
  g_object_class_install_property (object_class, PROP_FRAMES_PER_TENSOR,
      g_param_spec_uint ("frames-per-tensor", "Frames per tensor",
          "The number of frames in output tensor", 1, G_MAXUINT,
          DEFAULT_FRAMES_PER_TENSOR,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorConverter::set-timestamp:
   *
   * The flag to set timestamp when received a buffer with invalid timestamp.
   */
  g_object_class_install_property (object_class, PROP_SET_TIMESTAMP,
      g_param_spec_boolean ("set-timestamp", "Set timestamp",
          "The flag to set timestamp when received a buffer with invalid timestamp",
          DEFAULT_SET_TIMESTAMP, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorConverter::silent:
   *
   * The flag to enable/disable debugging messages.
   */
  g_object_class_install_property (object_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata (element_class,
      "TensorConverter",
      "Converter/Tensor",
      "Converts audio or video stream to tensor stream of C-Array for neural network framework filters",
      "MyungJoo Ham <myungjoo.ham@samsung.com>");

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&src_template));
  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&sink_template));

  element_class->change_state = gst_tensor_converter_change_state;
}

/**
 * @brief Initialize tensor_converter element.
 */
static void
gst_tensor_converter_init (GstTensorConverter * self)
{
  /** setup sink pad */
  self->sinkpad = gst_pad_new_from_static_template (&sink_template, "sink");
  gst_pad_set_event_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_converter_sink_event));
  gst_pad_set_query_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_converter_sink_query));
  gst_pad_set_chain_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_converter_chain));
  GST_PAD_SET_PROXY_CAPS (self->sinkpad);
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad);

  /** setup src pad */
  self->srcpad = gst_pad_new_from_static_template (&src_template, "src");
  gst_pad_set_query_function (self->srcpad,
      GST_DEBUG_FUNCPTR (gst_tensor_converter_src_query));
  GST_PAD_SET_PROXY_CAPS (self->srcpad);
  gst_element_add_pad (GST_ELEMENT (self), self->srcpad);

  /** init properties */
  self->silent = DEFAULT_SILENT;
  self->set_timestamp = DEFAULT_SET_TIMESTAMP;
  self->frames_per_tensor = DEFAULT_FRAMES_PER_TENSOR;
  self->in_media_type = _NNS_MEDIA_END;
  self->frame_size = 0;
  self->remove_padding = FALSE;
  gst_tensor_info_init (&self->tensor_info);

  self->adapter = gst_adapter_new ();
  gst_tensor_converter_reset (self);
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_tensor_converter_finalize (GObject * object)
{
  GstTensorConverter *self;

  self = GST_TENSOR_CONVERTER (object);

  gst_tensor_converter_reset (self);

  if (self->adapter) {
    g_object_unref (self->adapter);
    self->adapter = NULL;
  }

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_converter properties.
 */
static void
gst_tensor_converter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorConverter *self;

  self = GST_TENSOR_CONVERTER (object);

  switch (prop_id) {
    case PROP_INPUT_DIMENSION:
      if (gst_tensor_parse_dimension (g_value_get_string (value),
              self->tensor_info.dimension) == 0)
        GST_WARNING ("input dimension unknown (optinal).");
      break;
    case PROP_INPUT_TYPE:
      self->tensor_info.type = gst_tensor_get_type (g_value_get_string (value));
      if (self->tensor_info.type == _NNS_END)
        GST_WARNING ("input type unknown (optional).");
      break;
    case PROP_FRAMES_PER_TENSOR:
      self->frames_per_tensor = g_value_get_uint (value);
      silent_debug ("Set frames in output = %d", self->frames_per_tensor);
      break;
    case PROP_SET_TIMESTAMP:
      self->set_timestamp = g_value_get_boolean (value);
      silent_debug ("Set timestamp = %d", self->set_timestamp);
      break;
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      silent_debug ("Set silent = %d", self->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter for tensor_converter properties.
 */
static void
gst_tensor_converter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorConverter *self;

  self = GST_TENSOR_CONVERTER (object);

  switch (prop_id) {
    case PROP_INPUT_DIMENSION:
      if (gst_tensor_dimension_is_valid (self->tensor_info.dimension)) {
        g_value_take_string (value,
            gst_tensor_get_dimension_string (self->tensor_info.dimension));
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_INPUT_TYPE:
      if (self->tensor_info.type != _NNS_END) {
        g_value_set_string (value,
            tensor_element_typename[self->tensor_info.type]);
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_FRAMES_PER_TENSOR:
      g_value_set_uint (value, self->frames_per_tensor);
      break;
    case PROP_SET_TIMESTAMP:
      g_value_set_boolean (value, self->set_timestamp);
      break;
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief This function handles sink event.
 */
static gboolean
gst_tensor_converter_sink_event (GstPad * pad, GstObject * parent,
    GstEvent * event)
{
  GstTensorConverter *self;

  self = GST_TENSOR_CONVERTER (parent);

  GST_DEBUG_OBJECT (self, "Received %s event: %" GST_PTR_FORMAT,
      GST_EVENT_TYPE_NAME (event), event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *in_caps;
      GstCaps *out_caps;

      gst_event_parse_caps (event, &in_caps);
      silent_debug_caps (in_caps, "in-caps");

      if (gst_tensor_converter_parse_caps (self, in_caps)) {
        out_caps = gst_tensor_caps_from_config (&self->tensor_config);
        silent_debug_caps (out_caps, "out-caps");

        gst_pad_set_caps (self->srcpad, out_caps);

        gst_event_unref (event);
        event = gst_event_new_caps (out_caps);

        gst_caps_unref (out_caps);
        return gst_pad_push_event (self->srcpad, event);
      }
      break;
    }
    case GST_EVENT_FLUSH_STOP:
      gst_tensor_converter_reset (self);
      break;
    case GST_EVENT_SEGMENT:
    {
      GstSegment seg;

      gst_event_copy_segment (event, &seg);
      silent_debug ("received seg %s", gst_format_get_name (seg.format));

      self->segment = seg;
      self->have_segment = TRUE;

      if (seg.format == GST_FORMAT_TIME) {
        return gst_pad_push_event (self->srcpad, event);
      }

      if (seg.format == GST_FORMAT_BYTES) {
        /* handle seg event in chain function */
        self->need_segment = TRUE;
        gst_event_unref (event);
        return TRUE;
      }

      GST_ERROR_OBJECT (self, "Unsupported format = %s\n",
          gst_format_get_name (seg.format));
      gst_event_unref (event);
      return FALSE;
    }
    default:
      break;
  }

  return gst_pad_event_default (pad, parent, event);
}

/**
 * @brief This function handles sink pad query.
 */
static gboolean
gst_tensor_converter_sink_query (GstPad * pad, GstObject * parent,
    GstQuery * query)
{
  GstTensorConverter *self;

  self = GST_TENSOR_CONVERTER (parent);

  GST_DEBUG_OBJECT (self, "Received %s query: %" GST_PTR_FORMAT,
      GST_QUERY_TYPE_NAME (query), query);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CAPS:
    {
      GstCaps *caps;
      GstCaps *filter;

      gst_query_parse_caps (query, &filter);
      caps = gst_tensor_converter_query_caps (self, pad, filter);

      gst_query_set_caps_result (query, caps);
      gst_caps_unref (caps);
      return TRUE;
    }
    case GST_QUERY_ACCEPT_CAPS:
    {
      GstCaps *caps;
      GstCaps *template_caps;
      gboolean res = FALSE;

      gst_query_parse_accept_caps (query, &caps);
      silent_debug_caps (caps, "accept-caps");

      if (gst_caps_is_fixed (caps)) {
        template_caps = gst_pad_get_pad_template_caps (pad);

        res = gst_caps_can_intersect (template_caps, caps);
        gst_caps_unref (template_caps);
      }

      gst_query_set_accept_caps_result (query, res);
      return TRUE;
    }
    default:
      break;
  }

  return gst_pad_query_default (pad, parent, query);
}

/**
 * @brief This function handles src pad query.
 */
static gboolean
gst_tensor_converter_src_query (GstPad * pad, GstObject * parent,
    GstQuery * query)
{
  GstTensorConverter *self;

  self = GST_TENSOR_CONVERTER (parent);

  GST_DEBUG_OBJECT (self, "Received %s query: %" GST_PTR_FORMAT,
      GST_QUERY_TYPE_NAME (query), query);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CAPS:
    {
      GstCaps *caps;
      GstCaps *filter;

      gst_query_parse_caps (query, &filter);
      caps = gst_tensor_converter_query_caps (self, pad, filter);

      gst_query_set_caps_result (query, caps);
      gst_caps_unref (caps);
      return TRUE;
    }
    default:
      break;
  }

  return gst_pad_query_default (pad, parent, query);
}

/**
 * @brief Chain function, this function does the actual processing.
 */
static GstFlowReturn
gst_tensor_converter_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  GstTensorConverter *self;
  GstTensorConfig *config;
  GstAdapter *adapter;
  GstBuffer *inbuf;
  gsize avail, buf_size, frame_size, out_size;
  guint frames_in, frames_out;
  GstFlowReturn ret = GST_FLOW_OK;
  GstClockTime pts, dts, duration;
  gboolean have_framerate;

  buf_size = gst_buffer_get_size (buf);
  g_return_val_if_fail (buf_size > 0, GST_FLOW_ERROR);

  self = GST_TENSOR_CONVERTER (parent);

  g_assert (self->tensor_configured);
  config = &self->tensor_config;

  have_framerate = (config->rate_n > 0 && config->rate_d > 0);

  frames_out = self->frames_per_tensor;
  inbuf = buf;

  switch (self->in_media_type) {
    case _NNS_VIDEO:
    {
      guint color, width, height, type;

      color = config->info.dimension[0];
      width = config->info.dimension[1];
      height = config->info.dimension[2];
      type = tensor_element_size[config->info.type];

      /** colorspace * width * height * type */
      frame_size = color * width * height * type;

      /** supposed 1 frame in buffer */
      g_assert ((buf_size / self->frame_size) == 1);
      frames_in = 1;

      if (self->remove_padding) {
        GstMapInfo src_info, dest_info;
        int d0, d1;
        unsigned int src_idx = 0, dest_idx = 0;
        size_t size, offset;

        inbuf = gst_buffer_new_and_alloc (frame_size);
        gst_buffer_memset (inbuf, 0, 0, frame_size);

        g_assert (gst_buffer_map (buf, &src_info, GST_MAP_READ));
        g_assert (gst_buffer_map (inbuf, &dest_info, GST_MAP_WRITE));

        /**
         * Refer: https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html
         */
        size = offset = color * width * type;

        g_assert (offset % 4);
        if (offset % 4) {
          offset += 4 - (offset % 4);
        }

        for (d0 = 0; d0 < frames_in; d0++) {
          for (d1 = 0; d1 < height; d1++) {
            memcpy (dest_info.data + dest_idx, src_info.data + src_idx, size);
            dest_idx += size;
            src_idx += offset;
          }
        }

        gst_buffer_unmap (buf, &src_info);
        gst_buffer_unmap (inbuf, &dest_info);

        /** copy timestamps */
        gst_buffer_copy_into (inbuf, buf, GST_BUFFER_COPY_METADATA, 0, -1);

        gst_buffer_unref (buf);
      }
      break;
    }
#ifndef NO_AUDIO
    case _NNS_AUDIO:
      /* number of bytes for one frame */
      frame_size = self->frame_size;
      frames_in = buf_size / frame_size;
      break;
#endif
    case _NNS_TEXT:
      /* supposed 1 frame in buffer */
      frame_size = self->frame_size;
      frames_in = 1;

      if (buf_size != frame_size) {
        GstMapInfo src_info, dest_info;

        inbuf = gst_buffer_new_and_alloc (frame_size);
        gst_buffer_memset (inbuf, 0, 0, frame_size);

        g_assert (gst_buffer_map (buf, &src_info, GST_MAP_READ));
        g_assert (gst_buffer_map (inbuf, &dest_info, GST_MAP_WRITE));

        strncpy ((char *) dest_info.data, (char *) src_info.data, frame_size);

        gst_buffer_unmap (buf, &src_info);
        gst_buffer_unmap (inbuf, &dest_info);

        /** copy timestamps */
        gst_buffer_copy_into (inbuf, buf, GST_BUFFER_COPY_METADATA, 0, -1);

        gst_buffer_unref (buf);
      }
      break;
    case _NNS_OCTET:
      /** get frame size from the properties */
      frame_size = self->frame_size;
      g_assert (frame_size > 0);
      g_assert ((buf_size % frame_size) == 0);
      frames_in = buf_size / frame_size;
      break;
    default:
      GST_ERROR_OBJECT (self, "Unsupported type %d\n", self->in_media_type);
      g_assert (0);
      return GST_FLOW_ERROR;
  }

  /* convert format (bytes > time) and push segment event */
  if (self->need_segment) {
    GstSegment seg;
    guint64 start;

    g_assert (self->have_segment);
    start = self->segment.start;

    gst_segment_init (&seg, GST_FORMAT_TIME);

    if (have_framerate) {
      if (start > 0) {
        start = gst_util_uint64_scale_int (start * config->rate_d, GST_SECOND,
            frame_size * config->rate_n);
        seg.start = seg.time = start;
      }
    }

    self->segment = seg;
    self->need_segment = FALSE;

    gst_pad_push_event (self->srcpad, gst_event_new_segment (&seg));
  }

  if (self->set_timestamp) {
    /* set duration */
    duration = GST_BUFFER_DURATION (inbuf);

    if (!GST_CLOCK_TIME_IS_VALID (duration)) {
      if (have_framerate) {
        duration =
            gst_util_uint64_scale_int (frames_in * config->rate_d, GST_SECOND,
            config->rate_n);

        GST_BUFFER_DURATION (inbuf) = duration;
      }
    }

    /* set timestamp if buffer has invalid timestamp */
    pts = GST_BUFFER_TIMESTAMP (inbuf);

    if (!GST_CLOCK_TIME_IS_VALID (pts)) {
      pts = self->segment.start;

      if (have_framerate) {
        if (GST_CLOCK_TIME_IS_VALID (self->old_timestamp)) {
          pts = self->old_timestamp + duration;
        }
      } else {
        GstClock *clock;

        clock = gst_element_get_clock (GST_ELEMENT (self));

        if (clock) {
          GstClockTime now, base;

          base = gst_element_get_base_time (GST_ELEMENT (self));
          now = gst_clock_get_time (clock);

          pts = (base < now) ? (now - base) : 0;
          gst_object_unref (clock);
        }
      }

      GST_BUFFER_TIMESTAMP (inbuf) = pts;
    }
  }

  /* update old timestamp */
  self->old_timestamp = GST_BUFFER_TIMESTAMP (inbuf);

  if (frames_in == frames_out) {
    silent_debug_timestamp (inbuf);

    /** do nothing, push the incoming buffer */
    return gst_pad_push (self->srcpad, inbuf);
  }

  adapter = self->adapter;
  g_assert (adapter != NULL);

  duration = GST_BUFFER_DURATION (inbuf);
  if (GST_CLOCK_TIME_IS_VALID (duration)) {
    /** supposed same duration for incoming buffer */
    duration = gst_util_uint64_scale_int (duration, frames_out, frames_in);
  }

  gst_adapter_push (adapter, inbuf);

  out_size = frames_out * frame_size;
  while ((avail = gst_adapter_available (adapter)) >= out_size &&
      ret == GST_FLOW_OK) {
    GstBuffer *outbuf;
    guint64 pts_dist, dts_dist;

    pts = gst_adapter_prev_pts (adapter, &pts_dist);
    dts = gst_adapter_prev_dts (adapter, &dts_dist);

    /**
     * Update timestamp.
     * If frames-in is larger then frames-out, the same timestamp (pts and dts) would be returned.
     */
    if (frames_in > 1 && have_framerate) {
      if (GST_CLOCK_TIME_IS_VALID (pts)) {
        pts +=
            gst_util_uint64_scale_int (pts_dist * config->rate_d, GST_SECOND,
            config->rate_n * frame_size);
      }

      if (GST_CLOCK_TIME_IS_VALID (dts)) {
        dts +=
            gst_util_uint64_scale_int (dts_dist * config->rate_d, GST_SECOND,
            config->rate_n * frame_size);
      }
    }

    outbuf = gst_adapter_take_buffer (adapter, out_size);
    outbuf = gst_buffer_make_writable (outbuf);

    /** set timestamp */
    GST_BUFFER_PTS (outbuf) = pts;
    GST_BUFFER_DTS (outbuf) = dts;
    GST_BUFFER_DURATION (outbuf) = duration;

    silent_debug_timestamp (outbuf);

    ret = gst_pad_push (self->srcpad, outbuf);
  }

  return ret;
}

/**
 * @brief Called to perform state change.
 */
static GstStateChangeReturn
gst_tensor_converter_change_state (GstElement * element,
    GstStateChange transition)
{
  GstTensorConverter *self;
  GstStateChangeReturn ret;

  self = GST_TENSOR_CONVERTER (element);

  switch (transition) {
    case GST_STATE_CHANGE_READY_TO_PAUSED:
      gst_tensor_converter_reset (self);
      break;
    default:
      break;
  }

  ret = GST_ELEMENT_CLASS (parent_class)->change_state (element, transition);

  switch (transition) {
    case GST_STATE_CHANGE_PAUSED_TO_READY:
      gst_tensor_converter_reset (self);
      break;
    default:
      break;
  }

  return ret;
}

/**
 * @brief Clear and reset data.
 */
static void
gst_tensor_converter_reset (GstTensorConverter * self)
{
  if (self->adapter) {
    gst_adapter_clear (self->adapter);
  }

  self->tensor_configured = FALSE;
  gst_tensor_config_init (&self->tensor_config);

  self->have_segment = FALSE;
  self->need_segment = FALSE;
  gst_segment_init (&self->segment, GST_FORMAT_TIME);

  self->old_timestamp = GST_CLOCK_TIME_NONE;
}

/**
 * @brief Determine if we need zero-padding
 * @return TRUE if we need to add (or remove) stride per row from the stream data.
 */
static gboolean
gst_tensor_converter_video_stride (GstVideoFormat format, gint width)
{
  /**
   * @todo The actual list is much longer, fill them.
   * (read https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html)
   */
  switch (format) {
    case GST_VIDEO_FORMAT_GRAY8:
    case GST_VIDEO_FORMAT_RGB:
    case GST_VIDEO_FORMAT_BGR:
    case GST_VIDEO_FORMAT_I420:
      if (width % 4) {
        return TRUE;
      }
      break;
    default:
      break;
  }

  return FALSE;
}

/**
 * @brief Set the tensor config structure from video info (internal static function)
 * @param self this pointer to GstTensorConverter
 * @param config tensor config structure to be filled
 * @param info video info structure
 * @note Change dimention if tensor contains N frames.
 * @return TRUE if supported type
 */
static gboolean
gst_tensor_converter_parse_video (GstTensorConverter * self,
    GstTensorConfig * config, const GstVideoInfo * info)
{
  /**
   * Refer: https://www.tensorflow.org/api_docs/python/tf/summary/image
   * A 4-D uint8 or float32 Tensor of shape [batch_size, height, width, channels]
   * where channels is 1, 3, or 4.
   */
  GstVideoFormat format;
  guint i;

  g_return_val_if_fail (config != NULL, FALSE);
  gst_tensor_config_init (config);

  g_return_val_if_fail (info != NULL, FALSE);

  format = GST_VIDEO_INFO_FORMAT (info);

  /* [color-space][width][height][frames] */
  switch (format) {
    case GST_VIDEO_FORMAT_GRAY8:
      config->info.type = _NNS_UINT8;
      config->info.dimension[0] = 1;
      break;
    case GST_VIDEO_FORMAT_RGB:
    case GST_VIDEO_FORMAT_BGR:
      config->info.type = _NNS_UINT8;
      config->info.dimension[0] = 3;
      break;
    case GST_VIDEO_FORMAT_RGBx:
    case GST_VIDEO_FORMAT_BGRx:
    case GST_VIDEO_FORMAT_xRGB:
    case GST_VIDEO_FORMAT_xBGR:
    case GST_VIDEO_FORMAT_RGBA:
    case GST_VIDEO_FORMAT_BGRA:
    case GST_VIDEO_FORMAT_ARGB:
    case GST_VIDEO_FORMAT_ABGR:
      config->info.type = _NNS_UINT8;
      config->info.dimension[0] = 4;
      break;
    default:
      /* unsupported format */
      GST_WARNING_OBJECT (self, "Unsupported format = %s\n",
          GST_STR_NULL (gst_video_format_to_string (format)));
      break;
  }

  config->info.dimension[1] = GST_VIDEO_INFO_WIDTH (info);
  config->info.dimension[2] = GST_VIDEO_INFO_HEIGHT (info);

  /* Supposed 1 frame in tensor, change dimension[3] if tensor contains N frames. */
  for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++) {
    config->info.dimension[i] = 1;
  }

  config->rate_n = GST_VIDEO_INFO_FPS_N (info);
  config->rate_d = GST_VIDEO_INFO_FPS_D (info);

  return (config->info.type != _NNS_END);
}

#ifndef NO_AUDIO
/**
 * @brief Set the tensor config structure from audio info (internal static function)
 * @param self this pointer to GstTensorConverter
 * @param config tensor config structure to be filled
 * @param info audio info structure
 * @note Change dimention if tensor contains N frames.
 * @return TRUE if supported type
 */
static gboolean
gst_tensor_converter_parse_audio (GstTensorConverter * self,
    GstTensorConfig * config, const GstAudioInfo * info)
{
  /**
   * Refer: https://www.tensorflow.org/api_docs/python/tf/summary/audio
   * A 3-D float32 Tensor of shape [batch_size, frames, channels]
   * or a 2-D float32 Tensor of shape [batch_size, frames].
   */
  GstAudioFormat format;
  guint i;

  g_return_val_if_fail (config != NULL, FALSE);
  gst_tensor_config_init (config);

  g_return_val_if_fail (info != NULL, FALSE);

  format = GST_AUDIO_INFO_FORMAT (info);

  /* [channels][frames] */
  switch (format) {
    case GST_AUDIO_FORMAT_S8:
      config->info.type = _NNS_INT8;
      break;
    case GST_AUDIO_FORMAT_U8:
      config->info.type = _NNS_UINT8;
      break;
    case GST_AUDIO_FORMAT_S16:
      config->info.type = _NNS_INT16;
      break;
    case GST_AUDIO_FORMAT_U16:
      config->info.type = _NNS_UINT16;
      break;
    case GST_AUDIO_FORMAT_S32:
      config->info.type = _NNS_INT32;
      break;
    case GST_AUDIO_FORMAT_U32:
      config->info.type = _NNS_UINT32;
      break;
    case GST_AUDIO_FORMAT_F32:
      config->info.type = _NNS_FLOAT32;
      break;
    case GST_AUDIO_FORMAT_F64:
      config->info.type = _NNS_FLOAT64;
      break;
    default:
      /* unsupported format */
      GST_WARNING_OBJECT (self, "Unsupported format = %s\n",
          GST_STR_NULL (gst_audio_format_to_string (format)));
      break;
  }

  config->info.dimension[0] = GST_AUDIO_INFO_CHANNELS (info);

  /* Supposed 1 frame in tensor, change dimension[1] if tensor contains N frames. */
  for (i = 1; i < NNS_TENSOR_RANK_LIMIT; i++) {
    config->info.dimension[i] = 1;
  }

  config->rate_n = GST_AUDIO_INFO_RATE (info);
  config->rate_d = 1;

  return (config->info.type != _NNS_END);
}
#endif

/**
 * @brief Set the tensor config structure from text info (internal static function)
 * @param self this pointer to GstTensorConverter
 * @param config tensor config structure to be filled
 * @param structure caps structure
 * @note Change dimention if tensor contains N frames.
 * @return TRUE if supported type
 */
static gboolean
gst_tensor_converter_parse_text (GstTensorConverter * self,
    GstTensorConfig * config, const GstStructure * structure)
{
  /**
   * Refer: https://www.tensorflow.org/api_docs/python/tf/summary/text
   * A string-type Tensor
   */
  const gchar *format_string;
  guint i;

  g_return_val_if_fail (config != NULL, FALSE);
  gst_tensor_config_init (config);

  g_return_val_if_fail (structure != NULL, FALSE);

  format_string = gst_structure_get_string (structure, "format");
  if (format_string) {
    if (g_str_equal (format_string, "utf8")) {
      config->info.type = _NNS_UINT8;
    } else {
      /* unsupported format */
      GST_WARNING_OBJECT (self, "Unsupported format = %s\n", format_string);
    }
  }

  /* [size][frames] */
  /* Fixed size of string, we cannot get the size from caps. */
  config->info.dimension[0] = 0;

  /* Supposed 1 frame in tensor, change dimension[1] if tensor contains N frames. */
  for (i = 1; i < NNS_TENSOR_RANK_LIMIT; i++) {
    config->info.dimension[i] = 1;
  }

  if (gst_structure_has_field (structure, "framerate")) {
    gst_structure_get_fraction (structure, "framerate", &config->rate_n,
        &config->rate_d);
  } else {
    /* cannot get the framerate for text type */
    config->rate_n = 0;
    config->rate_d = 1;
  }

  return (config->info.type != _NNS_END);
}

/**
 * @brief Set the tensor config structure from octet stream (internal static function)
 * @param self this pointer to GstTensorConverter
 * @param config tensor config structure to be filled
 * @param structure caps structure
 * @note Change tensor dimention and type.
 * @return TRUE if supported type
 */
static gboolean
gst_tensor_converter_parse_octet (GstTensorConverter * self,
    GstTensorConfig * config, const GstStructure * structure)
{
  g_return_val_if_fail (config != NULL, FALSE);
  gst_tensor_config_init (config);

  g_return_val_if_fail (structure != NULL, FALSE);

  /**
   * Raw byte-stream (application/octet-stream)
   * We cannot get the exact tensor info from caps.
   * All tensor info should be updated.
   */
  config->info.type = _NNS_UINT8;

  if (gst_structure_has_field (structure, "framerate")) {
    gst_structure_get_fraction (structure, "framerate", &config->rate_n,
        &config->rate_d);
  } else {
    /* cannot get the framerate */
    config->rate_n = 0;
    config->rate_d = 1;
  }

  return (config->info.type != _NNS_END);
}

/**
 * @brief Get supported format list.
 */
static void
gst_tensor_converter_get_format_list (GValue * list, ...)
{
  GValue item = G_VALUE_INIT;
  gchar *str;
  va_list args;

  g_value_init (list, GST_TYPE_LIST);

  va_start (args, list);
  while ((str = va_arg (args, gchar *))) {
    g_value_init (&item, G_TYPE_STRING);
    g_value_set_string (&item, str);

    gst_value_list_append_value (list, &item);
    g_value_unset (&item);
  }
  va_end (args);
}

/**
 * @brief Get pad caps for caps negotiation.
 */
static GstCaps *
gst_tensor_converter_query_caps (GstTensorConverter * self, GstPad * pad,
    GstCaps * filter)
{
  GstCaps *caps;

  caps = gst_pad_get_current_caps (pad);
  if (!caps) {
    caps = gst_pad_get_pad_template_caps (pad);
  }

  if (pad == self->sinkpad) {
    GstCaps *peer_caps;

    /* get possible caps from downstream element */
    peer_caps = gst_pad_peer_query_caps (self->srcpad, NULL);

    if (peer_caps) {
      silent_debug_caps (peer_caps, "peer caps");

      if (gst_caps_get_size (peer_caps) > 0) {
        GstTensorConfig config;
        GstStructure *st;
        GstCaps *media_caps, *tmp;
        guint i, caps_len;
        media_type type;

        /* get tensor info from peer caps */
        st = gst_caps_get_structure (peer_caps, 0);
        gst_tensor_config_from_structure (&config, st);

        /* convert peer caps to possible media caps */
        media_caps = gst_caps_from_string (GST_TENSOR_MEDIA_CAPS_STR);
        media_caps = gst_caps_make_writable (media_caps);
        caps_len = gst_caps_get_size (media_caps);

        for (i = 0; i < caps_len; ++i) {
          st = gst_caps_get_structure (media_caps, i);
          type = gst_tensor_media_type_from_structure (st);

          switch (type) {
            case _NNS_VIDEO:
              /* video caps from tensor info */
              if (config.info.type == _NNS_UINT8) {
                GValue supported_formats = G_VALUE_INIT;
                gint colorspace, width, height;

                colorspace = config.info.dimension[0];
                switch (colorspace) {
                  case 1:
                    gst_tensor_converter_get_format_list (&supported_formats,
                        "GRAY8", NULL);
                    break;
                  case 3:
                    gst_tensor_converter_get_format_list (&supported_formats,
                        "RGB", "BGR", NULL);
                    break;
                  case 4:
                    gst_tensor_converter_get_format_list (&supported_formats,
                        "RGBx", "BGRx", "xRGB", "xBGR", "RGBA", "BGRA", "ARGB",
                        "ABGR", NULL);
                    break;
                  default:
                    /* unsupported format, set default video formats */
                    break;
                }

                if (gst_value_list_get_size (&supported_formats) > 0) {
                  gst_structure_set_value (st, "format", &supported_formats);
                }
                g_value_unset (&supported_formats);

                if ((width = config.info.dimension[1]) > 0) {
                  gst_structure_set (st, "width", G_TYPE_INT, width, NULL);
                }

                if ((height = config.info.dimension[2]) > 0) {
                  gst_structure_set (st, "height", G_TYPE_INT, height, NULL);
                }

                if (config.rate_n >= 0 && config.rate_d > 0) {
                  gst_structure_set (st, "framerate", GST_TYPE_FRACTION,
                      config.rate_n, config.rate_d, NULL);
                }
              }
              break;
#ifndef NO_AUDIO
            case _NNS_AUDIO:
              /* audio caps from tensor info */
              if (config.info.type != _NNS_END) {
                gint channels, samplerate;
                GstAudioFormat aformat;

                switch (config.info.type) {
                  case _NNS_INT8:
                    aformat = GST_AUDIO_FORMAT_S8;
                    break;
                  case _NNS_UINT8:
                    aformat = GST_AUDIO_FORMAT_U8;
                    break;
                  case _NNS_INT16:
                    aformat = GST_AUDIO_FORMAT_S16;
                    break;
                  case _NNS_UINT16:
                    aformat = GST_AUDIO_FORMAT_U16;
                    break;
                  case _NNS_INT32:
                    aformat = GST_AUDIO_FORMAT_S32;
                    break;
                  case _NNS_UINT32:
                    aformat = GST_AUDIO_FORMAT_U32;
                    break;
                  case _NNS_FLOAT32:
                    aformat = GST_AUDIO_FORMAT_F32;
                    break;
                  case _NNS_FLOAT64:
                    aformat = GST_AUDIO_FORMAT_F64;
                    break;
                  default:
                    /* unsupported format */
                    aformat = GST_AUDIO_FORMAT_UNKNOWN;
                    break;
                }

                if (aformat != GST_AUDIO_FORMAT_UNKNOWN) {
                  gst_structure_set (st, "format", G_TYPE_STRING,
                      gst_audio_format_to_string (aformat), NULL);

                  if ((channels = config.info.dimension[0]) > 0) {
                    gst_structure_set (st, "channels", G_TYPE_INT, channels,
                        NULL);
                  }

                  if ((samplerate = config.rate_n) > 0) {
                    gst_structure_set (st, "rate", G_TYPE_INT, samplerate,
                        NULL);
                  }
                }
              }
              break;
#endif
            default:
              /* do nothing for text and octet stream */
              break;
          }
        }

        /* intersect with pad caps */
        tmp = gst_caps_intersect_full (media_caps, caps,
            GST_CAPS_INTERSECT_FIRST);
        gst_caps_unref (caps);
        caps = tmp;

        gst_caps_unref (media_caps);
      }

      gst_caps_unref (peer_caps);
    }
  }

  silent_debug_caps (caps, "caps");
  silent_debug_caps (filter, "filter");

  if (filter) {
    GstCaps *intersection;

    intersection =
        gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (caps);
    caps = intersection;
  }

  silent_debug_caps (caps, "result");
  return caps;
}

/**
 * @brief Parse caps and set tensor info.
 */
static gboolean
gst_tensor_converter_parse_caps (GstTensorConverter * self,
    const GstCaps * caps)
{
  GstStructure *structure;
  GstTensorConfig config;
  media_type in_type;
  gint frames_dim = -1; /** dimension index of frames in configured tensor */

  g_return_val_if_fail (caps != NULL, FALSE);
  g_return_val_if_fail (gst_caps_is_fixed (caps), FALSE);

  structure = gst_caps_get_structure (caps, 0);
  in_type = gst_tensor_media_type_from_structure (structure);

  switch (in_type) {
    case _NNS_VIDEO:
    {
      GstVideoInfo info;

      gst_video_info_init (&info);
      if (!gst_video_info_from_caps (&info, caps)) {
        GST_ERROR_OBJECT (self, "Failed to get video info from caps.\n");
        return FALSE;
      }

      if (!gst_tensor_converter_parse_video (self, &config, &info)) {
        GST_ERROR_OBJECT (self, "Failed to configure tensor from video info.");
        return FALSE;
      }

      /**
       * Emit Warning if RSTRIDE = RU4 (3BPP) && Width % 4 > 0
       * @todo Add more conditions!
       */
      if (gst_tensor_converter_video_stride (GST_VIDEO_INFO_FORMAT (&info),
              GST_VIDEO_INFO_WIDTH (&info))) {
        self->remove_padding = TRUE;
        silent_debug ("Set flag to remove padding, width = %d",
            GST_VIDEO_INFO_WIDTH (&info));

        GST_WARNING_OBJECT (self,
            "\nYOUR STREAM CONFIGURATION INCURS PERFORMANCE DETERIORATION!\n"
            "Please use 4 x n as image width for inputs.\n");
      }

      frames_dim = 3;
      self->frame_size = GST_VIDEO_INFO_SIZE (&info);
      break;
    }
#ifndef NO_AUDIO
    case _NNS_AUDIO:
    {
      GstAudioInfo info;

      gst_audio_info_init (&info);
      if (!gst_audio_info_from_caps (&info, caps)) {
        GST_ERROR_OBJECT (self, "Failed to get audio info from caps.\n");
        return FALSE;
      }

      if (!gst_tensor_converter_parse_audio (self, &config, &info)) {
        GST_ERROR_OBJECT (self, "Failed to configure tensor from audio info.");
        return FALSE;
      }

      frames_dim = 1;
      self->frame_size = GST_AUDIO_INFO_BPF (&info);
      break;
    }
#endif
    case _NNS_TEXT:
    {
      if (!gst_tensor_converter_parse_text (self, &config, structure)) {
        GST_ERROR_OBJECT (self, "Failed to configure tensor from text info.");
        return FALSE;
      }

      /* get fixed size of text string from property */
      if (!gst_tensor_dimension_is_valid (self->tensor_info.dimension)) {
        GST_ERROR_OBJECT (self,
            "Failed to get tensor info, need to update string size.");

        g_critical ("Please set the property input-dim to convert stream.\n"
            "For example, input-dim=30 to handle up to 30 bytes of string per frame.");
        return FALSE;
      }

      config.info.dimension[0] = self->tensor_info.dimension[0];
      frames_dim = 1;
      self->frame_size = gst_tensor_info_get_size (&config.info);
      break;
    }
    case _NNS_OCTET:
    {
      if (!gst_tensor_converter_parse_octet (self, &config, structure)) {
        GST_ERROR_OBJECT (self, "Failed to configure tensor from octet info.");
        return FALSE;
      }

      /* update tensor info from properties */
      if (!gst_tensor_info_validate (&self->tensor_info)) {
        GST_ERROR_OBJECT (self,
            "Failed to get tensor info, need to update dimension and type.");

        g_critical
            ("Please set the properties input-dim and input-type to convert stream.\n"
            "For example, input-dim=30:1 input-type=unit8 to handle 30 bytes of bin data.");
        return FALSE;
      }

      config.info = self->tensor_info;
      self->frame_size = gst_tensor_info_get_size (&config.info);
      break;
    }
    default:
      GST_ERROR_OBJECT (self, "Unsupported type %d\n", in_type);
      return FALSE;
  }

  /** set the number of frames in dimension */
  if (frames_dim >= 0) {
    config.info.dimension[frames_dim] = self->frames_per_tensor;
  }

  if (!gst_tensor_config_validate (&config)) {
    /** not fully configured */
    GST_ERROR_OBJECT (self, "Failed to configure tensor info.\n");
    return FALSE;
  }

  if (gst_tensor_info_validate (&self->tensor_info)) {
    /** compare tensor info */
    if (!gst_tensor_info_is_equal (&self->tensor_info, &config.info)) {
      GST_ERROR_OBJECT (self, "Failed, mismatched tensor info.\n");
      return FALSE;
    }
  }

  self->in_media_type = in_type;
  self->tensor_configured = TRUE;
  self->tensor_config = config;
  return TRUE;
}

/**
 * @brief Function to initialize the plugin.
 *
 * See GstPluginInitFunc() for more details.
 */
NNSTREAMER_PLUGIN_INIT (tensor_converter)
{
  GST_DEBUG_CATEGORY_INIT (gst_tensor_converter_debug, "tensor_converter",
      0, "tensor_converter element");

  return gst_element_register (plugin, "tensor_converter",
      GST_RANK_NONE, GST_TYPE_TENSOR_CONVERTER);
}

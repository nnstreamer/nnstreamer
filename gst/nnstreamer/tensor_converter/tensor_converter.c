/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	tensor_converter.c
 * @date	26 Mar 2018
 * @brief	GStreamer plugin to convert media types to tensors (as a filter for other general neural network filters)
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 * @todo        For flatbuffers, support other/tensors with properties
 * @todo        Subplugins are not tested, yet.
 */

/**
 * SECTION:element-tensor_converter
 *
 * A filter that converts media stream to tensor stream for NN frameworks.
 * The output is always in the format of other/tensor or other/tensors.
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

#ifdef NO_VIDEO
#include "converter-media-info-no-video.h"
#else
#include "converter-media-info-video.h"
#endif

#ifdef NO_AUDIO
#include "converter-media-info-no-audio.h"
#else
#include "converter-media-info-audio.h"
#endif
#include <nnstreamer_log.h>
#include <nnstreamer_subplugin.h>

/**
 * @brief Caps string for text input
 */
#define TEXT_CAPS_STR "text/x-raw, format = (string) utf8"

#define append_text_caps_template(caps) \
    gst_caps_append (caps, gst_caps_from_string (TEXT_CAPS_STR))

/**
 * @brief Caps string for binary stream
 */
#define OCTET_CAPS_STR "application/octet-stream"

#define append_octet_caps_template(caps) \
    gst_caps_append (caps, gst_caps_from_string (OCTET_CAPS_STR))

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
 * @todo For flatbuffers, support other/tensors.
 */
enum
{
  PROP_0,
  PROP_INPUT_DIMENSION,
  PROP_INPUT_TYPE,
  PROP_FRAMES_PER_TENSOR,
  PROP_SET_TIMESTAMP,
  PROP_SUBPLUGINS,
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
static gboolean gst_tensor_converter_update_caps (GstTensorConverter * self,
    GstPad * pad, GstTensorsConfig * config);
static const NNStreamerExternalConverter *findExternalConverter (const char
    *media_type_name);

/**
 * @brief Initialize the tensor_converter's class.
 */
static void
gst_tensor_converter_class_init (GstTensorConverterClass * klass)
{
  GObjectClass *object_class;
  GstElementClass *element_class;
  GstPadTemplate *pad_template;
  GstCaps *pad_caps;
  guint total, i;
  const NNStreamerExternalConverter *ex;
  subplugin_info_s info;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_converter_debug, "tensor_converter", 0,
      "Element to convert media stream to tensor stream");

  object_class = (GObjectClass *) klass;
  element_class = (GstElementClass *) klass;

  /* GObjectClass vmethods */
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
   * The number of frames in outgoing buffer. (buffer is a single tensor instance)
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
   * GstTensorConverter::sub-plugins:
   *
   * Registrable sub-plugins list of tensor-converter.
   */
  g_object_class_install_property (object_class, PROP_SUBPLUGINS,
      g_param_spec_string ("sub-plugins", "Sub-plugins",
          "Registrable sub-plugins list", "",
          G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  /**
   * GstTensorConverter::silent:
   *
   * The flag to enable/disable debugging messages.
   */
  g_object_class_install_property (object_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /* set src pad template */
  pad_caps =
      gst_caps_from_string (GST_TENSOR_CAP_DEFAULT "; "
      GST_TENSORS_CAP_DEFAULT);

  pad_template = gst_pad_template_new ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
      pad_caps);
  gst_element_class_add_pad_template (element_class, pad_template);

  gst_caps_unref (pad_caps);

  /* set sink pad template */
  pad_caps = gst_caps_new_empty ();

  /* append caps string for all media types */
  append_video_caps_template (pad_caps);
  append_audio_caps_template (pad_caps);
  append_text_caps_template (pad_caps);
  append_octet_caps_template (pad_caps);

  /* append sub-plugin template caps */
  total = nnsconf_get_subplugin_info (NNSCONF_PATH_CONVERTERS, &info);
  for (i = 0; i < total; i++) {
    ex = get_subplugin (NNS_SUBPLUGIN_CONVERTER, info.names[i]);
    if (ex && ex->query_caps)
      gst_caps_append (pad_caps, ex->query_caps (NULL));
  }

  pad_template = gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
      pad_caps);
  gst_element_class_add_pad_template (element_class, pad_template);

  gst_caps_unref (pad_caps);

  gst_element_class_set_static_metadata (element_class,
      "TensorConverter",
      "Converter/Tensor",
      "Converts audio or video stream to tensor stream of C-Array for neural network framework filters",
      "MyungJoo Ham <myungjoo.ham@samsung.com>");

  /* GstElementClass vmethods */
  element_class->change_state = gst_tensor_converter_change_state;
}

/**
 * @brief Initialize tensor_converter element.
 */
static void
gst_tensor_converter_init (GstTensorConverter * self)
{
  /** setup sink pad */
  self->sinkpad =
      gst_pad_new_from_template (gst_element_class_get_pad_template
      (GST_ELEMENT_GET_CLASS (self), "sink"), "sink");
  gst_pad_set_event_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_converter_sink_event));
  gst_pad_set_query_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_converter_sink_query));
  gst_pad_set_chain_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_converter_chain));
  GST_PAD_SET_PROXY_CAPS (self->sinkpad);
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad);

  /** setup src pad */
  self->srcpad =
      gst_pad_new_from_template (gst_element_class_get_pad_template
      (GST_ELEMENT_GET_CLASS (self), "src"), "src");
  gst_pad_set_query_function (self->srcpad,
      GST_DEBUG_FUNCPTR (gst_tensor_converter_src_query));
  GST_PAD_SET_PROXY_CAPS (self->srcpad);
  gst_element_add_pad (GST_ELEMENT (self), self->srcpad);

  /** init properties */
  self->silent = DEFAULT_SILENT;
  self->set_timestamp = DEFAULT_SET_TIMESTAMP;
  self->frames_per_tensor = DEFAULT_FRAMES_PER_TENSOR;
  self->in_media_type = _NNS_MEDIA_INVALID;
  self->frame_size = 0;
  self->remove_padding = FALSE;
  self->externalConverter = NULL;
  gst_tensors_info_init (&self->tensors_info);

  self->adapter = gst_adapter_new ();
  g_assert (self->adapter != NULL);
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
  GstTensorsInfo *info;
  guint i, j, num;
  const gchar *value_str;

  self = GST_TENSOR_CONVERTER (object);
  info = &self->tensors_info;

  switch (prop_id) {
    case PROP_INPUT_DIMENSION:
      value_str = g_value_get_string (value);
      num = gst_tensors_info_parse_dimensions_string (info, value_str);

      if (num == 0) {
        GST_WARNING ("%s is invalid dimension string.", value_str);
      } else if (info->num_tensors > 0 && info->num_tensors != num) {
        GST_WARNING ("%s, the number of tensor is %u.", value_str, num);
      }

      /* prevent invalid value, init dimensions. */
      for (i = num; i < NNS_TENSOR_SIZE_LIMIT; ++i) {
        for (j = 0; j < NNS_TENSOR_RANK_LIMIT; ++j)
          info->info[i].dimension[j] = 0;
      }

      info->num_tensors = num;
      break;
    case PROP_INPUT_TYPE:
      value_str = g_value_get_string (value);
      num = gst_tensors_info_parse_types_string (info, value_str);

      if (num == 0) {
        GST_WARNING ("%s is invalid type string.", value_str);
      } else if (info->num_tensors > 0 && info->num_tensors != num) {
        GST_WARNING ("%s, the number of tensor is %u.", value_str, num);
      }

      /* prevent invalid value, init types. */
      for (i = num; i < NNS_TENSOR_SIZE_LIMIT; ++i) {
        info->info[i].type = _NNS_END;
      }

      info->num_tensors = num;
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
  GstTensorsInfo *info;

  self = GST_TENSOR_CONVERTER (object);
  info = &self->tensors_info;

  switch (prop_id) {
    case PROP_INPUT_DIMENSION:
      if (info->num_tensors > 0) {
        g_value_take_string (value,
            gst_tensors_info_get_dimensions_string (info));
      } else {
        g_value_set_string (value, "");
      }
      break;
    case PROP_INPUT_TYPE:
      if (info->num_tensors > 0) {
        g_value_take_string (value, gst_tensors_info_get_types_string (info));
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
    case PROP_SUBPLUGINS:
    {
      subplugin_info_s sinfo;

      nnsconf_get_subplugin_info (NNSCONF_PATH_CONVERTERS, &sinfo);
      g_value_take_string (value, g_strjoinv (",", sinfo.names));
      break;
    }
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

      gst_event_parse_caps (event, &in_caps);
      silent_debug_caps (in_caps, "in-caps");

      if (gst_tensor_converter_parse_caps (self, in_caps)) {
        gst_event_unref (event);
        return gst_tensor_converter_update_caps (self, self->srcpad,
            &self->tensors_config);
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

/** @brief Chain function's private routine */
static void
_gst_tensor_converter_chain_segment (GstTensorConverter * self,
    GstTensorsConfig * config, gsize frame_size)
{
  if (self->need_segment) {
    GstSegment seg;
    guint64 start;
    gboolean have_framerate;

    have_framerate = (config->rate_n > 0 && config->rate_d > 0);

    /** This is an internal logic error. */
    g_assert (self->have_segment);
    start = self->segment.start;

    gst_segment_init (&seg, GST_FORMAT_TIME);

    if (have_framerate && start > 0) {
      start = gst_util_uint64_scale_int (start * config->rate_d, GST_SECOND,
          frame_size * config->rate_n);
      seg.start = seg.time = start;
    }

    self->segment = seg;
    self->need_segment = FALSE;

    gst_pad_push_event (self->srcpad, gst_event_new_segment (&seg));
  }
}

/** @brief Chain function's private routine */
static void
_gst_tensor_converter_chain_timestamp (GstTensorConverter * self,
    GstTensorsConfig * config, GstBuffer * inbuf, guint frames_in)
{
  if (self->set_timestamp) {
    GstClockTime pts, duration;
    gboolean have_framerate;

    have_framerate = (config->rate_n > 0 && config->rate_d > 0);

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
}

/** @brief Chain function's private routine to push buffer into src pad */
static GstFlowReturn
_gst_tensor_converter_chain_push (GstTensorConverter * self, GstBuffer * buf)
{
  GstBuffer *buffer = buf;

  if (self->in_media_type == _NNS_OCTET) {
    /* configure multi tensors */
    if (self->tensors_info.num_tensors > 1) {
      GstMemory *mem;
      GstMapInfo info;
      gpointer data;
      gsize idx, size;
      guint i;

      g_assert (self->frames_per_tensor == 1);

      buffer = gst_buffer_new ();

      mem = gst_buffer_get_all_memory (buf);
      gst_memory_map (mem, &info, GST_MAP_READ);

      idx = 0;
      for (i = 0; i < self->tensors_info.num_tensors; ++i) {
        size = gst_tensors_info_get_size (&self->tensors_info, i);
        data = g_memdup (info.data + idx, size);
        idx += size;

        gst_buffer_append_memory (buffer,
            gst_memory_new_wrapped (0, data, size, 0, size, data, g_free));
      }

      gst_memory_unmap (mem, &info);
      gst_memory_unref (mem);

      /* copy timestamps */
      gst_buffer_copy_into (buffer, buf, GST_BUFFER_COPY_METADATA, 0, -1);

      gst_buffer_unref (buf);
    }
  }

  return gst_pad_push (self->srcpad, buffer);
}

/** @brief Chain function's private routine to push multiple buffers */
static GstFlowReturn
_gst_tensor_converter_chain_chunk (GstTensorConverter * self,
    GstTensorsConfig * config, GstBuffer * inbuf,
    guint frames_in, guint frames_out, gsize frame_size)
{
  GstAdapter *adapter;
  GstFlowReturn ret = GST_FLOW_OK;
  GstClockTime pts, dts, duration;
  gsize avail, out_size;
  gboolean have_framerate;

  adapter = self->adapter;
  g_assert (adapter != NULL);

  have_framerate = (config->rate_n > 0 && config->rate_d > 0);

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

    ret = _gst_tensor_converter_chain_push (self, outbuf);
  }

  return ret;
}

/**
 * @brief Chain function, this function does the actual processing.
 */
static GstFlowReturn
gst_tensor_converter_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  GstTensorConverter *self;
  GstTensorsConfig *config;
  GstBuffer *inbuf;
  gsize buf_size, frame_size;
  guint frames_in, frames_out;

  buf_size = gst_buffer_get_size (buf);
  g_return_val_if_fail (buf_size > 0, GST_FLOW_ERROR);

  self = GST_TENSOR_CONVERTER (parent);

  /** This is an internal logic error. */
  g_assert (self->tensors_configured);
  config = &self->tensors_config;

  frames_out = self->frames_per_tensor;
  inbuf = buf;

  switch (self->in_media_type) {
    case _NNS_VIDEO:
    {
      guint color, width, height;
      gsize type;

      color = config->info.info[0].dimension[0];
      width = config->info.info[0].dimension[1];
      height = config->info.info[0].dimension[2];
      type = gst_tensor_get_element_size (config->info.info[0].type);

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

        if (FALSE == gst_buffer_map (buf, &src_info, GST_MAP_READ)) {
          ml_logf ("Cannot map src buffer at tensor_converter/video.\n");
          gst_buffer_unref (inbuf);     /* the new buffer is wasted. */
          return GST_FLOW_ERROR;
        }
        if (FALSE == gst_buffer_map (inbuf, &dest_info, GST_MAP_WRITE)) {
          ml_logf ("Cannot map dest buffer at tensor_converter/video.\n");
          gst_buffer_unmap (buf, &src_info);
          gst_buffer_unref (inbuf);     /* the new buffer is wasted. */
          return GST_FLOW_ERROR;
        }

        /**
         * Refer: https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html
         */
        size = offset = color * width * type;

        g_assert (offset % 4); /** Internal logic error! */
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
    case _NNS_AUDIO:
      /* number of bytes for one frame */
      frame_size = self->frame_size;
      frames_in = buf_size / frame_size;
      break;
    case _NNS_TEXT:
      /* supposed 1 frame in buffer */
      frame_size = self->frame_size;
      frames_in = 1;

      if (buf_size != frame_size) {
        GstMapInfo src_info, dest_info;
        gsize block_size = MIN (buf_size, frame_size);

        inbuf = gst_buffer_new_and_alloc (frame_size);
        gst_buffer_memset (inbuf, 0, 0, frame_size);

        if (FALSE == gst_buffer_map (buf, &src_info, GST_MAP_READ)) {
          ml_logf ("Cannot map src buffer at tensor_converter/text.\n");
          gst_buffer_unref (inbuf);     /* the new buffer is wasted. */
          return GST_FLOW_ERROR;
        }
        if (FALSE == gst_buffer_map (inbuf, &dest_info, GST_MAP_WRITE)) {
          ml_logf ("Cannot map dest buffer at tensor_converter/text.\n");
          gst_buffer_unmap (buf, &src_info);
          gst_buffer_unref (inbuf);     /* the new buffer is wasted. */
          return GST_FLOW_ERROR;
        }

        memcpy (dest_info.data, src_info.data, block_size);

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
    case _NNS_MEDIA_PLUGINS:
    {
      GstTensorsConfig new_config;

      if (self->externalConverter == NULL ||
          self->externalConverter->convert == NULL)
        return GST_FLOW_NOT_SUPPORTED;

      inbuf =
          self->externalConverter->convert (buf, &frame_size, &frames_in,
          &new_config);

      if (gst_tensors_config_is_equal (config, &new_config) != TRUE) {
        if (gst_tensor_converter_update_caps (self, self->srcpad,
                &new_config) == TRUE) {
          self->tensors_config = new_config;
        } else {
          return GST_FLOW_ERROR;
        }
      }

      g_assert (inbuf != NULL);
      g_assert (frame_size > 0);

      if (inbuf != buf)
        gst_buffer_unref (buf);

      break;
    }
    default:
      GST_ERROR_OBJECT (self, "Unsupported type %d\n", self->in_media_type);
      return GST_FLOW_ERROR;
  }

  /** convert format (bytes > time) and push segment event.
    * It will push event if needed (self->need_segment is true). */
  _gst_tensor_converter_chain_segment (self, config, frame_size);

  /** configures timestamp if required (self->set_timestamp is true) */
  _gst_tensor_converter_chain_timestamp (self, config, inbuf, frames_in);

  /* update old timestamp */
  self->old_timestamp = GST_BUFFER_TIMESTAMP (inbuf);

  if (frames_in == frames_out) {
    silent_debug_timestamp (inbuf);

    /** do nothing, push the incoming buffer */
    return _gst_tensor_converter_chain_push (self, inbuf);
  }

  /* push multiple buffers */
  return _gst_tensor_converter_chain_chunk (self, config, inbuf, frames_in,
      frames_out, frame_size);
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

  self->tensors_configured = FALSE;
  gst_tensors_config_init (&self->tensors_config);

  self->have_segment = FALSE;
  self->need_segment = FALSE;
  gst_segment_init (&self->segment, GST_FORMAT_TIME);

  self->old_timestamp = GST_CLOCK_TIME_NONE;
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
 * @brief Set the tensors config structure from video info (internal static function)
 * @param self this pointer to GstTensorConverter
 * @param caps caps for media stream
 * @param config tensors config structure to be filled
 * @note Change dimension if tensor contains N frames.
 * @return TRUE if supported type
 */
static gboolean
gst_tensor_converter_parse_video (GstTensorConverter * self,
    const GstCaps * caps, GstTensorsConfig * config)
{
  /**
   * Refer: https://www.tensorflow.org/api_docs/python/tf/summary/image
   * A 4-D uint8 or float32 Tensor of shape [batch_size, height, width, channels]
   * where channels is 1, 3, or 4.
   */
  GstVideoInfo vinfo;
  GstVideoFormat format;
  gint width, height;
  guint i;

  g_return_val_if_fail (config != NULL, FALSE);

  gst_tensors_config_init (config);

  gst_video_info_init (&vinfo);
  if (!gst_video_info_from_caps (&vinfo, caps)) {
    GST_ERROR_OBJECT (self, "Failed to get video info from caps.");
    return FALSE;
  }

  format = GST_VIDEO_INFO_FORMAT (&vinfo);
  width = GST_VIDEO_INFO_WIDTH (&vinfo);
  height = GST_VIDEO_INFO_HEIGHT (&vinfo);

  config->info.num_tensors = 1;

  /* [color-space][width][height][frames] */
  switch (format) {
    case GST_VIDEO_FORMAT_GRAY8:
      config->info.info[0].type = _NNS_UINT8;
      config->info.info[0].dimension[0] = 1;
      break;
    case GST_VIDEO_FORMAT_RGB:
    case GST_VIDEO_FORMAT_BGR:
      config->info.info[0].type = _NNS_UINT8;
      config->info.info[0].dimension[0] = 3;
      break;
    case GST_VIDEO_FORMAT_RGBx:
    case GST_VIDEO_FORMAT_BGRx:
    case GST_VIDEO_FORMAT_xRGB:
    case GST_VIDEO_FORMAT_xBGR:
    case GST_VIDEO_FORMAT_RGBA:
    case GST_VIDEO_FORMAT_BGRA:
    case GST_VIDEO_FORMAT_ARGB:
    case GST_VIDEO_FORMAT_ABGR:
      config->info.info[0].type = _NNS_UINT8;
      config->info.info[0].dimension[0] = 4;
      break;
    default:
      /* unsupported format */
      GST_WARNING_OBJECT (self, "Unsupported format = %s\n",
          GST_STR_NULL (gst_video_format_to_string (format)));
      break;
  }

  config->info.info[0].dimension[1] = width;
  config->info.info[0].dimension[2] = height;

  /* Supposed 1 frame in tensor, change dimension[3] if tensor contains N frames. */
  for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++) {
    config->info.info[0].dimension[i] = 1;
  }

  config->rate_n = GST_VIDEO_INFO_FPS_N (&vinfo);
  config->rate_d = GST_VIDEO_INFO_FPS_D (&vinfo);

  /**
   * Emit Warning if RSTRIDE = RU4 (3BPP) && Width % 4 > 0
   * @todo Add more conditions!
   */
  if (gst_tensor_converter_video_stride (format, width)) {
    self->remove_padding = TRUE;
    silent_debug ("Set flag to remove padding, width = %d", width);

    GST_WARNING_OBJECT (self,
        "\nYOUR STREAM CONFIGURATION INCURS PERFORMANCE DETERIORATION!\n"
        "Please use 4 x n as image width for inputs.\n");
  }

  self->frame_size = GST_VIDEO_INFO_SIZE (&vinfo);
  return (config->info.info[0].type != _NNS_END);
}

/**
 * @brief Set the tensors config structure from audio info (internal static function)
 * @param self this pointer to GstTensorConverter
 * @param caps caps for media stream
 * @param config tensors config structure to be filled
 * @note Change dimension if tensor contains N frames.
 * @return TRUE if supported type
 */
static gboolean
gst_tensor_converter_parse_audio (GstTensorConverter * self,
    const GstCaps * caps, GstTensorsConfig * config)
{
  /**
   * Refer: https://www.tensorflow.org/api_docs/python/tf/summary/audio
   * A 3-D float32 Tensor of shape [batch_size, frames, channels]
   * or a 2-D float32 Tensor of shape [batch_size, frames].
   */
  GstAudioInfo ainfo;
  GstAudioFormat format;
  gint channels;
  guint i;

  g_return_val_if_fail (config != NULL, FALSE);

  gst_tensors_config_init (config);

  gst_audio_info_init (&ainfo);
  if (!gst_audio_info_from_caps (&ainfo, caps)) {
    GST_ERROR_OBJECT (self, "Failed to get audio info from caps.\n");
    return FALSE;
  }

  format = GST_AUDIO_INFO_FORMAT (&ainfo);
  channels = GST_AUDIO_INFO_CHANNELS (&ainfo);

  config->info.num_tensors = 1;

  /* [channels][frames] */
  switch (format) {
    case GST_AUDIO_FORMAT_S8:
      config->info.info[0].type = _NNS_INT8;
      break;
    case GST_AUDIO_FORMAT_U8:
      config->info.info[0].type = _NNS_UINT8;
      break;
    case GST_AUDIO_FORMAT_S16:
      config->info.info[0].type = _NNS_INT16;
      break;
    case GST_AUDIO_FORMAT_U16:
      config->info.info[0].type = _NNS_UINT16;
      break;
    case GST_AUDIO_FORMAT_S32:
      config->info.info[0].type = _NNS_INT32;
      break;
    case GST_AUDIO_FORMAT_U32:
      config->info.info[0].type = _NNS_UINT32;
      break;
    case GST_AUDIO_FORMAT_F32:
      config->info.info[0].type = _NNS_FLOAT32;
      break;
    case GST_AUDIO_FORMAT_F64:
      config->info.info[0].type = _NNS_FLOAT64;
      break;
    default:
      /* unsupported format */
      GST_WARNING_OBJECT (self, "Unsupported format = %s\n",
          GST_STR_NULL (gst_audio_format_to_string (format)));
      break;
  }

  config->info.info[0].dimension[0] = channels;

  /* Supposed 1 frame in tensor, change dimension[1] if tensor contains N frames. */
  for (i = 1; i < NNS_TENSOR_RANK_LIMIT; i++) {
    config->info.info[0].dimension[i] = 1;
  }

  config->rate_n = GST_AUDIO_INFO_RATE (&ainfo);
  config->rate_d = 1;

  self->frame_size = GST_AUDIO_INFO_BPF (&ainfo);
  return (config->info.info[0].type != _NNS_END);
}

/**
 * @brief Set the tensors config structure from text info (internal static function)
 * @param self this pointer to GstTensorConverter
 * @param config tensors config structure to be filled
 * @param structure caps structure
 * @note Change dimension if tensors contains N frames.
 * @return TRUE if supported type
 */
static gboolean
gst_tensor_converter_parse_text (GstTensorConverter * self,
    GstTensorsConfig * config, const GstStructure * structure)
{
  /**
   * Refer: https://www.tensorflow.org/api_docs/python/tf/summary/text
   * A string-type Tensor
   */
  const gchar *format_string;
  guint i, text_size;

  g_return_val_if_fail (config != NULL, FALSE);
  g_return_val_if_fail (structure != NULL, FALSE);

  gst_tensors_config_init (config);

  /* get fixed size of text string from property */
  text_size = self->tensors_info.info[0].dimension[0];
  if (text_size == 0) {
    GST_ERROR_OBJECT (self,
        "Failed to get tensor info, need to update string size.");

    ml_loge ("Please set the property input-dim to convert stream.\n"
        "For example, input-dim=30 to handle up to 30 bytes of string per frame.");
    return FALSE;
  }

  format_string = gst_structure_get_string (structure, "format");
  if (format_string) {
    if (g_ascii_strcasecmp (format_string, "utf8") == 0) {
      config->info.info[0].type = _NNS_UINT8;
    } else {
      /* unsupported format */
      GST_WARNING_OBJECT (self, "Unsupported format = %s\n", format_string);
      return FALSE;
    }
  }

  config->info.num_tensors = 1;

  /* [size][frames] */
  /* Fixed size of string, we cannot get the size from caps. */
  config->info.info[0].dimension[0] = text_size;

  /* Supposed 1 frame in tensor, change dimension[1] if tensor contains N frames. */
  for (i = 1; i < NNS_TENSOR_RANK_LIMIT; i++) {
    config->info.info[0].dimension[i] = 1;
  }

  if (gst_structure_has_field (structure, "framerate")) {
    gst_structure_get_fraction (structure, "framerate", &config->rate_n,
        &config->rate_d);
  } else {
    /* cannot get the framerate for text type */
    config->rate_n = 0;
    config->rate_d = 1;
  }

  self->frame_size = gst_tensor_info_get_size (&config->info.info[0]);
  return (config->info.info[0].type != _NNS_END);
}

/**
 * @brief Set the tensors configs structure from octet stream (internal static function)
 * @param self this pointer to GstTensorConverter
 * @param config tensors config structure to be filled
 * @param structure caps structure
 * @note Change tensors dimension and type.
 * @return TRUE if supported type
 */
static gboolean
gst_tensor_converter_parse_octet (GstTensorConverter * self,
    GstTensorsConfig * config, const GstStructure * structure)
{
  g_return_val_if_fail (config != NULL, FALSE);
  g_return_val_if_fail (structure != NULL, FALSE);

  gst_tensors_config_init (config);

  /* update tensor info from properties */
  if (!gst_tensors_info_validate (&self->tensors_info)) {
    GST_ERROR_OBJECT (self,
        "Failed to get tensor info, need to update dimension and type.");

    ml_loge
        ("Please set the properties input-dim and input-type to convert stream.\n"
        "For example, input-dim=30 input-type=unit8 to handle 30 bytes of bin data.");
    return FALSE;
  }

  if (self->tensors_info.num_tensors > 1 && self->frames_per_tensor > 1) {
    ml_loge
        ("Cannot configure multiple tensors. Please set the property frames-per-tensor 1 to convert stream.");
    return FALSE;
  }

  /**
   * Raw byte-stream (application/octet-stream)
   * We cannot get the exact tensors info from caps.
   * All tensors info should be updated.
   */
  gst_tensors_info_copy (&config->info, &self->tensors_info);

  if (gst_structure_has_field (structure, "framerate")) {
    gst_structure_get_fraction (structure, "framerate", &config->rate_n,
        &config->rate_d);
  } else {
    /* cannot get the framerate */
    config->rate_n = 0;
    config->rate_d = 1;
  }

  self->frame_size = gst_tensors_info_get_size (&config->info, -1);
  return (config->info.info[0].type != _NNS_END);
}

/**
 * @brief Get possible media-caps from downstream element.
 */
static GstCaps *
gst_tensor_converter_get_possible_media_caps (GstTensorConverter * self)
{
  GstCaps *media_caps = NULL;
  GstCaps *peer_caps;

  /* get possible caps from downstream element */
  peer_caps = gst_pad_peer_query_caps (self->srcpad, NULL);

  if (peer_caps) {
    silent_debug_caps (peer_caps, "peer caps");

    if (gst_caps_get_size (peer_caps) > 0) {
      GstTensorsConfig config;
      GstStructure *st;
      guint i, caps_len;
      media_type type;

      /* get tensors info from peer caps */
      st = gst_caps_get_structure (peer_caps, 0);
      gst_tensors_config_from_structure (&config, st);

      /* convert peer caps to possible media caps */
      media_caps = gst_pad_get_pad_template_caps (self->sinkpad);
      media_caps = gst_caps_make_writable (media_caps);

      caps_len = gst_caps_get_size (media_caps);

      for (i = 0; i < caps_len; ++i) {
        st = gst_caps_get_structure (media_caps, i);
        type = gst_tensor_media_type_from_structure (st);

        switch (type) {
          case _NNS_VIDEO:
            /* video caps from tensor info */
            if (is_video_supported (self)
                && config.info.info[0].type == _NNS_UINT8) {
              GValue supported_formats = G_VALUE_INIT;
              gint colorspace, width, height;

              colorspace = config.info.info[0].dimension[0];
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

              if (G_VALUE_TYPE (&supported_formats) == GST_TYPE_LIST &&
                  gst_value_list_get_size (&supported_formats) > 0) {
                gst_structure_set_value (st, "format", &supported_formats);
              }
              g_value_unset (&supported_formats);

              if ((width = config.info.info[0].dimension[1]) > 0) {
                gst_structure_set (st, "width", G_TYPE_INT, width, NULL);
              }

              if ((height = config.info.info[0].dimension[2]) > 0) {
                gst_structure_set (st, "height", G_TYPE_INT, height, NULL);
              }

              if (config.rate_n >= 0 && config.rate_d > 0) {
                gst_structure_set (st, "framerate", GST_TYPE_FRACTION,
                    config.rate_n, config.rate_d, NULL);
              }
            }
            break;
          case _NNS_AUDIO:
            /* audio caps from tensor info */
            if (is_audio_supported (self)
                && config.info.info[0].type != _NNS_END) {
              gint ch, rate;
              GstAudioFormat aformat;

              switch (config.info.info[0].type) {
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

                if ((ch = config.info.info[0].dimension[0]) > 0) {
                  gst_structure_set (st, "channels", G_TYPE_INT, ch, NULL);
                }

                if ((rate = config.rate_n) > 0) {
                  gst_structure_set (st, "rate", G_TYPE_INT, rate, NULL);
                }
              }
            }
            break;
          default:
            /* do nothing for text and octet stream */
            break;
        }
      }
    }

    gst_caps_unref (peer_caps);
  }

  return media_caps;
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
    GstCaps *media_caps;

    media_caps = gst_tensor_converter_get_possible_media_caps (self);
    if (media_caps) {
      /* intersect with pad caps */
      GstCaps *tmp = gst_caps_intersect_full (media_caps, caps,
          GST_CAPS_INTERSECT_FIRST);
      gst_caps_unref (caps);
      caps = tmp;

      gst_caps_unref (media_caps);
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
 * @brief Parse caps and set tensors info.
 */
static gboolean
gst_tensor_converter_parse_caps (GstTensorConverter * self,
    const GstCaps * caps)
{
  GstStructure *structure;
  GstTensorsConfig config;
  media_type in_type;
  gint frames_dim = -1; /** dimension index of frames in configured tensors */

  g_return_val_if_fail (caps != NULL, FALSE);
  g_return_val_if_fail (gst_caps_is_fixed (caps), FALSE);

  structure = gst_caps_get_structure (caps, 0);
  in_type = gst_tensor_media_type_from_structure (structure);

  switch (in_type) {
    case _NNS_VIDEO:
      if (is_video_supported (self)) {
        if (!gst_tensor_converter_parse_video (self, caps, &config)) {
          GST_ERROR_OBJECT (self,
              "Failed to configure tensor from video info.");
          return FALSE;
        }

        frames_dim = 3;
      } else {
        ml_loge
            ("\n This binary does not support video type. Please build NNStreamer with disable-video-support : false\n");
        return FALSE;
      }
      break;
    case _NNS_AUDIO:
      if (is_audio_supported (self)) {
        if (!gst_tensor_converter_parse_audio (self, caps, &config)) {
          GST_ERROR_OBJECT (self,
              "Failed to configure tensor from audio info.");
          return FALSE;
        }

        frames_dim = 1;
      } else {
        ml_loge
            ("\n This binary does not support audio type. Please build NNStreamer with disable-audio-support : false\n");
        return FALSE;
      }
      break;
    case _NNS_TEXT:
      if (!gst_tensor_converter_parse_text (self, &config, structure)) {
        GST_ERROR_OBJECT (self, "Failed to configure tensor from text info.");
        return FALSE;
      }

      frames_dim = 1;
      break;
    case _NNS_OCTET:
      if (!gst_tensor_converter_parse_octet (self, &config, structure)) {
        GST_ERROR_OBJECT (self, "Failed to configure tensors from octet info.");
        return FALSE;
      }

      break;
    default:
    {
      /* if found, configure in_mdeia_type = _NNS_MEDIA_PLUGINS */
      const gchar *struct_name;

      struct_name = gst_structure_get_name (structure);

      if (struct_name != NULL) {
        const NNStreamerExternalConverter *ex;

        ex = findExternalConverter (struct_name);

        if (ex != NULL && self->externalConverter == NULL) {
          in_type = _NNS_MEDIA_PLUGINS;
          self->externalConverter = ex;

          if (NULL == ex->get_out_config || !ex->get_out_config (caps, &config)) {
            GST_ERROR_OBJECT (self,
                "Failed to get tensors info from %s. Check the given options.",
                struct_name);
            ml_loge ("Please set the options property correctly.\n");
            self->externalConverter = NULL;
            return FALSE;
          }
          break;
        }
      }
      GST_ERROR_OBJECT (self, "Unsupported type %d\n", in_type);
      return FALSE;
    }
  }

  /** set the number of frames in dimension */
  if (frames_dim >= 0) {
    config.info.info[0].dimension[frames_dim] = self->frames_per_tensor;
  }

  if (!gst_tensors_config_validate (&config)) {
    /** not fully configured */
    GST_ERROR_OBJECT (self, "Failed to configure tensors info.\n");
    return FALSE;
  }

  if (gst_tensors_info_validate (&self->tensors_info)) {
    /** compare tensor info */
    if (!gst_tensors_info_is_equal (&self->tensors_info, &config.info)) {
      GST_ERROR_OBJECT (self, "Failed, mismatched tensor info.\n");
      return FALSE;
    }
  }

  self->in_media_type = in_type;
  self->tensors_configured = TRUE;
  self->tensors_config = config;

  return TRUE;
}

/**
 * @brief Update pas caps from tensors config
 */
static gboolean
gst_tensor_converter_update_caps (GstTensorConverter * self,
    GstPad * pad, GstTensorsConfig * config)
{
  GstCaps *curr_caps, *out_caps = NULL;

  out_caps = gst_tensors_get_caps (pad, config);

  /* Update pad caps. If it is different */
  curr_caps = gst_pad_get_current_caps (pad);
  if (curr_caps == NULL || !gst_caps_is_equal (curr_caps, out_caps)) {
    gst_pad_set_caps (pad, out_caps);
  }

  if (curr_caps)
    gst_caps_unref (curr_caps);

  gst_caps_unref (out_caps);
  return TRUE;
}

/**
 * @brief Converter's external subplugins should call this at init.
 */
int
registerExternalConverter (NNStreamerExternalConverter * ex)
{
  return register_subplugin (NNS_SUBPLUGIN_CONVERTER, ex->name, ex);
}

/**
 * @brief Converter's external subplugins should call this at exit.
 */
void
unregisterExternalConverter (const char *name)
{
  unregister_subplugin (NNS_SUBPLUGIN_CONVERTER, name);
}

/**
 * @brief Internal static function to find registered subplugins.
 */
static const NNStreamerExternalConverter *
findExternalConverter (const char *media_type)
{
  guint total, i, j, caps_size;
  subplugin_info_s info;
  GstCaps *caps;
  const gchar *caps_name;
  const NNStreamerExternalConverter *ex;

  total = nnsconf_get_subplugin_info (NNSCONF_PATH_CONVERTERS, &info);
  for (i = 0; i < total; i++) {
    ex = get_subplugin (NNS_SUBPLUGIN_CONVERTER, info.names[i]);

    if (ex && ex->query_caps) {
      caps = ex->query_caps (NULL);
      caps_size = gst_caps_get_size (caps);

      for (j = 0; j < caps_size; j++) {
        caps_name = gst_structure_get_name (gst_caps_get_structure (caps, j));
        if (g_strcmp0 (media_type, caps_name) == 0) {
          /* found matched media type */
          gst_caps_unref (caps);
          return ex;
        }
      }

      gst_caps_unref (caps);
    }
  }

  return NULL;
}

/**
 * @brief set custom property description for tensor converter sub-plugin
 */
void
nnstreamer_converter_set_custom_property_desc (const char *name,
    const char *prop, ...)
{
  va_list varargs;

  va_start (varargs, prop);
  subplugin_set_custom_property_desc (NNS_SUBPLUGIN_CONVERTER, name, prop,
      varargs);
  va_end (varargs);
}

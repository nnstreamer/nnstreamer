/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer/NNStreamer Tensor-Rate
 * Copyright (C) 2020 Dongju Chae <dongju.chae@samsung.com>
 */

/**
 * @file    gsttensorrate.c
 * @date    24 Sep 2020
 * @brief   GStreamer plugin to adjust tensor rate
 * @see     https://github.com/nnstreamer/nnstreamer
 * @author  Dongju Chae <dongju.chae@samsung.com>
 * @bug     No known bugs except for NYI items
 */

/**
 * SECTION:element-tensor_rate
 *
 * This element controls a frame rate of tensor streams in the pipeline.
 *
 * Basically, this element takes an incoming stream of tensor frames, and
 * produces an adjusted stream that matches the source pad's framerate.
 * The adjustment is performed by dropping and duplicating tensor frames.
 * By default the element will simply negotiate the same framerate on its
 * source and sink pad.
 *
 * Also, when 'throttle' property is set, it propagates a specified frame-rate
 * to upstream elements by sending qos events, which prevents unnecessary
 * data from upstream elements.
 *
 * <refsect2>
 * <title>Example launch line with tensor rate</title>
 * gst-launch-1.0 videotestsrc
 *      ! video/x-raw,width=640,height=480,framerate=15/1
 *      ! tensor_converter
 *      ! tensor_rate framerate=10/1 throttle=true
 *      ! tensor_decoder mode=direct_video
 *      ! videoconvert
 *      ! autovideosink
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <nnstreamer_log.h>

#include "gsttensorrate.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

#ifndef ABSDIFF
#define ABSDIFF(a, b) (((a) > (b)) ? (a) - (b) : (b) - (a))
#endif

#define silent_debug(...) do { \
    if (DBG) { \
      GST_DEBUG_OBJECT (self, __VA_ARGS__); \
    } \
  } while (0)

#define silent_debug_caps(caps,msg) do {\
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

GST_DEBUG_CATEGORY_STATIC (gst_tensor_rate_debug);
#define GST_CAT_DEFAULT gst_tensor_rate_debug

#define CAPS_STRING GST_TENSOR_CAP_DEFAULT "; " GST_TENSORS_CAP_DEFAULT

#define GST_TENSOR_RATE_SCALED_TIME(self, count)\
  gst_util_uint64_scale (count,\
      self->to_rate_denominator * GST_SECOND, self->to_rate_numerator)

/** @brief default parameters */
#define DEFAULT_SILENT    TRUE
#define DEFAULT_THROTTLE  TRUE

/**
 * @brief tensor_rate properties
 */
enum
{
  PROP_0,
  PROP_IN,
  PROP_OUT,
  PROP_DUP,
  PROP_DROP,
  PROP_SILENT,
  PROP_THROTTLE,
  PROP_FRAMERATE,
};

/**
 * @brief The capabilities of the inputs
 */
static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

/**
 * @brief The capabilities of the outputs
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (CAPS_STRING));

static GParamSpec *pspec_drop = NULL;
static GParamSpec *pspec_duplicate = NULL;

#define gst_tensor_rate_parent_class parent_class
G_DEFINE_TYPE (GstTensorRate, gst_tensor_rate, GST_TYPE_BASE_TRANSFORM);

/* GObject vmethod implementations */
static void gst_tensor_rate_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_rate_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_rate_finalize (GObject * object);

/* GstBaseTransform vmethod implementations */
static GstFlowReturn gst_tensor_rate_transform_ip (GstBaseTransform * trans,
    GstBuffer * buffer);
static GstCaps *gst_tensor_rate_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * _rate);
static GstCaps *gst_tensor_rate_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_tensor_rate_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
static void gst_tensor_rate_swap_prev (GstTensorRate * self,
    GstBuffer * buffer, gint64 time);
static GstFlowReturn gst_tensor_rate_flush_prev (GstTensorRate * self,
    gboolean duplicate, GstClockTime next_intime);

static void gst_tensor_rate_notify_drop (GstTensorRate * self);
static void gst_tensor_rate_notify_duplicate (GstTensorRate * self);

static gboolean gst_tensor_rate_start (GstBaseTransform * trans);
static gboolean gst_tensor_rate_stop (GstBaseTransform * trans);
static gboolean gst_tensor_rate_sink_event (GstBaseTransform * trans,
    GstEvent * event);

static void gst_tensor_rate_install_properties (GObjectClass * gobject_class);

/**
 * @brief initialize the tensor_rate's class (GST Standard)
 */
static void
gst_tensor_rate_class_init (GstTensorRateClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_rate_debug, "tensor_rate", 0,
      "Tensor Rate to control streams based on tensor(s) values");

  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_rate_set_property;
  gobject_class->get_property = gst_tensor_rate_get_property;
  gobject_class->finalize = gst_tensor_rate_finalize;

  gst_tensor_rate_install_properties (gobject_class);

  gst_element_class_set_details_simple (gstelement_class,
      "TensorRate",
      "Filter/Tensor",
      "Adjusts a framerate of incoming tensors",
      "Dongju Chae <dongju.chae@samsung.com>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));

  trans_class->passthrough_on_same_caps = TRUE;
  trans_class->transform_ip_on_passthrough = TRUE;

  /* Processing units */
  trans_class->transform_ip = GST_DEBUG_FUNCPTR (gst_tensor_rate_transform_ip);

  /* Negotiation units */
  trans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_rate_transform_caps);
  trans_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_tensor_rate_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_rate_set_caps);

  /* setup sink event */
  trans_class->sink_event = GST_DEBUG_FUNCPTR (gst_tensor_rate_sink_event);

  /* start/stop to call open/close */
  trans_class->start = GST_DEBUG_FUNCPTR (gst_tensor_rate_start);
  trans_class->stop = GST_DEBUG_FUNCPTR (gst_tensor_rate_stop);
}

/**
 * @brief push the buffer to src pad
 */
static GstFlowReturn
gst_tensor_rate_push_buffer (GstTensorRate * self, GstBuffer * outbuf,
    gboolean duplicate, GstClockTime next_intime)
{
  GstFlowReturn res;
  GstClockTime push_ts;

  GST_BUFFER_OFFSET (outbuf) = self->out;
  GST_BUFFER_OFFSET_END (outbuf) = self->out + 1;
  GST_BUFFER_FLAG_UNSET (outbuf, GST_BUFFER_FLAG_DISCONT);

  if (duplicate)
    GST_BUFFER_FLAG_SET (outbuf, GST_BUFFER_FLAG_GAP);
  else
    GST_BUFFER_FLAG_UNSET (outbuf, GST_BUFFER_FLAG_GAP);

  /* this is the timestamp we put on the buffer */
  push_ts = self->next_ts;

  self->out++;
  self->out_frame_count++;

  if (self->to_rate_numerator) {
    GstClockTimeDiff duration;

    duration = GST_TENSOR_RATE_SCALED_TIME (self, self->out_frame_count);

    /* interpolate next expected timestamp in the segment */
    self->next_ts = self->segment.base + self->segment.start +
        self->base_ts + duration;

    GST_BUFFER_DURATION (outbuf) = self->next_ts - push_ts;
  } else {
    /** There must always be a valid duration on prevbuf if rate > 0,
     * it is ensured in the transform_ip function */
    g_assert (GST_BUFFER_PTS_IS_VALID (outbuf));
    g_assert (GST_BUFFER_DURATION_IS_VALID (outbuf));
    g_assert (GST_BUFFER_DURATION (outbuf) != 0);

    self->next_ts = GST_BUFFER_PTS (outbuf) + GST_BUFFER_DURATION (outbuf);
  }

  /* adapt for looping, bring back to time in current segment. */
  GST_BUFFER_TIMESTAMP (outbuf) = push_ts - self->segment.base;

  silent_debug ("old is best, dup, pushing buffer outgoing ts %"
      GST_TIME_FORMAT, GST_TIME_ARGS (push_ts));

  res = gst_pad_push (GST_BASE_TRANSFORM_SRC_PAD (self), outbuf);

  return res;
}

/**
 * @brief flush the oldest buffer
 */
static GstFlowReturn
gst_tensor_rate_flush_prev (GstTensorRate * self, gboolean duplicate,
    GstClockTime next_intime)
{
  GstBuffer *outbuf;

  if (!self->prevbuf) {
    ml_logi ("got EOS before any buffer was received");
    return GST_FLOW_OK;
  }

  outbuf = gst_buffer_ref (self->prevbuf);
  /* make sure we can write to the metadata */
  outbuf = gst_buffer_make_writable (outbuf);

  return gst_tensor_rate_push_buffer (self, outbuf, duplicate, next_intime);
}

/**
 * @brief swap a previous buffer
 */
static void
gst_tensor_rate_swap_prev (GstTensorRate * self, GstBuffer * buffer,
    gint64 time)
{
  silent_debug ("swap_prev: storing buffer %p in prev", buffer);

  if (self->prevbuf)
    gst_buffer_unref (self->prevbuf);
  self->prevbuf = buffer != NULL ? gst_buffer_ref (buffer) : NULL;
  self->prev_ts = time;
}

/**
 * @brief reset variables of the element (GST Standard)
 */
static void
gst_tensor_rate_reset (GstTensorRate * self)
{
  self->in = 0;
  self->out = 0;
  self->drop = 0;
  self->dup = 0;

  self->out_frame_count = 0;

  self->base_ts = 0;
  self->next_ts = GST_CLOCK_TIME_NONE;
  self->last_ts = GST_CLOCK_TIME_NONE;

  self->sent_qos_on_passthrough = FALSE;

  gst_tensor_rate_swap_prev (self, NULL, 0);
}

/**
 * @brief initialize the new element (GST Standard)
 */
static void
gst_tensor_rate_init (GstTensorRate * self)
{
  gst_tensor_rate_reset (self);

  self->silent = DEFAULT_SILENT;
  self->throttle = DEFAULT_THROTTLE;

  /* decided from caps negotiation */
  self->from_rate_numerator = 0;
  self->from_rate_denominator = 0;
  self->to_rate_numerator = 0;
  self->to_rate_denominator = 0;

  /* specified from property */
  self->rate_n = -1;
  self->rate_d = -1;

  gst_segment_init (&self->segment, GST_FORMAT_TIME);
}

/**
 * @brief Function to finalize instance. (GST Standard)
 */
static void
gst_tensor_rate_finalize (GObject * object)
{
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_rate properties.
 */
static void
gst_tensor_rate_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorRate *self = GST_TENSOR_RATE (object);

  GST_OBJECT_LOCK (self);

  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      break;
    case PROP_THROTTLE:
      self->throttle = g_value_get_boolean (value);
      break;
    case PROP_FRAMERATE:
    {
      const gchar *str = g_value_get_string (value);
      gchar **strv = g_strsplit (str, "/", -1);
      gint rate_n, rate_d;

      if (g_strv_length (strv) != 2) {
        ml_loge ("Please specify a proper 'framerate' property");
        break;
      }

      rate_n = (gint) g_ascii_strtoll (strv[0], NULL, 10);
      if (errno == ERANGE || rate_n < 0) {
        ml_loge ("Invalid frame rate numerator in 'framerate'");
        g_strfreev (strv);
        break;
      }

      rate_d = (gint) g_ascii_strtoll (strv[1], NULL, 10);
      if (errno == ERANGE || rate_d <= 0) {
        ml_loge ("Invalid frame rate denominator in 'framerate'");
        g_strfreev (strv);
        break;
      }

      self->rate_n = rate_n;
      self->rate_d = rate_d;

      g_strfreev (strv);
    }
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }

  GST_OBJECT_UNLOCK (self);
}

/**
 * @brief Getter for tensor_rate properties.
 */
static void
gst_tensor_rate_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorRate *self = GST_TENSOR_RATE (object);

  GST_OBJECT_LOCK (self);

  switch (prop_id) {
    case PROP_IN:
      g_value_set_uint64 (value, self->in);
      break;
    case PROP_OUT:
      g_value_set_uint64 (value, self->out);
      break;
    case PROP_DROP:
      g_value_set_uint64 (value, self->drop);
      break;
    case PROP_DUP:
      g_value_set_uint64 (value, self->dup);
      break;
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    case PROP_THROTTLE:
      g_value_set_boolean (value, self->throttle);
      break;
    case PROP_FRAMERATE:
      if (self->rate_n < 0 || self->rate_d <= 0) {
        g_value_set_string (value, "");
      } else {
        gchar *str = g_strdup_printf ("%d/%d", self->rate_n, self->rate_d);
        g_value_take_string (value, str);
      }
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }

  GST_OBJECT_UNLOCK (self);
}

#define THROTTLE_DELAY_RATIO (0.999)

/**
 * @brief send throttling qos event to upstream elements
 */
static void
gst_tensor_rate_send_qos_throttle (GstTensorRate * self, GstClockTime timestamp)
{
  GstPad *sinkpad = GST_BASE_TRANSFORM_SINK_PAD (&self->element);
  GstClockTimeDiff delay;
  GstEvent *event;

  delay = GST_TENSOR_RATE_SCALED_TIME (self, 1);
  delay = (GstClockTimeDiff) (((gdouble) delay) * THROTTLE_DELAY_RATIO);

  event = gst_event_new_qos (GST_QOS_TYPE_THROTTLE,
      0.9 /** unused */ , delay, timestamp);

  silent_debug ("Send throttling event with delay: %" GST_TIME_FORMAT,
      GST_TIME_ARGS (delay));

  gst_pad_push_event (sinkpad, event);
}

/**
 * @brief in-place transform
 */
static GstFlowReturn
gst_tensor_rate_transform_ip (GstBaseTransform * trans, GstBuffer * buffer)
{
  GstTensorRate *self = GST_TENSOR_RATE (trans);
  GstFlowReturn res = GST_BASE_TRANSFORM_FLOW_DROPPED;
  GstClockTime intime, in_ts, in_dur;

  /* make sure the denominators are not 0 */
  if (self->from_rate_denominator == 0 || self->to_rate_denominator == 0) {
    ml_loge ("No framerate negotiated");
    return GST_FLOW_NOT_NEGOTIATED;
  }

  /* tensor streams do not support reverse playback */
  if (G_UNLIKELY (self->segment.rate < 0.0)) {
    ml_loge ("Unsupported reverse playback\n");
    return GST_FLOW_ERROR;
  }

  in_ts = GST_BUFFER_TIMESTAMP (buffer);
  in_dur = GST_BUFFER_DURATION (buffer);

  if (G_UNLIKELY (!GST_CLOCK_TIME_IS_VALID (in_ts))) {
    in_ts = self->last_ts;
    if (G_UNLIKELY (!GST_CLOCK_TIME_IS_VALID (in_ts))) {
      ml_logw ("Discard an invalid buffer");
      return GST_BASE_TRANSFORM_FLOW_DROPPED;
    }
  }

  self->in++;

  /* update the last timestamp */
  self->last_ts = in_ts;
  if (GST_CLOCK_TIME_IS_VALID (in_dur))
    self->last_ts += in_dur;

  silent_debug ("got buffer with timestamp %" GST_TIME_FORMAT,
      GST_TIME_ARGS (in_ts));

  intime = in_ts + self->segment.base;

  /* let's send a QoS event even if pass-through is used on the same caps */
  if (gst_base_transform_is_passthrough (trans)) {
    if (!self->sent_qos_on_passthrough) {
      self->sent_qos_on_passthrough = TRUE;
      gst_tensor_rate_send_qos_throttle (self, intime);
    }

    self->out++;
    return GST_FLOW_OK;
  }

  /* we need to have two buffers to compare */
  if (self->prevbuf == NULL) {
    gst_tensor_rate_swap_prev (self, buffer, intime);
    if (!GST_CLOCK_TIME_IS_VALID (self->next_ts)) {
      self->next_ts = intime;
      self->base_ts = in_ts - self->segment.start;
      self->out_frame_count = 0;
    }
  } else {
    GstClockTime prevtime;
    gint64 diff1 = 0, diff2 = 0;
    guint count = 0;

    prevtime = self->prev_ts;

    silent_debug ("BEGINNING prev buf %" GST_TIME_FORMAT " new buf %"
        GST_TIME_FORMAT " outgoing ts %" GST_TIME_FORMAT,
        GST_TIME_ARGS (prevtime), GST_TIME_ARGS (intime),
        GST_TIME_ARGS (self->next_ts));

    /* drop new buffer if it's before previous one */
    if (intime < prevtime) {
      silent_debug ("The new buffer (%" GST_TIME_FORMAT ") is before "
          "the previous buffer (%" GST_TIME_FORMAT
          "). Dropping new buffer.", GST_TIME_ARGS (intime),
          GST_TIME_ARGS (prevtime));
      self->drop++;

      if (!self->silent)
        gst_tensor_rate_notify_drop (self);

      return GST_BASE_TRANSFORM_FLOW_DROPPED;
    }

    /* got 2 buffers, see which one is the best */
    do {
      GstClockTime next_ts;

      /* Make sure that we have a duration for previous buffer */
      if (!GST_BUFFER_DURATION_IS_VALID (self->prevbuf))
        GST_BUFFER_DURATION (self->prevbuf) =
            intime > prevtime ? intime - prevtime : 0;

      next_ts = self->base_ts + (self->next_ts - self->base_ts);

      diff1 = ABSDIFF (prevtime, next_ts);
      diff2 = ABSDIFF (intime, next_ts);

      silent_debug ("diff with prev %" GST_TIME_FORMAT
          " diff with new %" GST_TIME_FORMAT " outgoing ts %"
          GST_TIME_FORMAT, GST_TIME_ARGS (diff1),
          GST_TIME_ARGS (diff2), GST_TIME_ARGS (next_ts));

      /* output first one when its the best */
      if (diff1 <= diff2) {
        GstFlowReturn r;
        count++;

        /* on error the _flush function posted a warning already */
        if ((r = gst_tensor_rate_flush_prev (self,
                    count > 1, intime)) != GST_FLOW_OK) {
          return r;
        }
      }

      /**
       * continue while the first one was the best, if they were equal avoid
       * going into an infinite loop
       */
    } while (diff1 < diff2);

    /* if we outputted the first buffer more then once, we have dups */
    if (count > 1) {
      self->dup += count - 1;
      if (!self->silent)
        gst_tensor_rate_notify_duplicate (self);
    }
    /* if we didn't output the first buffer, we have a drop */
    else if (count == 0) {
      self->drop++;

      if (!self->silent)
        gst_tensor_rate_notify_drop (self);

      gst_tensor_rate_send_qos_throttle (self, intime);
    }

    /* swap in new one when it's the best */
    gst_tensor_rate_swap_prev (self, buffer, intime);
  }

  return res;
}

/**
 * @brief configure tensor-srcpad cap from "proposed" cap. (GST Standard)
 *
 * @trans ("this" pointer)
 * @direction (why do we need this?)
 * @caps sinkpad cap (if direction GST_PAD_SINK)
 * @filter this element's cap (don't know specifically.)
 *
 * Be careful not to fix/set caps at this stage. Negotiation not completed yet.
 */
static GstCaps *
gst_tensor_rate_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstTensorRate *self = GST_TENSOR_RATE (trans);
  GstCaps *result = gst_caps_new_empty ();
  gint i;

  silent_debug ("Direction = %d\n", direction);
  silent_debug_caps (caps, "from");
  silent_debug_caps (filter, "filter");

  for (i = 0; i < gst_caps_get_size (caps); i++) {
    GstStructure *s, *const_s = gst_caps_get_structure (caps, i);

    s = gst_structure_copy (const_s);

    /* when a target framerate is specified */
    if (direction == GST_PAD_SINK && self->rate_n >= 0 && self->rate_d > 0) {
      gst_structure_set (s, "framerate", GST_TYPE_FRACTION,
          self->rate_n, self->rate_d, NULL);
    } else {
      gst_structure_set (s, "framerate", GST_TYPE_FRACTION_RANGE,
          0, 1, G_MAXINT, 1, NULL);
    }

    result = gst_caps_merge_structure_full (result, s,
        gst_caps_features_copy (gst_caps_get_features (caps, i)));
  }

  if (filter && gst_caps_get_size (filter) > 0) {
    GstCaps *intersection =
        gst_caps_intersect_full (filter, result, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (result);
    result = intersection;
  }

  silent_debug_caps (result, "to");

  return result;
}

/**
 * @brief fixate caps. required vmethod of GstBaseTransform.
 */
static GstCaps *
gst_tensor_rate_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  GstStructure *s;
  gint num, denom;

  s = gst_caps_get_structure (caps, 0);
  if (G_UNLIKELY (!gst_structure_get_fraction (s, "framerate", &num, &denom)))
    return othercaps;

  othercaps = gst_caps_truncate (othercaps);
  othercaps = gst_caps_make_writable (othercaps);

  s = gst_caps_get_structure (othercaps, 0);
  gst_structure_fixate_field_nearest_fraction (s, "framerate", num, denom);

  return gst_caps_fixate (othercaps);
}

/**
 * @brief set caps. required vmethod of GstBaseTransform.
 */
static gboolean
gst_tensor_rate_set_caps (GstBaseTransform * trans,
    GstCaps * in_caps, GstCaps * out_caps)
{
  GstTensorRate *self = GST_TENSOR_RATE (trans);
  GstStructure *structure;
  gint rate_numerator, rate_denominator;

  silent_debug ("setcaps called in: %" GST_PTR_FORMAT " out: %" GST_PTR_FORMAT,
      in_caps, out_caps);

  structure = gst_caps_get_structure (in_caps, 0);

  if (!gst_structure_get_fraction (structure, "framerate",
          &rate_numerator, &rate_denominator))
    goto no_framerate;

  self->from_rate_numerator = rate_numerator;
  self->from_rate_denominator = rate_denominator;

  structure = gst_caps_get_structure (out_caps, 0);

  if (!gst_structure_get_fraction (structure, "framerate",
          &rate_numerator, &rate_denominator))
    goto no_framerate;

  if (self->to_rate_numerator)
    self->base_ts += GST_TENSOR_RATE_SCALED_TIME (self, self->out_frame_count);

  self->out_frame_count = 0;
  self->to_rate_numerator = rate_numerator;
  self->to_rate_denominator = rate_denominator;

  /**
   * After a setcaps, our caps may have changed. In that case, we can't use
   * the old buffer, if there was one (it might have different dimensions)
   */
  silent_debug ("swapping old buffers");
  gst_tensor_rate_swap_prev (self, NULL, GST_CLOCK_TIME_NONE);
  self->last_ts = GST_CLOCK_TIME_NONE;

  return TRUE;

no_framerate:
  silent_debug ("no framerate specified");
  return FALSE;
}

/**
 * @brief notify a frame drop event
 * @param[in] self "this" pointer
*/
static void
gst_tensor_rate_notify_drop (GstTensorRate * self)
{
  g_object_notify_by_pspec ((GObject *) self, pspec_drop);
}

/**
 * @brief notify a frame duplicate event
 * @param[in] self "this" pointer
 */
static void
gst_tensor_rate_notify_duplicate (GstTensorRate * self)
{
  g_object_notify_by_pspec ((GObject *) self, pspec_duplicate);
}

#define MAGIC_LIMIT  25
/**
 * @brief Event handler for sink pad of tensor rate.
 * @param[in] trans "this" pointer
 * @param[in] event a passed event object
 * @return TRUE if there is no error.
 */
static gboolean
gst_tensor_rate_sink_event (GstBaseTransform * trans, GstEvent * event)
{
  GstTensorRate *self = GST_TENSOR_RATE (trans);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_SEGMENT:
    {
      GstSegment segment;
      gint seqnum;

      silent_debug ("Got %s", gst_event_type_get_name (GST_EVENT_TYPE (event)));

      gst_event_copy_segment (event, &segment);
      if (segment.format != GST_FORMAT_TIME) {
        ml_loge ("Got segment but doesn't have GST_FORMAT_TIME value");
        return FALSE;
      }

      /* close up the previous segment, if appropriate */
      if (self->prevbuf) {
        gint count = 0;
        GstFlowReturn res;

        res = GST_FLOW_OK;
        /**
         * fill up to the end of current segment,
         * or only send out the stored buffer if there is no specific stop.
         * regardless, prevent going loopy in strange cases
         */
        while (res == GST_FLOW_OK && count <= MAGIC_LIMIT
            && ((GST_CLOCK_TIME_IS_VALID (self->segment.stop)
                    && GST_CLOCK_TIME_IS_VALID (self->next_ts)
                    && self->next_ts - self->segment.base <
                    self->segment.stop) || count < 1)) {
          res =
              gst_tensor_rate_flush_prev (self, count > 0, GST_CLOCK_TIME_NONE);
          count++;
        }
        if (count > 1) {
          self->dup += count - 1;
          if (!self->silent)
            gst_tensor_rate_notify_duplicate (self);
        }
        /* clean up for the new one; _chain will resume from the new start */
        gst_tensor_rate_swap_prev (self, NULL, 0);
      }

      self->base_ts = 0;
      self->out_frame_count = 0;
      self->next_ts = GST_CLOCK_TIME_NONE;

      gst_segment_copy_into (&segment, &self->segment);

      silent_debug ("updated segment: %" GST_SEGMENT_FORMAT, &self->segment);

      seqnum = gst_event_get_seqnum (event);
      gst_event_unref (event);
      event = gst_event_new_segment (&segment);
      gst_event_set_seqnum (event, seqnum);

      break;
    }
    case GST_EVENT_SEGMENT_DONE:
    case GST_EVENT_EOS:
    {
      gint count = 0;
      GstFlowReturn res = GST_FLOW_OK;

      silent_debug ("Got %s", gst_event_type_get_name (GST_EVENT_TYPE (event)));

      /* If the segment has a stop position, fill the segment */
      if (GST_CLOCK_TIME_IS_VALID (self->segment.stop)) {
        /**
         * fill up to the end of current segment,
         * or only send out the stored buffer if there is no specific stop.
         * regardless, prevent going loopy in strange cases
         */
        while (res == GST_FLOW_OK && count <= MAGIC_LIMIT
            && (GST_CLOCK_TIME_IS_VALID (self->segment.stop)
                && GST_CLOCK_TIME_IS_VALID (self->next_ts)
                && (self->next_ts - self->segment.base < self->segment.stop))) {
          res = gst_tensor_rate_flush_prev (self, count > 0,
              GST_CLOCK_TIME_NONE);
          count++;
        }
      } else if (self->prevbuf) {
        /**
         * Output at least one frame but if the buffer duration is valid, output
         * enough frames to use the complete buffer duration
         */
        if (GST_BUFFER_DURATION_IS_VALID (self->prevbuf)) {
          GstClockTime end_ts =
              self->next_ts + GST_BUFFER_DURATION (self->prevbuf);

          while (res == GST_FLOW_OK && count <= MAGIC_LIMIT &&
              ((GST_CLOCK_TIME_IS_VALID (self->segment.stop)
                      && GST_CLOCK_TIME_IS_VALID (self->next_ts)
                      && self->next_ts - self->segment.base < end_ts)
                  || count < 1)) {
            res =
                gst_tensor_rate_flush_prev (self, count > 0,
                GST_CLOCK_TIME_NONE);
            count++;
          }
        } else {
          res = gst_tensor_rate_flush_prev (self, FALSE, GST_CLOCK_TIME_NONE);
          count = 1;
        }
      }

      if (count > 1) {
        self->dup += count - 1;
        if (!self->silent)
          gst_tensor_rate_notify_duplicate (self);
      }

      break;
    }
    case GST_EVENT_FLUSH_STOP:
      /* also resets the segment */
      silent_debug ("Got %s", gst_event_type_get_name (GST_EVENT_TYPE (event)));
      gst_tensor_rate_reset (self);
      break;
    case GST_EVENT_GAP:
      /* no gaps after tensor rate, ignore the event */
      silent_debug ("Got %s", gst_event_type_get_name (GST_EVENT_TYPE (event)));
      gst_event_unref (event);
      return TRUE;
    default:
      break;
  }

  /* other events are handled in the default event handler */
  return GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event (trans, event);
}

/**
 * @brief Called when the element starts processing. optional vmethod of BaseTransform
 * @param[in] trans "this" pointer
 * @return TRUE if there is no error.
 */
static gboolean
gst_tensor_rate_start (GstBaseTransform * trans)
{
  GstTensorRate *self = GST_TENSOR_RATE (trans);
  gst_tensor_rate_reset (self);
  return TRUE;
}

/**
 * @brief Called when the element stops processing. optional vmethod of BaseTransform
 * @param[in] trans "this" pointer
 * @return TRUE if there is no error.
 */
static gboolean
gst_tensor_rate_stop (GstBaseTransform * trans)
{
  GstTensorRate *self = GST_TENSOR_RATE (trans);
  gst_tensor_rate_reset (self);
  return TRUE;
}

/**
 * @brief Installs all the properties for tensor_rate
 * @param[in] gobject_class Glib object class whose properties will be set
 */
static void
gst_tensor_rate_install_properties (GObjectClass * object_class)
{
  /* PROP_IN */
  g_object_class_install_property (object_class, PROP_IN,
      g_param_spec_uint64 ("in", "In",
          "Number of input frames",
          0, G_MAXUINT64, 0, G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  /* PROP_OUT */
  g_object_class_install_property (object_class, PROP_OUT,
      g_param_spec_uint64 ("out", "Out",
          "Number of output frames",
          0, G_MAXUINT64, 0, G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  /* PROP_DUP */
  pspec_duplicate = g_param_spec_uint64 ("duplicate", "Duplicate",
      "Number of duplicated frames", 0,
      G_MAXUINT64, 0, G_PARAM_READABLE | G_PARAM_STATIC_STRINGS);
  g_object_class_install_property (object_class, PROP_DUP, pspec_duplicate);

  /* PROP_DROP */
  pspec_drop =
      g_param_spec_uint64 ("drop", "Drop", "Number of dropped frames", 0,
      G_MAXUINT64, 0, G_PARAM_READABLE | G_PARAM_STATIC_STRINGS);
  g_object_class_install_property (object_class, PROP_DROP, pspec_drop);

  /* PROP_SILENT */
  g_object_class_install_property (object_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent",
          "Don't produce verbose output including dropped/duplicated frames",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /* PROP_THROTTLE */
  g_object_class_install_property (object_class, PROP_THROTTLE,
      g_param_spec_boolean ("throttle", "Throttle",
          "Send QoS events to upstream elements to limit a incoming data rate",
          DEFAULT_THROTTLE, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /* PROP_FRAMERATE */
  g_object_class_install_property (object_class, PROP_FRAMERATE,
      g_param_spec_string ("framerate", "Framerate",
          "Specify a target framerate to adjust (e.g., framerate=10/1). "
          "Otherwise, the latest processing time will be a target interval.",
          "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));
}

/**
 * GStreamer Tensor_Filter
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
 *
 */
/**
 * @file	tensor_filter.c
 * @date	24 May 2018
 * @brief	GStreamer plugin to use general neural network frameworks as filters
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 * @todo  set priority among properties
 * @todo  logic for dynamic properties(like model change)
 *
 * This is the main plugin for per-NN-framework plugins.
 * Specific implementations for each NN framework must be written
 * in each framework specific files; e.g., tensor_filter_tensorflow_lite.c
 *
 */

/**
 * SECTION:element-tensor_filter
 *
 * A plugin that invokes neural network models and their framework or
 * an independent shared object implementing tensor_filter_custom.h.
 * The input and output are always in the format of other/tensor or
 * other/tensors.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! tensor_filter framework=tensorflow-lite, model=./inception_v3.pb, input=3:224:224, output=1000 ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 *
 * If input is other/tensor C array input[1][224][224][3] and
 * output is other/tensor C array output[1][1][1][1000]
 *
 * The current QoS policy: In a nnstreamer pipeline, the QoS is currently satisfied
 * by adjusting a input or output framerate, initiated by 'tensor_rate' element.
 * When 'tensor_filter' receives a throttling QoS event from the 'tensor_rate' element,
 * it compares the average processing latency and throttling delay, and takes the
 * maximum value as the threshold to drop incoming frames by checking a buffer timestamp.
 * In this way, 'tensor filter' can avoid unncessary calculation and adjust a framerate,
 * effectively reducing resource utilizations.
 * Even in the case of receiving QoS events from multiple downstream pipelines (e.g., tee),
 * 'tensor_filter' takes the minimum value as the throttling delay for downstream pipeline
 * with more tight QoS requirement. Lastly, 'tensor_filter' also sends QoS events to
 * upstream elements (e.g., tensor_converter, tensor_src) to possibly reduce incoming
 * framerates, which is a better solution than dropping framerates.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <nnstreamer_util.h>

#include "tensor_filter.h"

/** @todo rename & move this to better location */
#define EVENT_NAME_UPDATE_MODEL "evt_update_model"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_filter_debug);
#define GST_CAT_DEFAULT gst_tensor_filter_debug

#define TF_MODELNAME(prop) \
    ((prop)->model_files ? ((prop)->model_files[0]) : "[No Model File]")

/**
 * @brief Default caps string for both sink and source pad.
 */
#define CAPS_STRING GST_TENSOR_CAP_DEFAULT ";" GST_TENSORS_CAP_MAKE ("{ static, flexible }")

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

#define gst_tensor_filter_parent_class parent_class
G_DEFINE_TYPE (GstTensorFilter, gst_tensor_filter, GST_TYPE_BASE_TRANSFORM);

/**
 * @brief Headroom (extra duration) added to actual latency estimate reported
 *        to LATENCY query, to limit number of updates when tracking the
 *        maximum value - arbitrarily set to 5%.
 */
#define LATENCY_REPORT_HEADROOM 0.05

/**
 * @brief Threshold deciding when tracking latency estimate that current
 *        value is sufficiently lower than reported value so that a
 *        notification update is necessary - arbitrarily set to 25%.
 */
#define LATENCY_REPORT_THRESHOLD 0.25

/* GObject vmethod implementations */
static void gst_tensor_filter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_filter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static void gst_tensor_filter_finalize (GObject * object);

/* GstBaseTransform vmethod implementations */
static GstFlowReturn gst_tensor_filter_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf);
static GstCaps *gst_tensor_filter_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter);
static GstCaps *gst_tensor_filter_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps);
static gboolean gst_tensor_filter_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_tensor_filter_query (GstBaseTransform * trans,
    GstPadDirection direction, GstQuery * query);
static gboolean gst_tensor_filter_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize);
static gboolean gst_tensor_filter_start (GstBaseTransform * trans);
static gboolean gst_tensor_filter_stop (GstBaseTransform * trans);
static gboolean gst_tensor_filter_sink_event (GstBaseTransform * trans,
    GstEvent * event);
static gboolean gst_tensor_filter_src_event (GstBaseTransform * trans,
    GstEvent * event);

/**
 * @brief initialize the tensor_filter's class
 */
static void
gst_tensor_filter_class_init (GstTensorFilterClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *trans_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_filter_debug, "tensor_filter", 0,
      "Tensor filter to invoke neural network model");

  trans_class = (GstBaseTransformClass *) klass;
  gstelement_class = (GstElementClass *) trans_class;
  gobject_class = (GObjectClass *) gstelement_class;

  gobject_class->set_property = gst_tensor_filter_set_property;
  gobject_class->get_property = gst_tensor_filter_get_property;
  gobject_class->finalize = gst_tensor_filter_finalize;

  gst_tensor_filter_install_properties (gobject_class);

  gst_element_class_set_details_simple (gstelement_class,
      "TensorFilter",
      "Filter/Tensor",
      "Handles NN Frameworks (e.g., tensorflow) as Media Filters with other/tensor type stream",
      "MyungJoo Ham <myungjoo.ham@samsung.com>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));

  /* Refer: https://gstreamer.freedesktop.org/documentation/design/element-transform.html */
  trans_class->passthrough_on_same_caps = FALSE;

  /* Processing units */
  trans_class->transform = GST_DEBUG_FUNCPTR (gst_tensor_filter_transform);

  /* Negotiation units */
  trans_class->transform_caps =
      GST_DEBUG_FUNCPTR (gst_tensor_filter_transform_caps);
  trans_class->fixate_caps = GST_DEBUG_FUNCPTR (gst_tensor_filter_fixate_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR (gst_tensor_filter_set_caps);

  /* Allocation units */
  trans_class->transform_size =
      GST_DEBUG_FUNCPTR (gst_tensor_filter_transform_size);

  /* setup events */
  trans_class->sink_event = GST_DEBUG_FUNCPTR (gst_tensor_filter_sink_event);
  trans_class->src_event = GST_DEBUG_FUNCPTR (gst_tensor_filter_src_event);

  /* start/stop to call open/close */
  trans_class->start = GST_DEBUG_FUNCPTR (gst_tensor_filter_start);
  trans_class->stop = GST_DEBUG_FUNCPTR (gst_tensor_filter_stop);

  /* Queries */
  trans_class->query = GST_DEBUG_FUNCPTR (gst_tensor_filter_query);
}

/**
 * @brief initialize the new element
 * instantiate pads and add them to element
 * set pad calback functions
 * initialize instance structure
 */
static void
gst_tensor_filter_init (GstTensorFilter * self)
{
  GstTensorFilterPrivate *priv = &self->priv;

  gst_tensor_filter_common_init_property (priv);
  /* init qos properties */
  self->prev_ts = GST_CLOCK_TIME_NONE;
  self->throttling_delay = 0;
  self->throttling_accum = 0;
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_tensor_filter_finalize (GObject * object)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;

  self = GST_TENSOR_FILTER (object);
  priv = &self->priv;

  gst_tensor_filter_common_close_fw (priv);
  gst_tensor_filter_common_free_property (priv);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Calculate tensor buffer size.
 * @param self "this" pointer
 * @param index index of tensors
 * @return tensor buffer size
 */
static gsize
gst_tensor_filter_get_tensor_size (GstTensorFilter * self, guint index,
    gboolean is_input)
{
  GstTensorFilterPrivate *priv;
  GstTensorsInfo *info;

  priv = &self->priv;
  if (is_input)
    info = &priv->prop.input_meta;
  else
    info = &priv->prop.output_meta;

  /* Internal Logic Error: out of bound */
  if (index >= info->num_tensors) {
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, FAILED,
        ("tensor_filter's core has inconsistent data. Please report to https://github.com/nnstreamer/nnstreamer/issues . The index argeument (%u) of tensors is greater-than or equal-to the number of tensors (%u)",
            index, info->num_tensors));
    return 0;
  }

  return gst_tensor_info_get_size (&info->info[index]);
}

/**
 * @brief Setter for tensor_filter properties.
 */
static void
gst_tensor_filter_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;

  self = GST_TENSOR_FILTER (object);
  priv = &self->priv;

  silent_debug (self, "Setting property for prop %d.\n", prop_id);

  if (!gst_tensor_filter_common_set_property (priv, prop_id, value, pspec))
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
}

/**
 * @brief Getter for tensor_filter properties.
 */
static void
gst_tensor_filter_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;

  self = GST_TENSOR_FILTER (object);
  priv = &self->priv;

  silent_debug (self, "Getting property for prop %d.\n", prop_id);

  if (!gst_tensor_filter_common_get_property (priv, prop_id, value, pspec))
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
}

/**
 * @brief Free the data allocated for tensor transform
 * @details default function for tensor filter framework if not provided by the
 *          framework. The data is in GPtrArray - first element is private data
 *          of framework and second element is the data to be freed.
 */
static void
gst_tensor_filter_destroy_notify (void *data)
{
  GPtrArray *array = (GPtrArray *) data;
  GstTensorFilter *self = (GstTensorFilter *) g_ptr_array_index (array, 0);
  void *tensor_data = (void *) g_ptr_array_index (array, 1);
  g_ptr_array_free (array, TRUE);

  gst_tensor_filter_destroy_notify_util (&self->priv, tensor_data);
}

/**
 * @brief Allocate new memory block from given data.
 * @details tensor-filter should send event to sub-plugin when memory is freed.
 */
static GstMemory *
gst_tensor_filter_get_wrapped_mem (GstTensorFilter * self, gpointer data,
    gsize size)
{
  GPtrArray *data_array = g_ptr_array_new ();

  g_ptr_array_add (data_array, (gpointer) self);
  g_ptr_array_add (data_array, (gpointer) data);

  return gst_memory_new_wrapped (0, data, size, 0, size, (gpointer) data_array,
      gst_tensor_filter_destroy_notify);
}

/**
 * @brief Prepare statistics for performance profiling (e.g, latency, throughput)
 */
static void
prepare_statistics (GstTensorFilterPrivate * priv)
{
  priv->stat.latest_invoke_time = g_get_real_time ();
}

/**
 * @brief Helper function to accumulate latencies
 */
static void
accumulate_latency (void *data, void *user_data)
{
  gint64 *latency = data;
  gint64 *total_latency = user_data;

  *total_latency += *latency;
}

#define THRESHOLD_DROP_OLD  (2000)
#define THRESHOLD_CACHE_OLD (1000)

/**
 * @brief Record statistics for performance profiling (e.g, latency, throughput)
 */
static void
record_statistics (GstTensorFilterPrivate * priv)
{
  gint64 end_time = g_get_real_time ();
  gint64 *latency;
  GQueue *recent_latencies = priv->stat.recent_latencies;

  /* ignore first measurements that may be off */
  if (priv->stat.latency_ignore_count) {
    priv->stat.latency_ignore_count--;
    return;
  }

  latency = g_new (gint64, 1);
  *latency = end_time - priv->stat.latest_invoke_time;
  priv->stat.total_invoke_latency += *latency;
  priv->stat.total_invoke_num += 1;

  if (g_queue_get_length (recent_latencies) == GST_TF_STAT_MAX_RECENT)
    g_free (g_queue_pop_head (recent_latencies));
  g_queue_push_tail (recent_latencies, latency);

  /* the queue should have at least one element */
  g_assert (g_queue_get_length (recent_latencies) != 0);

  if (priv->latency_mode > 0 || priv->latency_reporting) {
    gint64 avg_latency = 0;

    g_queue_foreach (recent_latencies, accumulate_latency, &avg_latency);
    avg_latency /= g_queue_get_length (recent_latencies);

    /* check integer overflow */
    if (avg_latency <= INT32_MAX)
      priv->prop.latency = (gint) avg_latency;
    else
      priv->prop.latency = -1;

    ml_logi ("[%s] Invoke took %.3f ms", TF_MODELNAME (&(priv->prop)),
        (*latency) / 1000.0);
  }

  if (priv->throughput_mode > 0) {
    gint throughput_int = -1;

    if (priv->stat.total_invoke_latency != 0) {
      gdouble throughput =
          (gdouble) (priv->stat.total_invoke_num * G_USEC_PER_SEC * 1000) /
          priv->stat.total_invoke_latency;

      /* check integer overflow */
      if (throughput <= INT32_MAX)
        throughput_int = (gint) throughput;
    }

    /* note that it's a 1000x larger value than actual throughput */
    priv->prop.throughput = throughput_int;

    ml_logi ("[%s] Throughput: %.2f FPS", TF_MODELNAME (&(priv->prop)),
        throughput_int / 1000.0);
  }

  /**
   * statistics values are monotonously increasing.
   * to avoid potential overflow, let's cache old values and subtract them
   * from the statistics if some threshold is exceeded.
   */
  if (priv->stat.total_invoke_num > THRESHOLD_DROP_OLD) {
    priv->stat.total_invoke_latency -= priv->stat.old_total_invoke_latency;
    priv->stat.total_invoke_num -= priv->stat.old_total_invoke_num;
    /* drop cached values */
    priv->stat.old_total_invoke_latency = 0;
    priv->stat.old_total_invoke_num = 0;
  } else if (priv->stat.total_invoke_num > THRESHOLD_CACHE_OLD) {
    /* cache old values if they are not yet set */
    if (priv->stat.old_total_invoke_num == 0) {
      priv->stat.old_total_invoke_latency = priv->stat.total_invoke_latency;
      priv->stat.old_total_invoke_num = priv->stat.total_invoke_num;
    }
  }
}


/**
 * @brief Track estimated latency and notify pipeline when it changes.
 *        Latency estimates may be a bit jittery. On the principle we want to
 *        inform pipeline with the latency from longest inference.
 *        However, first inference may take much longer, or model filter
 *        configuration may change. Therefore any change of more than 10%
 *        (arbitrary value) to a lower latency is also reported to pipeline.
 *        Notification is done sending LATENCY message to bus. Upon receipt,
 *        application will initiate a pipeline latency probe via LATENCY query.
 */
static void
track_latency (GstTensorFilter * self)
{
  GstTensorFilterPrivate *priv = &self->priv;
  gdouble estimated, reported, deviation;

  GST_OBJECT_LOCK (self);
  estimated = priv->prop.latency * GST_USECOND;
  reported = priv->latency_reported;
  GST_OBJECT_UNLOCK (self);

  if ((priv->latency_reporting) && (estimated > 0)) {
    if (reported > 0)
      deviation = ABS (estimated - reported) / reported;
    else
      deviation = 0;

    if ((estimated > reported) || (deviation > LATENCY_REPORT_THRESHOLD)) {

      ml_logd ("[%s] latency reported:%.0f estimated:%.0f deviation:%.4f",
          TF_MODELNAME (&(priv->prop)), reported, estimated, deviation);

      gst_element_post_message (GST_ELEMENT_CAST (self),
          gst_message_new_latency (GST_OBJECT_CAST (self)));
    }
  }
}

/**
 * @brief Check throttling delay and send qos overflow event to upstream elements
 */
static gboolean
gst_tensor_filter_check_throttling_delay (GstBaseTransform * trans,
    GstBuffer * inbuf)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;

  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;

  GST_OBJECT_LOCK (trans);

  if (self->throttling_delay != 0) {
    GstClockTime curr_ts = GST_BUFFER_PTS (inbuf);
    GstClockTime prev_ts = self->prev_ts;

    self->prev_ts = curr_ts;

    if (GST_CLOCK_TIME_IS_VALID (prev_ts)) {
      GstClockTimeDiff diff = curr_ts - prev_ts;
      GstClockTimeDiff delay;

      self->throttling_accum += diff;

      /* check whether the average latency is longer than throttling delay */
      delay = MAX (priv->prop.latency * 1000, self->throttling_delay);

      if (self->throttling_accum < delay) {
        GstClockTimeDiff duration = GST_BUFFER_DURATION (inbuf);        /* original */
        gdouble avg_rate = gst_guint64_to_gdouble (duration) /
            gst_guint64_to_gdouble (delay);

        /**
         * Send qos overflow event to upstream elements.
         * Upstream elements (e.g., tensor_src, tensor_converter) may handle this.
         */
        GstPad *sinkpad = GST_BASE_TRANSFORM_SINK_PAD (&self->element);
        GstEvent *event = gst_event_new_qos (GST_QOS_TYPE_OVERFLOW,
            avg_rate, (self->throttling_accum - delay), curr_ts);

        gst_pad_push_event (sinkpad, event);

        GST_OBJECT_UNLOCK (trans);
        return TRUE;
      }

      self->throttling_accum = 0;
    }
  }

  GST_OBJECT_UNLOCK (trans);
  return FALSE;
}

/**
 * @brief Check input paramters for gst_tensor_filter_transform ();
 */
static GstFlowReturn
_gst_tensor_filter_transform_validate (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstTensorFilter *self = GST_TENSOR_FILTER_CAST (trans);
  GstTensorFilterPrivate *priv = &self->priv;
  GstTensorFilterProperties *prop = &priv->prop;

  if (G_UNLIKELY (!priv->configured)) {
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, TYPE_NOT_FOUND,
        ("The tensor_filter instance is not configured (pad caps not negotiated). Property info (framework = '%s', framework_opened = %d, model[0] = '%s', num-models = %d, custom_properties = '%s'.",
            prop ? prop->fwname : "property info is NULL.",
            prop ? prop->fw_opened : -1,
            prop ? TF_MODELNAME (prop) : "property info is NULL.",
            prop ? prop->num_models : -1,
            prop ? prop->custom_properties : "property info is NULL."));
    return GST_FLOW_NOT_NEGOTIATED;
  }
  if (G_UNLIKELY (!priv->fw)) {
    /**
      * This is fatal; if framework is not configured until this stage,
      * it means that an extension is missing or not configured.
      * We need readable messages for non-developers
      */
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, FAILED,
        ("Framework (filter subplugin) is not found or not configured: 'framework=%s'. Please check if the given framework name is correct and the given model path is consistent with the intended framework especially if 'framework=auto' is given. Please refer to the warning messages created when the framework property is set.",
            priv->prop.fwname));
    ml_logf
        ("\nA corresponding nnstreamer extension (tensor_filter subplugin) is not installed or the framework property of tensor_filter is incorrect: [%s] is not found.\n\n",
        prop->fwname);
    return GST_FLOW_ERROR;
  }
  if (G_UNLIKELY (!priv->fw->run_without_model) &&
      G_UNLIKELY (!(prop->model_files &&
              prop->num_models > 0 && prop->model_files[0]))) {
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, FAILED,
        ("For the framework='%s', its model filepath is not provided and this framework requires a model file. Thus, we cannot proceed with tensor_filter for inferences. Please provide a valid model file path.",
            prop->fwname));
    return GST_FLOW_ERROR;
  }
  if ((GST_TF_FW_V0 (priv->fw) && G_UNLIKELY (!priv->fw->invoke_NN)) ||
      (GST_TF_FW_V1 (priv->fw) && G_UNLIKELY (!priv->fw->invoke))) {
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, FAILED,
        ("The tensor-filter subplugin for the framework='%s' does not have its mandatory methods (or callback functions). It appears that your subplugin implementation of '%s' is not completed. There is no 'invoke_NN (v1)' or 'invoke (v2)' methods available.",
            prop->fwname, prop->fwname));
    return GST_FLOW_ERROR;
  }

  silent_debug (self, "Invoking %s with %s model\n", priv->fw->name,
      GST_STR_NULL (prop->model_files[0]));

  /* skip input data when throttling delay is set */
  if (gst_tensor_filter_check_throttling_delay (trans, inbuf))
    return GST_BASE_TRANSFORM_FLOW_DROPPED;

  if (!outbuf) {
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, FAILED,
        ("The output buffer for the instance of tensor-filter subplugin (%s / %s) is null. Cannot proceed.",
            prop->fwname, TF_MODELNAME (prop)));
    return GST_FLOW_ERROR;
  }
  if (gst_buffer_get_size (outbuf) != 0) {
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, FAILED,
        ("The output buffer for the isntance of tensor-filter subplugin (%s / %s) already has a content (buffer size = %zu). It should be 0.",
            prop->fwname, TF_MODELNAME (prop), gst_buffer_get_size (outbuf)));
    return GST_FLOW_ERROR;
  }

  return GST_FLOW_OK;
}

/**
 * @brief non-ip transform. required vmethod of GstBaseTransform.
 */
static GstFlowReturn
gst_tensor_filter_transform (GstBaseTransform * trans,
    GstBuffer * inbuf, GstBuffer * outbuf)
{
  GstTensorFilter *self = GST_TENSOR_FILTER_CAST (trans);
  GstTensorFilterPrivate *priv = &self->priv;
  GstTensorFilterProperties *prop = &priv->prop;
  GstMemory *in_mem[NNS_TENSOR_SIZE_LIMIT] = { 0, };
  GstMapInfo in_info[NNS_TENSOR_SIZE_LIMIT];
  GstMemory *out_mem[NNS_TENSOR_SIZE_LIMIT] = { 0, };
  GstMapInfo out_info[NNS_TENSOR_SIZE_LIMIT];
  GstTensorMemory in_tensors[NNS_TENSOR_SIZE_LIMIT];
  GstTensorMemory invoke_tensors[NNS_TENSOR_SIZE_LIMIT];
  GstTensorMemory out_tensors[NNS_TENSOR_SIZE_LIMIT];
  GList *list;
  guint i, num_mems;
  gint ret;
  gboolean allocate_in_invoke, in_flexible, out_flexible;
  gboolean need_profiling;
  gsize expected, hsize;

  GstTensorMetaInfo in_meta[NNS_TENSOR_SIZE_LIMIT];
  GstTensorMetaInfo out_meta[NNS_TENSOR_SIZE_LIMIT];
  GstMemory *mem;

  /* 0. Check all properties. */
  GstFlowReturn retval = _gst_tensor_filter_transform_validate (trans, inbuf,
      outbuf);
  if (retval != GST_FLOW_OK)
    return retval;

  allocate_in_invoke = gst_tensor_filter_allocate_in_invoke (priv);

  in_flexible =
      gst_tensor_pad_caps_is_flexible (GST_BASE_TRANSFORM_SINK_PAD (trans));
  out_flexible =
      gst_tensor_pad_caps_is_flexible (GST_BASE_TRANSFORM_SRC_PAD (trans));

  /* 1. Get all input tensors from inbuf. */
  /* Internal Logic Error or GST Bug (sinkcap changed!) */
  num_mems = gst_buffer_n_memory (inbuf);

  for (i = 0; i < num_mems; i++) {
    in_mem[i] = gst_buffer_peek_memory (inbuf, i);
    if (!gst_memory_map (in_mem[i], &in_info[i], GST_MAP_READ)) {
      ml_logf_stacktrace
          ("gst_tensor_filter_transform: For the given input buffer, tensor-filter (%s : %s) cannot map input memory from the buffer for reading. The %u-th memory chunk (%u-th tensor) has failed for memory map.\n",
          prop->fwname, TF_MODELNAME (prop), i, i);
      goto mem_map_error;
    }

    hsize = 0;
    if (in_flexible) {
      gst_tensor_meta_info_parse_header (&in_meta[i], in_info[i].data);
      hsize = gst_tensor_meta_info_get_header_size (&in_meta[i]);
    }

    in_tensors[i].data = in_info[i].data + hsize;
    in_tensors[i].size = in_info[i].size - hsize;
  }

  /* 1.1 Prepare tensors to invoke. */
  if (priv->combi.in_combi_defined) {
    guint info_idx = 0;

    for (list = priv->combi.in_combi; list != NULL; list = list->next) {
      i = GPOINTER_TO_UINT (list->data);

      if (i >= num_mems) {
        ml_loge_stacktrace
            ("gst_tensor_filter_transform: Invalid input combination ('input-combination' property) for the tensor-filter (%s:%s). The %u'th combination's index is %u, which is out of bound (>= %u = the number of memory chunks (tensors) of incoming buffer). Because of buffer index inconsistency, it cannot continue (cannot map the memory for the input buffer).\n",
            prop->fwname, TF_MODELNAME (prop), info_idx, i, num_mems);
        goto mem_map_error;
      }

      expected = gst_tensor_filter_get_tensor_size (self, info_idx, TRUE);
      if (expected != in_tensors[i].size) {
        ml_loge_stacktrace
            ("gst_tensor_filter_transform: With the given input combination ('input-combination' property) of the tensor-filter, the incoming buffer size of combination index %u (%u'th combination) is %zd, which is invalid and is expected to be %zd. Because of buffer size inconsistency, it cannot continue (cannot map the memory for the input buffer).\n",
            i, info_idx, in_tensors[i].size, expected);
        goto mem_map_error;
      }

      invoke_tensors[info_idx++] = in_tensors[i];
    }
  } else {
    if (num_mems != prop->input_meta.num_tensors) {
      ml_loge_stacktrace
          ("gst_tensor_filter_transform: Input buffer has invalid number of memory blocks (%u), which is expected to be %u (the number of tensors). Maybe, the pad capability is not consistent with the actual input stream.\n",
          num_mems, prop->input_meta.num_tensors);
      goto mem_map_error;
    }

    for (i = 0; i < prop->input_meta.num_tensors; i++) {
      expected = gst_tensor_filter_get_tensor_size (self, i, TRUE);
      if (expected != in_tensors[i].size) {
        ml_loge_stacktrace
            ("gst_tensor_filter_transform: Input buffer size (%u'th memory chunk: %zd) is invalid, which is expected to be %zd, which is the frame size of the corresponding tensor. Maybe, the pad capability is not consistent with the actual input stream; if the size is supposed to change dynamically and the given neural network, framework, and the subpluigins can handle it, please consider using format=flexible.\n",
            i, in_tensors[i].size, expected);
        goto mem_map_error;
      }

      invoke_tensors[i] = in_tensors[i];
    }
  }

  /* 2. Prepare output tensors. */
  for (i = 0; i < prop->output_meta.num_tensors; i++) {
    out_tensors[i].data = NULL;
    out_tensors[i].size = gst_tensor_filter_get_tensor_size (self, i, FALSE);

    hsize = 0;
    if (out_flexible) {
      gst_tensor_info_convert_to_meta (&prop->output_meta.info[i],
          &out_meta[i]);
      hsize = gst_tensor_meta_info_get_header_size (&out_meta[i]);
    }

    /* allocate memory if allocate_in_invoke is FALSE */
    if (!allocate_in_invoke) {
      out_mem[i] =
          gst_allocator_alloc (NULL, out_tensors[i].size + hsize, NULL);
      if (!out_mem[i]) {
        ml_loge_stacktrace
            ("gst_tensor_filter_transform: cannot allocate memory for the output buffer (%u'th memory chunk for %u'th tensor), which requires %zd bytes. gst_allocate_alloc has returned Null. Out of memory?",
            i, i, out_tensors[i].size + hsize);
        goto mem_map_error;
      }
      if (!gst_memory_map (out_mem[i], &out_info[i], GST_MAP_WRITE)) {
        ml_loge_stacktrace
            ("gst_tensor_filter_transform: For the given output buffer, allocated by gst_tensor_filter_transform, it cannot map output memory buffer for the %u'th memory chunk (%u'th output tensor) for write.\n",
            i, i);
        goto mem_map_error;
      }

      out_tensors[i].data = out_info[i].data + hsize;

      /* append header */
      if (out_flexible) {
        if (FALSE == gst_tensor_meta_info_update_header
            (&out_meta[i], out_info[i].data)) {
          ml_loge_stacktrace
              ("gst_tensor_meta_info_update_header() has failed to update header for flexible format: invalid metadata or buffer for header is not available. This looks like an internal error of nnstreamer/tensor_filter. Please report to github.com/nnstreamer/nnstreamer/issues. %u'th output buffer has failed to update its header.\n",
              i);
          goto mem_map_error;
        }
      }
    }
  }

  need_profiling = (priv->latency_mode > 0 || priv->throughput_mode > 0 ||
      priv->latency_reporting);
  if (need_profiling)
    prepare_statistics (priv);

  /* 3. Call the filter-subplugin callback, "invoke" */
  GST_TF_FW_INVOKE_COMPAT (priv, ret, invoke_tensors, out_tensors);
  if (need_profiling) {
    record_statistics (priv);
    track_latency (self);
  }

  /* 4. Free map info and handle error case */
  for (i = 0; i < num_mems; i++)
    gst_memory_unmap (in_mem[i], &in_info[i]);

  if (!allocate_in_invoke) {
    for (i = 0; i < prop->output_meta.num_tensors; i++) {
      gst_memory_unmap (out_mem[i], &out_info[i]);
      if (ret != 0)
        gst_allocator_free (out_mem[i]->allocator, out_mem[i]);
    }
  }

  /** @todo define enum to indicate status code */
  if (ret < 0) {
    ml_loge_stacktrace
        ("Calling invoke function (inference instance) of the tensor-filter subplugin (%s for %s) has failed with error code (%d).\n",
        prop->fwname, TF_MODELNAME (prop), ret);
    return GST_FLOW_ERROR;
  } else if (ret > 0) {
    /* drop this buffer */
    return GST_BASE_TRANSFORM_FLOW_DROPPED;
  }

  /* 5. Update result */
  /* If output combination is defined, append input tensors first */
  if (priv->combi.out_combi_i_defined) {
    for (list = priv->combi.out_combi_i; list != NULL; list = list->next) {
      i = GPOINTER_TO_UINT (list->data);

      if (!in_flexible && out_flexible) {
        /* append header */
        gst_tensor_info_convert_to_meta (&priv->in_config.info.info[i],
            &in_meta[i]);
        mem = gst_tensor_meta_info_append_header (&in_meta[i], in_mem[i]);
      } else if (in_flexible && !out_flexible) {
        /* remove header */
        hsize = gst_tensor_meta_info_get_header_size (&in_meta[i]);
        mem = gst_memory_share (in_mem[i], hsize, -1);
      } else {
        mem = gst_memory_ref (in_mem[i]);
      }

      gst_buffer_append_memory (outbuf, mem);
    }
  }

  for (i = 0; i < prop->output_meta.num_tensors; i++) {
    if (priv->combi.out_combi_o_defined) {
      gboolean out_combi = FALSE;

      for (list = priv->combi.out_combi_o; list != NULL; list = list->next) {
        if (i == GPOINTER_TO_UINT (list->data)) {
          out_combi = TRUE;
          break;
        }
      }
      if (!out_combi) {
        /* release memory block if output tensor is not in the combi list */
        if (allocate_in_invoke) {
          gst_tensor_filter_destroy_notify_util (priv, out_tensors[i].data);
        } else {
          gst_allocator_free (out_mem[i]->allocator, out_mem[i]);
        }

        continue;
      }
    }

    if (allocate_in_invoke) {
      /* prepare memory block if successfully done */
      out_mem[i] = mem = gst_tensor_filter_get_wrapped_mem (self,
          out_tensors[i].data, out_tensors[i].size);

      if (out_flexible) {
        /* prepare new memory block with meta */
        out_mem[i] = gst_tensor_meta_info_append_header (&out_meta[i], mem);
        gst_memory_unref (mem);
      }
    }

    /* append the memory block to outbuf */
    gst_buffer_append_memory (outbuf, out_mem[i]);
  }

  return GST_FLOW_OK;
mem_map_error:
  num_mems = gst_buffer_n_memory (inbuf);
  for (i = 0; i < num_mems; i++) {
    if (in_mem[i])
      gst_memory_unmap (in_mem[i], &in_info[i]);
  }

  if (!allocate_in_invoke) {
    for (i = 0; i < prop->output_meta.num_tensors; i++) {
      if (out_mem[i]) {
        gst_memory_unmap (out_mem[i], &out_info[i]);
        gst_allocator_free (out_mem[i]->allocator, out_mem[i]);
      }
    }
  }
  return GST_FLOW_ERROR;
}

/**
 * @brief Configure input and output tensor info from incaps.
 * @param self "this" pointer
 * @param incaps received caps for sink pad
 * @return TRUE if fully configured
 */
static gboolean
gst_tensor_filter_configure_tensor (GstTensorFilter * self,
    const GstCaps * incaps)
{
  GstTensorFilterPrivate *priv;
  GstTensorFilterProperties *prop;
  GstStructure *structure;
  GstTensorsConfig in_config, out_config;
  GstTensorsInfo in_info, out_info;
  gboolean flexible;

  g_return_val_if_fail (incaps != NULL, FALSE);

  priv = &self->priv;
  prop = &priv->prop;
  gst_tensors_config_init (&in_config);
  gst_tensors_config_init (&out_config);
  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  /**
   * GstTensorFilter has to parse the tensor dimension and type from NN model.
   * 1. Call functions getInputDimension and getOutputDimension to get the dimension and type.
   * 2. If these functions are not defined, call setInputDimension with parsed info from caps.
   * 3. If set-prop configured dimension, verify the dimension with fw callbacks.
   */
  gst_tensor_filter_load_tensor_info (&self->priv);

  structure = gst_caps_get_structure (incaps, 0);
  if (G_UNLIKELY (!structure)) {
    gchar *capstr = gst_caps_to_string (incaps);
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, WRONG_TYPE,
        ("%s:%u The input stream padcap is not appropriate. Cannot get structured information from the given padcap: '%s' (gst_caps_get_structure() has returned NULL.). Please check if the caps you've given is correct especially if you have applied caps-filter in front of tensor-filter (%s:%s).",
            __func__, __LINE__, capstr, GST_STR_NULL (prop->fwname),
            TF_MODELNAME (prop)));
    g_free (capstr);
    goto done;
  }
  if (G_UNLIKELY (!gst_tensors_config_from_structure (&in_config, structure))) {
    gchar *capstr = gst_caps_to_string (incaps);
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, WRONG_TYPE,
        ("%s:%u The input stream padcap is not appropriate. It is not a valid tensor stream (other/tensors) or a tensor stream with a few required entries missing. For example, dimesion and type are missing for a static tensor (format=static or other/tensor (single tensor)). The given padcap is '%s' for tensor-filter (%s:%s).",
            __func__, __LINE__, capstr, GST_STR_NULL (prop->fwname),
            TF_MODELNAME (prop)));
    g_free (capstr);
    goto done;
  }

  /**
   * Check configuration from caps.
   * If true, fully configured tensor info from caps.
   */
  if (!gst_tensors_config_validate (&in_config)) {
    gchar *capstr = gst_caps_to_string (incaps);
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, WRONG_TYPE,
        ("%s:%u The input stream padcaps cannot be validated. It is not a valid tensor stream for tensor_filter element. Input stream type for tensor_filter (%s:%s) is %s. tensor-filter could get config from the input padcap; however, it cannot be validated: framerate or format is not valid. Try a static tensor stream with a valid (0/1 is also valid!) framerate.",
            __func__, __LINE__, GST_STR_NULL (prop->fwname),
            TF_MODELNAME (prop), capstr));
    g_free (capstr);
    goto done;
  }

  if (!gst_tensor_filter_common_get_combined_in_info (priv, &in_config.info,
          &in_info)) {
    gchar *capstr = gst_caps_to_string (incaps);
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, WRONG_TYPE,
        ("%s:%u Failed to configure combined input info for tensor-filter (%s:%s). The given padcap is '%s'. User has specified input combination (refer to the previous error log), which shuffles the order of tensors, which is not compatible with the given model. Please check the padcap, input combination, and tensor/dimension requirement of your neural network model.",
            __func__, __LINE__, GST_STR_NULL (prop->fwname),
            TF_MODELNAME (prop), capstr));
    g_free (capstr);
    goto done;
  }

  /* flexible tensor case, we cannot get the exact info from caps. */
  flexible = gst_tensors_config_is_flexible (&in_config);

  /** if set-property called and already has info, verify it! */
  if (prop->input_meta.num_tensors > 0) {
    if (flexible) {
      /**
       * If incoming tensor is flexible, we cannot validate tensor info here.
       * Need to compare buffer size in transform().
       */
      GST_INFO_OBJECT (self, "The input tensor is flexible.");
    } else if (!gst_tensors_info_is_equal (&in_info, &prop->input_meta)) {
      gchar *capstr = gst_caps_to_string (incaps);
      gchar *compare =
          gst_tensorsinfo_compare_to_string (&in_info, &prop->input_meta);
      GST_ELEMENT_ERROR_BTRACE (self, STREAM, WRONG_TYPE,
          ("%s:%u The input tensor of tensor_filter (%s:%s) is not compatible with the configured input information. Please check tensor-filter properties if you have given input dimensions explicitly; or the model properties if you have not given them. Check the input stream caps and related caps-filters, too. The given gstcap is %s, which is not compatible: %s",
              __func__, __LINE__, GST_STR_NULL (prop->fwname),
              TF_MODELNAME (prop), capstr, compare));
      g_free (compare);
      g_free (capstr);
      goto done;
    }
  } else {
    if (flexible) {
      gchar *capstr = gst_caps_to_string (incaps);
      /** @todo
       * We do not support this (flexible tensor for flexible input model).
       * Cap-negotiation of the current tensor-filter requires either side of
       * "model / set-property" or "incoming gstcaps" to be static/explicit.
       * Ideally, this should support flexible tensor for flexible input model,
       * leaving the negotiation to other elements, but we didn't implement it yet.
       */
      GST_ELEMENT_ERROR_BTRACE (self, STREAM, WRONG_TYPE,
          ("%s:%u The input tensor of tensor_filter (%s:%s) is flexible (gstcap: '%s'), which requires either explicit type/dimension tensor-filter property values or static input dimension models. The current version of tensor-filter does not support flexible tensors for dynamic-input models without explicit input dimension declaration. Declare type/dimension/tensors with tensor-filter properties, or apply caps-filter in front of tensor-filter, or use static tensor streams.",
              __func__, __LINE__, GST_STR_NULL (prop->fwname),
              TF_MODELNAME (prop), capstr));
      g_free (capstr);
      goto done;
    } else {
      gst_tensors_info_copy (&prop->input_meta, &in_info);
    }
  }

  prop->input_configured = TRUE;

  /** call setInputDimension if output tensor is not configured */
  if (!prop->output_configured) {
    if (gst_tensor_filter_common_get_out_info (priv, &prop->input_meta,
            &out_info)) {
      /** if set-property called and already has info, verify it! */
      if (prop->output_meta.num_tensors > 0) {
        if (!gst_tensors_info_is_equal (&out_info, &prop->output_meta)) {
          gchar *cmpstr = gst_tensorsinfo_compare_to_string (&out_info,
              &prop->output_meta);
          GST_ELEMENT_ERROR_BTRACE (self, STREAM, WRONG_TYPE,
              ("%s:%u The output tensor is not compatible with the configured tensor information. The configuration is usually set by tensor-filter properties declared by users or the given neural network itself. The following two tensor metadata are not compatible: %s.\n",
                  __func__, __LINE__, cmpstr));
          g_free (cmpstr);
          gst_tensors_info_free (&out_info);
          goto done;
        }
      } else {
        gst_tensors_info_copy (&prop->output_meta, &out_info);
      }

      prop->output_configured = TRUE;
    }

    if (!prop->output_configured) {
      GST_ELEMENT_ERROR_BTRACE (self, STREAM, WRONG_TYPE,
          ("%s:%u Failed to get output tensor info: not enough related information to configure output.\n",
              __func__, __LINE__));
      goto done;
    }
  }

  /**
   * @todo framerate of output tensors
   * How can we update the framerate?
   * GstTensorFilter cannot assure the framerate.
   * Simply set the framerate of out-tensor from incaps.
   */
  out_config.rate_n = in_config.rate_n;
  out_config.rate_d = in_config.rate_d;

  if (!gst_tensor_filter_common_get_combined_out_info (priv, &in_config.info,
          &prop->output_meta, &out_config.info)) {
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, WRONG_TYPE,
        ("%s:%u Failed to configure combined output info: please refer to the error message of gst_tensor_filter_common_get_combined_out_info(). ",
            __func__, __LINE__));
    goto done;
  }

  if (priv->configured) {
    /** already configured, compare to old. */
    g_assert (gst_tensors_config_is_equal (&priv->in_config, &in_config));
    g_assert (gst_tensors_config_is_equal (&priv->out_config, &out_config));
  } else {
    gst_tensors_config_copy (&priv->in_config, &in_config);
    gst_tensors_config_copy (&priv->out_config, &out_config);

    priv->configured = TRUE;
  }

done:
  gst_tensors_config_free (&in_config);
  gst_tensors_config_free (&out_config);
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
  return priv->configured;
}

/**
 * @brief configure tensor-srcpad cap from "proposed" cap.
 *
 * @trans ("this" pointer)
 * @direction (why do we need this?)
 * @caps sinkpad cap (if direction GST_PAD_SINK)
 * @filter this element's cap (don't know specifically.)
 *
 * Be careful not to fix/set caps at this stage. Negotiation not completed yet.
 */
static GstCaps *
gst_tensor_filter_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;
  GstTensorFilterProperties *prop;
  GstTensorsConfig in_config, out_config;
  GstPad *pad;
  GstCaps *result;
  GstCaps *peer_caps;
  GstStructure *structure;
  gboolean configured = FALSE;

  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;
  prop = &priv->prop;

  /* Not ready */
  if (priv->fw == NULL)
    return NULL;

  gst_tensors_config_init (&in_config);
  gst_tensors_config_init (&out_config);

  silent_debug_caps (self, caps, "from");
  silent_debug_caps (self, filter, "filter");

  if (direction == GST_PAD_SINK)
    pad = GST_BASE_TRANSFORM_SRC_PAD (trans);
  else
    pad = GST_BASE_TRANSFORM_SINK_PAD (trans);

  /**
   * GstTensorFilter has to parse the tensor dimension and type from NN model.
   * In this stage, in-caps may not be fixed yet.
   * To get the tensor info and generate pad-caps, call getInputDimension and getOutputDimension.
   * If these functions are not defined, we have to call setInputDimension, and then it will fully configure the tensor info.
   *
   * @todo how to set the framerate of output tensors
   */
  gst_tensor_filter_load_tensor_info (&self->priv);

  structure = gst_caps_get_structure (caps, 0);
  gst_tensors_config_from_structure (&in_config, structure);

  /* set framerate from input config */
  out_config.rate_n = in_config.rate_n;
  out_config.rate_d = in_config.rate_d;

  if (direction == GST_PAD_SINK) {
    GstTensorsInfo out_info;

    gst_tensors_info_init (&out_info);

    /* caps: sink pad. get src pad info */
    if (prop->output_configured) {
      /* caps with sub-plugin's tensor info */
      gst_tensors_info_copy (&out_info, &prop->output_meta);
      configured = TRUE;
    } else {
      /* check in-tensor info to call setInputDimension */
      configured = gst_tensor_filter_common_get_out_info (priv,
          &in_config.info, &out_info);
    }

    /* If output combination option is given, reconfigure tensor info */
    if (configured)
      configured = gst_tensor_filter_common_get_combined_out_info (priv,
          &in_config.info, &out_info, &out_config.info);

    gst_tensors_info_free (&out_info);
  } else {
    /* caps: src pad. get sink pad info */
    if (prop->input_configured && !priv->combi.in_combi_defined) {
      /* caps with sub-plugin's tensor info */
      gst_tensors_info_copy (&out_config.info, &prop->input_meta);
      configured = TRUE;
    }
  }

  if (configured) {
    /* output info may be configured */
    result = gst_tensor_pad_possible_caps_from_config (pad, &out_config);
  } else {
    /* we don't know the exact tensor info yet */
    result = gst_caps_from_string (CAPS_STRING);
  }

  /* Update caps dimension for src pad cap */
  if (direction == GST_PAD_SINK) {
    if ((peer_caps = gst_pad_peer_query_caps (pad, NULL))) {
      gst_tensor_caps_update_dimension (result, peer_caps);
      gst_caps_unref (peer_caps);
    }
  }

  if (filter && gst_caps_get_size (filter) > 0) {
    GstCaps *intersection;

    /**
     * @todo We do not have a testcase hitting here. Thus, we do not ensure the validity here.
     * However, according to gstreamer doxygen entry, if filter is given, that's not to be ignored.
     * For now, we assume that if caps-size is 0, filter is "ANY".
     */

    intersection =
        gst_caps_intersect_full (result, filter, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (result);
    result = intersection;
  }

  silent_debug_caps (self, result, "to");
  gst_tensors_config_free (&in_config);
  gst_tensors_config_free (&out_config);
  return result;
}

/**
 * @brief fixate caps. required vmethod of GstBaseTransform.
 */
static GstCaps *
gst_tensor_filter_fixate_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * othercaps)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;
  GstCaps *result;
  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;

  silent_debug (self, "fixate_caps, direction = %d\n", direction);
  silent_debug_caps (self, caps, "caps");
  silent_debug_caps (self, othercaps, "othercaps");

  /** Removes no-used-variable warning for priv in when DBG is set */
  if (priv->fw == NULL) {
    gst_caps_unref (othercaps);
    return NULL;
  }

  /**
   * To get the out-caps, GstTensorFilter has to parse tensor info from NN model.
   */

  result = gst_tensor_filter_transform_caps (trans, direction, caps, othercaps);
  gst_caps_unref (othercaps);
  result = gst_caps_make_writable (result);
  result = gst_caps_fixate (result);

  silent_debug_caps (self, result, "result");
  return result;
}

/**
 * @brief set caps. required vmethod of GstBaseTransform.
 */
static gboolean
gst_tensor_filter_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;
  GstStructure *structure;
  GstTensorsConfig config;

  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;

  silent_debug_caps (self, incaps, "incaps");
  silent_debug_caps (self, outcaps, "outcaps");

  if (!gst_tensor_filter_configure_tensor (self, incaps)) {
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, WRONG_TYPE,
        ("Failed to configure tensor. Please refer to the error log of gst_tensor_filter_configure_tensor ()."));
    return FALSE;
  }

  if (!gst_tensors_config_validate (&priv->in_config)) {
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, WRONG_TYPE,
        ("Failed to validate input tensor configuration. Please refer to the error log of gst_tensors_config_validate(): %s",
            GST_STR_NULL (_nnstreamer_error ())));
    return FALSE;
  }

  if (!gst_tensors_config_validate (&priv->out_config)) {
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, WRONG_TYPE,
        ("Failed to validate output tensor configuration. Please refer to the error log of gst_tensors_config_validate(): %s",
            GST_STR_NULL (_nnstreamer_error ())));
    return FALSE;
  }

  /** compare output tensor */
  structure = gst_caps_get_structure (outcaps, 0);
  gst_tensors_config_from_structure (&config, structure);
  if (gst_tensors_config_is_flexible (&config)) {
    GST_INFO_OBJECT (self, "Output tensor is flexible.");
  } else if (!gst_tensors_config_is_equal (&priv->out_config, &config)) {
    GstTensorFilterProperties *prop = &priv->prop;
    gchar *compare = gst_tensorsinfo_compare_to_string (&priv->out_config.info,
        &config.info);
    GST_ELEMENT_ERROR_BTRACE (self, STREAM, WRONG_TYPE,
        ("Set-caps failed. Invalid output config (padcaps) for tensor-filter (%s:%s): its format is static, but not equal to the internal configuration: %s\nThis might be an internal error. Please report to https://github.com/nnstreamer/nnstreamer/issues .",
            GST_STR_NULL (prop->fwname), TF_MODELNAME (prop), compare));
    g_free (compare);
    return FALSE;
  }

  return TRUE;
}

/**
 * @brief query handling, optional vmethod of GstBaseTransform.
 */
static gboolean
gst_tensor_filter_query (GstBaseTransform * trans,
    GstPadDirection direction, GstQuery * query)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;
  gboolean res = FALSE;

  UNUSED (direction);
  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_LATENCY:
    {
      GstClockTime min, max;
      gboolean live;
      gint estimated;
      gdouble latency;

      GST_OBJECT_LOCK (self);
      estimated = (gint) priv->prop.latency;
      GST_OBJECT_UNLOCK (self);

      if ((priv->latency_reporting) && (estimated > 0)) {
        if ((res = gst_pad_peer_query (GST_BASE_TRANSFORM (self)->sinkpad,
                    query))) {
          gst_query_parse_latency (query, &live, &min, &max);

          GST_DEBUG_OBJECT (self, "Peer latency: min %"
              GST_TIME_FORMAT " max %" GST_TIME_FORMAT,
              GST_TIME_ARGS (min), GST_TIME_ARGS (max));

          latency = (gdouble) estimated *GST_USECOND *
              (1 + LATENCY_REPORT_HEADROOM);
          priv->latency_reported = (gint64) latency;

          min += (gint64) latency;
          if (max != GST_CLOCK_TIME_NONE)
            max += (gint64) latency;

          GST_DEBUG_OBJECT (self, "Calculated total latency : min %"
              GST_TIME_FORMAT " max %" GST_TIME_FORMAT,
              GST_TIME_ARGS (min), GST_TIME_ARGS (max));

          gst_query_set_latency (query, live, min, max);
        }
      }
      if (!res) {
        res =
            GST_BASE_TRANSFORM_CLASS (parent_class)->query (trans, direction,
            query);
      }
      break;
    }
    default:
      res =
          GST_BASE_TRANSFORM_CLASS (parent_class)->query (trans, direction,
          query);
      break;
  }

  return res;
}

/**
 * @brief Tell the framework the required size of buffer based on the info of the other side pad. optional vmethod of BaseTransform
 *
 * We cannot directly get the value from size value, we need to review the pad-caps.
 * This is called when non-ip mode is used.
 */
static gboolean
gst_tensor_filter_transform_size (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, gsize size,
    GstCaps * othercaps, gsize * othersize)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;
  UNUSED (direction);
  UNUSED (caps);
  UNUSED (size);
  UNUSED (othercaps);
  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;
  /** Internal Logic Error. Cannot proceed without configured pipeline */
  g_assert (priv->configured);
  /**
   * Consider multi-tensors.
   * Set each memory block in transform()
   */
  *othersize = 0;
  return TRUE;
}

/**
 * @brief Event handler for sink pad of tensor filter.
 * @param trans "this" pointer
 * @param event a passed event object
 * @return TRUE if there is no error.
 */
static gboolean
gst_tensor_filter_sink_event (GstBaseTransform * trans, GstEvent * event)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;
  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;
  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CUSTOM_DOWNSTREAM:
    {
      const GstStructure *structure = gst_event_get_structure (event);
      int ret = -1;
      if (structure == NULL ||
          !gst_structure_has_name (structure, EVENT_NAME_UPDATE_MODEL))
        break;
      if (priv->is_updatable) {
        const GValue *value =
            gst_structure_get_value (structure, "model_files");
        if (value != NULL) {
          g_object_set (self, "model", value, NULL);
          ret = 0;
        }
      }

      gst_event_unref (event);
      return (ret == 0);
    }
    default:
      break;
  }

  /** other events are handled in the default event handler */
  return GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event (trans, event);
}

/**
 * @brief Event handler for src pad of tensor filter.
 * @param trans "this" pointer
 * @param event a passed event object
 * @return TRUE if there is no error.
 */
static gboolean
gst_tensor_filter_src_event (GstBaseTransform * trans, GstEvent * event)
{
  GstTensorFilter *self = GST_TENSOR_FILTER_CAST (trans);
  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_QOS:
    {
      GstQOSType type;
      GstClockTimeDiff diff;
      gst_event_parse_qos (event, &type, NULL, &diff, NULL);
      if (type == GST_QOS_TYPE_THROTTLE && diff > 0) {
        GST_OBJECT_LOCK (trans);
        if (self->throttling_delay != 0)
          /* set to more tight framerate */
          self->throttling_delay = MIN (self->throttling_delay, diff);
        else
          self->throttling_delay = diff;
        GST_OBJECT_UNLOCK (trans);
        gst_event_unref (event);
        /* enable the average latency profiling */
        g_object_set (self, "latency", 1, NULL);
        return TRUE;
      }
    }
      /* fall-through */
    default:
      break;
  }

  /** other events are handled in the default event handler */
  return GST_BASE_TRANSFORM_CLASS (parent_class)->src_event (trans, event);
}

/**
 * @brief Called when the element starts processing. optional vmethod of BaseTransform
 * @param trans "this" pointer
 * @return TRUE if there is no error.
 */
static gboolean
gst_tensor_filter_start (GstBaseTransform * trans)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;
  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;
  /* If it is not configured properly, don't allow to start! */
  if (priv->fw == NULL)
    return FALSE;
  gst_tensor_filter_common_open_fw (priv);
  return priv->prop.fw_opened;
}

/**
 * @brief Called when the element stops processing. optional vmethod of BaseTransform
 * @param trans "this" pointer
 * @return TRUE if there is no error.
 */
static gboolean
gst_tensor_filter_stop (GstBaseTransform * trans)
{
  GstTensorFilter *self;
  GstTensorFilterPrivate *priv;
  self = GST_TENSOR_FILTER_CAST (trans);
  priv = &self->priv;
  gst_tensor_filter_common_close_fw (priv);
  return TRUE;
}

/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file	tensor_sparse_enc.c
 * @date	27 Jul 2021
 * @brief	GStreamer element to encode dense tensors into sparse tensors
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug		No known bugs except for NYI items
 */

/**
 * SECTION:element-tensor_sparse_enc
 *
 * tensor_sparse_enc is a GStreamer element to encode incoming tensor into sparse format.
 *
 * The input is always in the format of other/tensors,format=static.
 * The output is always in the format of ohter/tensors,format=sparse.
 *
 * Please see also tensor_sparse_dec.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 ... ! other/tensors,format=static ! \
 *    tensor_sparse_enc ! tensor_sink
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include <nnstreamer_util.h>
#include "tensor_sparse_enc.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
#endif

GST_DEBUG_CATEGORY_STATIC (gst_tensor_sparse_enc_debug);
#define GST_CAT_DEFAULT gst_tensor_sparse_enc_debug

/**
 * @brief tensor_sparse_enc properties
 */
enum
{
  PROP_0,
  PROP_SILENT
};

/**
 * @brief Flag to print minimized log.
 */
#define DEFAULT_SILENT TRUE

/**
 * @brief Template for sink pad.
 */
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_CAP_DEFAULT));

/**
 * @brief Template for src pad.
 */
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_SPARSE_CAP_DEFAULT));

#define gst_tensor_sparse_enc_parent_class parent_class
G_DEFINE_TYPE (GstTensorSparseEnc, gst_tensor_sparse_enc, GST_TYPE_ELEMENT);

static void gst_tensor_sparse_enc_finalize (GObject * object);
static void gst_tensor_sparse_enc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_sparse_enc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static GstFlowReturn
gst_tensor_sparse_enc_chain (GstPad * pad, GstObject * parent, GstBuffer * buf);
static GstCaps *gst_tensor_sparse_enc_query_caps (GstTensorSparseEnc * self,
    GstPad * pad, GstCaps * filter);
static gboolean gst_tensor_sparse_enc_sink_event (GstPad * pad,
    GstObject * parent, GstEvent * event);
static gboolean gst_tensor_sparse_enc_sink_query (GstPad * pad,
    GstObject * parent, GstQuery * query);

/**
 * @brief Initialize the tensor_sparse's class.
 */
static void
gst_tensor_sparse_enc_class_init (GstTensorSparseEncClass * klass)
{
  GObjectClass *object_class;
  GstElementClass *element_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_sparse_enc_debug, "tensor_sparse_enc", 0,
      "Element to encode sparse tensors");

  object_class = (GObjectClass *) klass;
  element_class = (GstElementClass *) klass;

  object_class->set_property = gst_tensor_sparse_enc_set_property;
  object_class->get_property = gst_tensor_sparse_enc_get_property;
  object_class->finalize = gst_tensor_sparse_enc_finalize;

  /**
   * GstTensorSparseEnc::silent:
   *
   * The flag to enable/disable debugging messages.
   */
  g_object_class_install_property (object_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output",
          DEFAULT_SILENT, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&src_template));

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&sink_template));

  gst_element_class_set_static_metadata (element_class,
      "TensorSparseEnc",
      "Filter/Tensor",
      "Element to encode dense tensors into sparse tensors",
      "Samsung Electronics Co., Ltd.");
}

/**
 * @brief Initialize tensor_sparse_enc element.
 */
static void
gst_tensor_sparse_enc_init (GstTensorSparseEnc * self)
{
  /* setup sink pad */
  self->sinkpad = gst_pad_new_from_static_template (&sink_template, "sink");
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad);

  /* setup src pad */
  self->srcpad = gst_pad_new_from_static_template (&src_template, "src");
  gst_element_add_pad (GST_ELEMENT (self), self->srcpad);

  /* setup chain function */
  gst_pad_set_chain_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_sparse_enc_chain));

  /* setup event function */
  gst_pad_set_event_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_sparse_enc_sink_event));

  gst_pad_set_query_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_sparse_enc_sink_query));

  /* init properties */
  self->silent = DEFAULT_SILENT;
  gst_tensors_config_init (&self->in_config);
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_tensor_sparse_enc_finalize (GObject * object)
{
  GstTensorSparseEnc *self;
  self = GST_TENSOR_SPARSE_ENC (object);

  gst_tensors_config_free (&self->in_config);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_sparse_enc properties.
 */
static void
gst_tensor_sparse_enc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorSparseEnc *self;

  self = GST_TENSOR_SPARSE_ENC (object);

  switch (prop_id) {
    case PROP_SILENT:
      self->silent = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Getter for tensor_sparse_enc properties.
 */
static void
gst_tensor_sparse_enc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorSparseEnc *self;

  self = GST_TENSOR_SPARSE_ENC (object);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Parse caps and set tensors config
 */
static gboolean
gst_tensor_sparse_enc_parse_caps (GstTensorSparseEnc * self,
    const GstCaps * caps)
{
  GstStructure *structure;
  GstTensorsConfig config;

  g_return_val_if_fail (caps != NULL, FALSE);
  g_return_val_if_fail (gst_caps_is_fixed (caps), FALSE);

  structure = gst_caps_get_structure (caps, 0);

  if (!gst_tensors_config_from_structure (&config, structure) ||
      !gst_tensors_config_validate (&config)) {
    /** not fully configured */
    GST_ERROR_OBJECT (self, "Failed to configure tensors config.\n");
    return FALSE;
  }

  self->in_config = config;
  return TRUE;
}

/**
 * @brief This function handles sink pad event.
 */
static gboolean
gst_tensor_sparse_enc_sink_event (GstPad * pad, GstObject * parent,
    GstEvent * event)
{
  GstTensorSparseEnc *self;
  self = GST_TENSOR_SPARSE_ENC (parent);

  g_return_val_if_fail (event != NULL, FALSE);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_CAPS:
    {
      GstCaps *caps;
      gboolean ret;

      gst_event_parse_caps (event, &caps);
      silent_debug_caps (self, caps, "caps");

      ret = gst_tensor_sparse_enc_parse_caps (self, caps);
      gst_event_unref (event);
      return ret;
    }
    default:
      break;
  }

  return gst_pad_event_default (pad, parent, event);
}

/**
 * @brief Get pad caps for caps negotiation.
 */
static GstCaps *
gst_tensor_sparse_enc_query_caps (GstTensorSparseEnc * self, GstPad * pad,
    GstCaps * filter)
{
  GstCaps *caps;

  caps = gst_pad_get_current_caps (pad);
  if (!caps) {
    /** pad don't have current caps. use the template caps */
    caps = gst_pad_get_pad_template_caps (pad);
  }

  silent_debug_caps (self, caps, "caps");
  silent_debug_caps (self, filter, "filter");

  if (filter) {
    GstCaps *intersection;
    intersection =
        gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (caps);
    caps = intersection;
  }

  silent_debug_caps (self, caps, "result");
  return caps;
}

/**
 * @brief This function handles sink pad query.
 */
static gboolean
gst_tensor_sparse_enc_sink_query (GstPad * pad, GstObject * parent,
    GstQuery * query)
{
  GstTensorSparseEnc *self;

  self = GST_TENSOR_SPARSE_ENC (parent);

  GST_DEBUG_OBJECT (self, "Received %s query: %" GST_PTR_FORMAT,
      GST_QUERY_TYPE_NAME (query), query);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CAPS:
    {
      GstCaps *caps;
      GstCaps *filter;

      gst_query_parse_caps (query, &filter);
      caps = gst_tensor_sparse_enc_query_caps (self, pad, filter);
      silent_debug_caps (self, filter, "filter");
      silent_debug_caps (self, caps, "caps");
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
      silent_debug_caps (self, caps, "caps");

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
 * @brief Internal function to transform the input buffer.
 */
static GstFlowReturn
gst_tensor_sparse_enc_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  GstFlowReturn ret = GST_FLOW_OK;
  GstTensorSparseEnc *self = GST_TENSOR_SPARSE_ENC (parent);
  GstMemory *mem;
  GstBuffer *outbuf;
  GstTensorsInfo *info;
  guint i;

  UNUSED (pad);

  info = &self->in_config.info;
  buf = gst_tensor_buffer_from_config (buf, &self->in_config);
  outbuf = gst_buffer_new ();

  for (i = 0; i < info->num_tensors; ++i) {
    GstTensorMetaInfo meta;
    gst_tensor_info_convert_to_meta (&info->info[i], &meta);

    meta.format = _NNS_TENSOR_FORMAT_SPARSE;
    meta.media_type = _NNS_TENSOR;

    /* do real encoding here */
    mem = gst_buffer_peek_memory (buf, i);
    mem = gst_tensor_sparse_from_dense (&meta, mem);
    if (!mem) {
      nns_loge ("failed to convert to sparse tensor");
      ret = GST_FLOW_ERROR;
      goto done;
    }

    gst_buffer_append_memory (outbuf, mem);
  }

  ret = gst_pad_push (self->srcpad, outbuf);

done:
  gst_buffer_unref (buf);
  if (ret != GST_FLOW_OK)
    gst_buffer_unref (outbuf);

  return ret;
}

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

/**
 * SECTION:element-tensor_crop
 *
 * tensor_crop is a GStreamer element to crop the regions of incoming tensor.
 * @todo TBU, update element description and example pipeline.
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <string.h>
#include "tensor_crop.h"

GST_DEBUG_CATEGORY_STATIC (gst_tensor_crop_debug);
#define GST_CAT_DEFAULT gst_tensor_crop_debug

/**
 * @brief tensor_crop properties
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
    GST_STATIC_CAPS (GST_TENSORS_FLEX_CAP_DEFAULT));

/**
 * @brief Template for src pad.
 */
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_TENSORS_FLEX_CAP_DEFAULT));

#define gst_tensor_crop_parent_class parent_class
G_DEFINE_TYPE (GstTensorCrop, gst_tensor_crop, GST_TYPE_ELEMENT);

static void gst_tensor_crop_finalize (GObject * object);
static void gst_tensor_crop_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_tensor_crop_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static GstFlowReturn gst_tensor_crop_chain (GstPad * pad, GstObject * parent,
    GstBuffer * buffer);

/**
 * @brief Initialize the tensor_crop's class.
 */
static void
gst_tensor_crop_class_init (GstTensorCropClass * klass)
{
  GObjectClass *object_class;
  GstElementClass *element_class;

  GST_DEBUG_CATEGORY_INIT (gst_tensor_crop_debug, "tensor_crop", 0,
      "Element to crop the regions of incoming tensor");

  object_class = (GObjectClass *) klass;
  element_class = (GstElementClass *) klass;

  object_class->set_property = gst_tensor_crop_set_property;
  object_class->get_property = gst_tensor_crop_get_property;
  object_class->finalize = gst_tensor_crop_finalize;

  /**
   * GstTensorCrop::silent:
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
      "TensorCrop",
      "Filter/Tensor",
      "Element to crop the regions of incoming tensor",
      "Samsung Electronics Co., Ltd.");
}

/**
 * @brief Initialize tensor_crop element.
 */
static void
gst_tensor_crop_init (GstTensorCrop * self)
{
  /* setup sink pad */
  self->sinkpad = gst_pad_new_from_static_template (&sink_template, "sink");
  gst_pad_set_chain_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_tensor_crop_chain));
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad);

  /* setup src pad */
  self->srcpad = gst_pad_new_from_static_template (&src_template, "src");
  gst_element_add_pad (GST_ELEMENT (self), self->srcpad);

  /* init properties */
  self->silent = DEFAULT_SILENT;
  gst_tensors_config_init (&self->in_config);
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_tensor_crop_finalize (GObject * object)
{
  GstTensorCrop *self;

  self = GST_TENSOR_CROP (object);

  gst_tensors_info_free (&self->in_config.info);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_crop properties.
 */
static void
gst_tensor_crop_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstTensorCrop *self;

  self = GST_TENSOR_CROP (object);

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
 * @brief Getter for tensor_crop properties.
 */
static void
gst_tensor_crop_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstTensorCrop *self;

  self = GST_TENSOR_CROP (object);

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
 * @brief Chain function, this function does the actual processing.
 */
static GstFlowReturn
gst_tensor_crop_chain (GstPad * pad, GstObject * parent, GstBuffer * buffer)
{
  GstTensorCrop *self;

  self = GST_TENSOR_CROP (parent);

  /**
   * @todo crop incoming buffer
   * 1. check multi tensor (tensor + crop-info)
   * 2. parse crop-info (defined struct, flexible)
   * 3. crop incoming tensor with crop-info
   * 4. finally push flex tensor
   */
  return gst_pad_push (self->srcpad, buffer);
}

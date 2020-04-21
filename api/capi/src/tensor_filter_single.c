/**
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
 *
 */
/**
 * @file	tensor_filter_single.c
 * @date	28 Aug 2019
 * @brief	Element to use general neural network framework directly without gstreamer pipeline
 * @see	  http://github.com/nnstreamer/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug	  No known bugs except for NYI items
 *
 * This is the main element for per-NN-framework plugins.
 * Specific implementations for each NN framework must be written
 * in each framework specific files; e.g., tensor_filter_tensorflow_lite.c
 *
 */

/**
 * SECTION:element-tensor_filter_single
 *
 * An element that invokes neural network models and their framework or
 * an independent shared object implementing tensor_filter_custom.h.
 * The input and output are always in the format of other/tensor or
 * other/tensors. This element is going to be the basis of single shot api.
 *
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <glib.h>
#include <string.h>

#include "tensor_filter_single.h"

#define g_tensor_filter_single_parent_class parent_class
G_DEFINE_TYPE (GTensorFilterSingle, g_tensor_filter_single, G_TYPE_OBJECT);

/* GObject vmethod implementations */
static void g_tensor_filter_single_finalize (GObject * object);
static void g_tensor_filter_single_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void g_tensor_filter_single_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);

/* GTensorFilterSingle method implementations */
static gboolean g_tensor_filter_single_invoke (GTensorFilterSingle * self,
    const GstTensorMemory * input, GstTensorMemory * output);
static gboolean g_tensor_filter_input_configured (GTensorFilterSingle * self);
static gboolean g_tensor_filter_output_configured (GTensorFilterSingle * self);
static gboolean g_tensor_filter_set_input_info (GTensorFilterSingle * self,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info);

/* Private functions */
static gboolean g_tensor_filter_single_start (GTensorFilterSingle * self);
static gboolean g_tensor_filter_single_stop (GTensorFilterSingle * self);

/**
 * @brief initialize the tensor_filter's class
 */
static void
g_tensor_filter_single_class_init (GTensorFilterSingleClass * klass)
{
  GObjectClass *gobject_class;

  gobject_class = (GObjectClass *) klass;

  gobject_class->set_property = g_tensor_filter_single_set_property;
  gobject_class->get_property = g_tensor_filter_single_get_property;
  gobject_class->finalize = g_tensor_filter_single_finalize;

  gst_tensor_filter_install_properties (gobject_class);

  klass->invoke = g_tensor_filter_single_invoke;
  klass->start = g_tensor_filter_single_start;
  klass->stop = g_tensor_filter_single_stop;
  klass->input_configured = g_tensor_filter_input_configured;
  klass->output_configured = g_tensor_filter_output_configured;
  klass->set_input_info = g_tensor_filter_set_input_info;
}

/**
 * @brief initialize the new element
 */
static void
g_tensor_filter_single_init (GTensorFilterSingle * self)
{
  GstTensorFilterPrivate *priv;

  priv = &self->priv;

  gst_tensor_filter_common_init_property (priv);
}

/**
 * @brief Function to finalize instance.
 */
static void
g_tensor_filter_single_finalize (GObject * object)
{
  GTensorFilterSingle *self;
  GstTensorFilterPrivate *priv;

  self = G_TENSOR_FILTER_SINGLE (object);
  priv = &self->priv;

  /** stop if not already stopped */
  if (priv->configured == TRUE) {
    g_tensor_filter_single_stop (self);
  }

  gst_tensor_filter_common_free_property (priv);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_filter_single properties.
 */
static void
g_tensor_filter_single_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GTensorFilterSingle *self;
  GstTensorFilterPrivate *priv;

  self = G_TENSOR_FILTER_SINGLE (object);
  priv = &self->priv;

  g_debug ("Setting property for prop %d.\n", prop_id);

  if (!gst_tensor_filter_common_set_property (priv, prop_id, value, pspec))
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
}

/**
 * @brief Getter for tensor_filter_single properties.
 */
static void
g_tensor_filter_single_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GTensorFilterSingle *self;
  GstTensorFilterPrivate *priv;

  self = G_TENSOR_FILTER_SINGLE (object);
  priv = &self->priv;

  g_debug ("Getting property for prop %d.\n", prop_id);

  if (!gst_tensor_filter_common_get_property (priv, prop_id, value, pspec))
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
}

/**
 * @brief Determine if input is configured
 * (both input and output tensor)
 */
static gboolean
g_tensor_filter_input_configured (GTensorFilterSingle * self)
{
  GstTensorFilterPrivate *priv;

  priv = &self->priv;

  return priv->prop.input_configured;
}

/**
 * @brief Determine if output is configured
 * (both input and output tensor)
 */
static gboolean
g_tensor_filter_output_configured (GTensorFilterSingle * self)
{
  GstTensorFilterPrivate *priv;

  priv = &self->priv;

  return priv->prop.output_configured;
}

/**
 * @brief Called when the element starts processing, if fw not laoded
 * @param self "this" pointer
 * @return TRUE if there is no error.
 */
static gboolean
g_tensor_filter_single_start (GTensorFilterSingle * self)
{
  GstTensorFilterPrivate *priv;

  priv = &self->priv;

  /** open framework, load model */
  if (priv->fw == NULL)
    return FALSE;

  gst_tensor_filter_common_open_fw (priv);

  if (priv->prop.fw_opened == FALSE)
    return FALSE;

  gst_tensor_filter_load_tensor_info (&self->priv);

  return TRUE;
}

/**
 * @brief Called when the element stops processing, if fw loaded
 * @param self "this" pointer
 * @return TRUE if there is no error.
 */
static gboolean
g_tensor_filter_single_stop (GTensorFilterSingle * self)
{
  GstTensorFilterPrivate *priv;

  priv = &self->priv;

  /** close framework, unload model */
  gst_tensor_filter_common_close_fw (priv);
  return TRUE;
}

/**
 * @brief Called when an input supposed to be invoked
 * @param self "this" pointer
 * @param input memory containing input data to run processing on
 * @param output memory to put output data into after processing
 * @return TRUE if there is no error.
 */
static gboolean
g_tensor_filter_single_invoke (GTensorFilterSingle * self,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  GstTensorFilterPrivate *priv;
  guint i;
  gint status;
  gboolean allocate_in_invoke;
  gboolean run_without_model;

  priv = &self->priv;

  if (G_UNLIKELY (!priv->fw))
    return FALSE;

  if (GST_TF_FW_V0 (priv->fw))
    run_without_model = priv->fw->run_without_model;
  else
    run_without_model = priv->info.run_without_model;

  if (G_UNLIKELY (!run_without_model) &&
      G_UNLIKELY (!(priv->prop.model_files &&
              priv->prop.num_models > 0 && priv->prop.model_files[0])))
    return FALSE;

  /** start if not already started */
  if (priv->configured == FALSE) {
    if (!g_tensor_filter_single_start (self)) {
      return FALSE;
    }
    priv->configured = TRUE;
  }

  /** Setup output buffer */
  allocate_in_invoke = gst_tensor_filter_allocate_in_invoke (priv);
  if (allocate_in_invoke == FALSE) {
    /* allocate memory if allocate_in_invoke is FALSE */
    for (i = 0; i < priv->prop.output_meta.num_tensors; i++) {
      output[i].data = g_malloc (output[i].size);
      if (!output[i].data) {
        g_critical ("Failed to allocate the output tensor.");
        goto error;
      }
    }
  }

  GST_TF_FW_INVOKE_COMPAT (priv, status, input, output);

  if (status == 0)
    return TRUE;

error:
  /* if failed to invoke the model, release allocated memory. */
  if (allocate_in_invoke == FALSE) {
    for (i = 0; i < priv->prop.output_meta.num_tensors; i++) {
      g_free (output[i].data);
      output[i].data = NULL;
    }
  }
  return FALSE;
}

/**
 * @brief Set input tensor information in the framework
 * @param self "this" pointer
 * @param in_info information on the input tensor
 * @return TRUE if there is no error.
 */
static gboolean
g_tensor_filter_set_input_info (GTensorFilterSingle * self,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  GstTensorFilterPrivate *priv;
  gint status = -EINVAL;
  gboolean ret = FALSE;

  priv = &self->priv;
  if (G_UNLIKELY (!priv->fw) || G_UNLIKELY (!priv->prop.fw_opened))
    return FALSE;

  gst_tensors_info_init (out_info);
  if (GST_TF_FW_V0 (priv->fw)) {
    if (G_LIKELY (priv->fw->setInputDimension))
      status = priv->fw->setInputDimension (&priv->prop, &priv->privateData,
          in_info, out_info);
  } else {
    status = priv->fw->getModelInfo (priv->fw, &priv->prop, &priv->privateData,
        SET_INPUT_INFO, (GstTensorsInfo *) in_info, out_info);
  }

  if (status == 0) {
    gst_tensors_info_copy (&priv->prop.input_meta, in_info);
    gst_tensors_info_copy (&priv->prop.output_meta, out_info);
    ret = TRUE;
  }

  return ret;
}

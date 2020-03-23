/**
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
 * @file	tensor_filter_single.c
 * @date	28 Aug 2019
 * @brief	Element to use general neural network framework directly without gstreamer pipeline
 * @see	  http://github.com/nnsuite/nnstreamer
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

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!priv->silent)
#endif

#define silent_debug_info(i,msg) do { \
  if (DBG) { \
    guint info_idx; \
    gchar *dim_str; \
    g_debug (msg " total %d", (i)->num_tensors); \
    for (info_idx = 0; info_idx < (i)->num_tensors; info_idx++) { \
      dim_str = gst_tensor_get_dimension_string ((i)->info[info_idx].dimension); \
      g_debug ("[%d] type=%d dim=%s", info_idx, (i)->info[info_idx].type, dim_str); \
      g_free (dim_str); \
    } \
  } \
} while (0)

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
 * @brief Load tensor info from NN model.
 * (both input and output tensor)
 */
static void
g_tensor_filter_load_tensor_info (GTensorFilterSingle * self)
{
  GstTensorFilterPrivate *priv;
  GstTensorFilterProperties *prop;
  GstTensorsInfo in_info, out_info;
  int res = -1;

  priv = &self->priv;
  prop = &priv->prop;

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  /* supposed fixed in-tensor info if getInputDimension is defined. */
  if (!prop->input_configured) {
    res = -1;
    if (priv->prop.fw_opened && priv->fw && priv->fw->getInputDimension) {
      res = priv->fw->getInputDimension
          (&priv->prop, &priv->privateData, &in_info);
    }

    if (res == 0) {
      g_assert (in_info.num_tensors > 0);

      /** if set-property called and already has info, verify it! */
      if (prop->input_meta.num_tensors > 0) {
        if (!gst_tensors_info_is_equal (&in_info, &prop->input_meta)) {
          g_critical ("The input tensor is not compatible.");
          gst_tensor_filter_compare_tensors (&in_info, &prop->input_meta);
          goto done;
        }
      } else {
        gst_tensors_info_copy (&prop->input_meta, &in_info);
      }

      prop->input_configured = TRUE;
      silent_debug_info (&in_info, "input tensor");
    }
  }

  /* supposed fixed out-tensor info if getOutputDimension is defined. */
  if (!prop->output_configured) {
    res = -1;
    if (priv->prop.fw_opened && priv->fw && priv->fw->getOutputDimension) {
      res = priv->fw->getOutputDimension
          (&priv->prop, &priv->privateData, &out_info);
    }

    if (res == 0) {
      g_assert (out_info.num_tensors > 0);

      /** if set-property called and already has info, verify it! */
      if (prop->output_meta.num_tensors > 0) {
        if (!gst_tensors_info_is_equal (&out_info, &prop->output_meta)) {
          g_critical ("The output tensor is not compatible.");
          gst_tensor_filter_compare_tensors (&out_info, &prop->output_meta);
          goto done;
        }
      } else {
        gst_tensors_info_copy (&prop->output_meta, &out_info);
      }

      prop->output_configured = TRUE;
      silent_debug_info (&out_info, "output tensor");
    }
  }

done:
  gst_tensors_info_free (&in_info);
  gst_tensors_info_free (&out_info);
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

  g_tensor_filter_load_tensor_info (self);

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
  gboolean allocate_in_invoke;

  priv = &self->priv;

  if (G_UNLIKELY (!priv->fw) || G_UNLIKELY (!priv->fw->invoke_NN))
    return FALSE;
  if (G_UNLIKELY (!priv->fw->run_without_model) &&
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
  for (i = 0; i < priv->prop.output_meta.num_tensors; i++) {
    /* allocate memory if allocate_in_invoke is FALSE */
    if (allocate_in_invoke == FALSE) {
      output[i].data = g_malloc (output[i].size);
      if (!output[i].data) {
        g_critical ("Failed to allocate the output tensor.");
        goto error;
      }
    }
  }

  if (priv->fw->invoke_NN (&priv->prop, &priv->privateData, input, output) == 0)
    return TRUE;

error:
  if (allocate_in_invoke == FALSE)
    for (i = 0; i < priv->prop.output_meta.num_tensors; i++)
      g_free (output[i].data);
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
  int status;
  gboolean ret = FALSE;

  priv = &self->priv;
  if (G_UNLIKELY (!priv->fw) || G_UNLIKELY (!priv->fw->setInputDimension))
    return FALSE;

  gst_tensors_info_init (out_info);
  status = priv->fw->setInputDimension (&priv->prop, &priv->privateData,
      in_info, out_info);
  if (status == 0) {
    gst_tensors_info_copy (&priv->prop.input_meta, in_info);
    gst_tensors_info_copy (&priv->prop.output_meta, out_info);
    ret = TRUE;
  }

  return ret;
}

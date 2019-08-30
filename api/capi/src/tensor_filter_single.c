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

#include <nnstreamer/nnstreamer_plugin_api.h>
#include <nnstreamer/tensor_typedef.h>
#include <nnstreamer/tensor_filter/tensor_filter_common.h>

#include "tensor_filter_single.h"

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG (!self->silent)
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
    GstTensorMemory * input, GstTensorMemory * output);
static gboolean g_tensor_filter_input_configured (GTensorFilterSingle * self);
static gboolean g_tensor_filter_output_configured (GTensorFilterSingle * self);

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
}

/**
 * @brief initialize the new element
 */
static void
g_tensor_filter_single_init (GTensorFilterSingle * self)
{
  GstTensorFilterProperties *prop;

  prop = &self->prop;

  /* init NNFW properties */
  prop->fwname = NULL;
  prop->fw_opened = FALSE;
  prop->input_configured = FALSE;
  prop->output_configured = FALSE;
  prop->model_file = NULL;
  prop->custom_properties = NULL;
  gst_tensors_info_init (&prop->input_meta);
  gst_tensors_info_init (&prop->output_meta);

  /* init internal properties */
  self->fw = NULL;
  self->privateData = NULL;
  self->silent = TRUE;
  self->started = FALSE;
  gst_tensors_config_init (&self->in_config);
  gst_tensors_config_init (&self->out_config);
}

/**
 * @brief Function to finalize instance.
 */
static void
g_tensor_filter_single_finalize (GObject * object)
{
  gboolean status;
  GTensorFilterSingle *self;
  GstTensorFilterProperties *prop;

  self = G_TENSOR_FILTER_SINGLE (object);

  /** stop if not already stopped */
  if (self->started == TRUE) {
    status = g_tensor_filter_single_stop (self);
    g_debug ("Tensor filter single stop status: %d", status);
  }

  prop = &self->prop;

  g_free_const (prop->fwname);
  g_free_const (prop->model_file);
  g_free_const (prop->custom_properties);

  gst_tensors_info_free (&prop->input_meta);
  gst_tensors_info_free (&prop->output_meta);

  gst_tensors_info_free (&self->in_config.info);
  gst_tensors_info_free (&self->out_config.info);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/**
 * @brief Setter for tensor_filter_single properties.
 */
static void
g_tensor_filter_single_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  /** TODO: share this with tensor_filter*/
}

/**
 * @brief Getter for tensor_filter_single properties.
 */
static void
g_tensor_filter_single_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GTensorFilterSingle *self;
  GstTensorFilterProperties *prop;

  self = G_TENSOR_FILTER_SINGLE (object);
  prop = &self->prop;

  g_debug ("Getting property for prop %d.\n", prop_id);

  switch (prop_id) {
    case PROP_SILENT:
      g_value_set_boolean (value, self->silent);
      break;
    default:
      if (!gst_tensor_filter_common_get_property (prop, prop_id, value, pspec))
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief Determine if input is configured
 * (both input and output tensor)
 */
static gboolean
g_tensor_filter_input_configured (GTensorFilterSingle * self)
{
  if (self->prop.input_configured)
    return TRUE;
  else
    return FALSE;
}

/**
 * @brief Determine if output is configured
 * (both input and output tensor)
 */
static gboolean
g_tensor_filter_output_configured (GTensorFilterSingle * self)
{
  if (self->prop.output_configured)
    return TRUE;
  else
    return FALSE;
}

/**
 * @brief Load tensor info from NN model.
 * (both input and output tensor)
 */
static void
g_tensor_filter_load_tensor_info (GTensorFilterSingle * self)
{
  GstTensorFilterProperties *prop;
  GstTensorsInfo in_info, out_info;
  int res = -1;

  prop = &self->prop;

  gst_tensors_info_init (&in_info);
  gst_tensors_info_init (&out_info);

  /* supposed fixed in-tensor info if getInputDimension is defined. */
  if (!prop->input_configured) {
    res = -1;
    if (self->prop.fw_opened && self->fw && self->fw->getInputDimension) {
      res = self->fw->getInputDimension
        (&self->prop, &self->privateData, &in_info);
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
    if (self->prop.fw_opened && self->fw && self->fw->getOutputDimension) {
      res = self->fw->getOutputDimension
        (&self->prop, &self->privateData, &out_info);
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
  /** open framework, load model */
  if (self->fw == NULL)
    return FALSE;

  if (self->prop.fw_opened == FALSE && self->fw) {
    if (self->fw->open != NULL) {
      if (self->fw->open (&self->prop, &self->privateData) == 0)
        self->prop.fw_opened = TRUE;
    } else {
      self->prop.fw_opened = TRUE;
    }
  }

  if (self->prop.fw_opened == FALSE)
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
  /** close framework, unload model */
  if (self->prop.fw_opened) {
    if (self->fw && self->fw->close) {
      self->fw->close (&self->prop, &self->privateData);
    }
    self->prop.fw_opened = FALSE;
    g_free_const (self->prop.fwname);
    self->prop.fwname = NULL;
    self->fw = NULL;
    self->privateData = NULL;
  }
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
    GstTensorMemory * input, GstTensorMemory * output)
{
  gboolean status = TRUE;
  /** TODO: fill this */

  /** start if not already started */
  if (self->started == FALSE)
    status = g_tensor_filter_single_start (self);

  return status;
}

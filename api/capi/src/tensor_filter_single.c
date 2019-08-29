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
#define DBG (!self->silent)
#endif

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

  klass->invoke = g_tensor_filter_single_invoke;

  /** TODO: share same setup properties as tensor_filter */
}

/**
 * @brief initialize the new element
 */
static void
g_tensor_filter_single_init (GTensorFilterSingle * self)
{
  /** TODO: fill this */
}

/**
 * @brief Function to finalize instance.
 */
static void
g_tensor_filter_single_finalize (GObject * object)
{
  gboolean status;
  GTensorFilterSingle *self;
  /** TODO: fill this */

  self = G_TENSOR_FILTER_SINGLE (object);

  /** stop if not already stopped */
  if (self->started == TRUE) {
    status = g_tensor_filter_single_stop (self);
    g_debug ("Tensor filter single stop status: %d", status);
  }
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
  /** TODO: share this with tensor_filter*/
}

/**
 * @brief Called when the element starts processing, if fw not laoded
 * @param self "this" pointer
 * @return TRUE if there is no error.
 */
static gboolean
g_tensor_filter_single_start (GTensorFilterSingle * self)
{
  /** TODO: fill this */
  /** open framework, load model */
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
  /** TODO: fill this */
  /** close framework, unload model */
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

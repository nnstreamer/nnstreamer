/**
 * Copyright (C) 2019 Parichay kapoor <pk.kapoor@samsung.com>
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
 * @file	tensor_filter_single.h
 * @date	28 Aug 2019
 * @brief	Element to use general neural network framework individually without gstreamer pipeline
 * @see	  http://github.com/nnsuite/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug	  No known bugs except for NYI items
 */

#ifndef __G_TENSOR_FILTER_SINGLE_H__
#define __G_TENSOR_FILTER_SINGLE_H__

#include <stdint.h>
#include <glib-object.h>

#include <nnstreamer/nnstreamer_subplugin.h>
#include <nnstreamer/nnstreamer_plugin_api_filter.h>

G_BEGIN_DECLS
#define G_TYPE_TENSOR_FILTER_SINGLE \
  (g_tensor_filter_single_get_type())
#define G_TENSOR_FILTER_SINGLE(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),G_TYPE_TENSOR_FILTER_SINGLE,GTensorFilterSingle))
#define G_TENSOR_FILTER_SINGLE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),G_TYPE_TENSOR_FILTER_SINGLE,GTensorFilterSingleClass))
#define G_IS_TENSOR_FILTER_SINGLE(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),G_TYPE_TENSOR_FILTER_SINGLE))
#define G_IS_TENSOR_FILTER_SINGLE_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),G_TYPE_TENSOR_FILTER_SINGLE))
#define G_TENSOR_FILTER_SINGLE_CAST(obj)  ((GTensorFilterSingle *)(obj))

typedef struct _GTensorFilterSingle GTensorFilterSingle;
typedef struct _GTensorFilterSingleClass GTensorFilterSingleClass;

/**
 * @brief Internal data structure for tensor_filter_single instances.
 */
struct _GTensorFilterSingle
{
  GObject element;     /**< This is the parent object */

  void *privateData; /**< NNFW plugin's private data is stored here */
  GstTensorFilterProperties prop; /**< NNFW plugin's properties */
  const GstTensorFilterFramework *fw; /**< The implementation core of the NNFW. NULL if not configured */

  /* internal properties for tensor_filter_single */
  gboolean silent; /**< Verbose mode if FALSE. int instead of gboolean for non-glib custom plugins */
  gboolean started; /**< filter has been started */
  GstTensorsConfig in_config; /**< input tensor info */
  GstTensorsConfig out_config; /**< output tensor info */
};

/**
 * @brief GTensorFilterSingleClass inherits GObjectClass.
 */
struct _GTensorFilterSingleClass
{
  GObjectClass parent; /**< inherits GObjectClass */


  /** Invoke the filter for execution. */
  gboolean (*invoke) (GTensorFilterSingle * self, GstTensorMemory * input, GstTensorMemory * output);
  /** Start the filter, must be called before invoke. */
  gboolean (*start) (GTensorFilterSingle * self);
  /** Stop the filter.*/
  gboolean (*stop) (GTensorFilterSingle * self);
  /** Check if the input is already configured */
  gboolean (*input_configured) (GTensorFilterSingle * self);
  /** Check if the output is already configured */
  gboolean (*output_configured) (GTensorFilterSingle * self);
};

/**
 * @brief Get Type function required for gst elements
 */
GType g_tensor_filter_single_get_type (void);

G_END_DECLS
#endif /* __G_TENSOR_FILTER_SINGLE_H__ */

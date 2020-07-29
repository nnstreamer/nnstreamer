/**
 * Copyright (C) 2019 Parichay kapoor <pk.kapoor@samsung.com>
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
 * @file	tensor_filter_single.h
 * @date	28 Aug 2019
 * @brief	Element to use general neural network framework individually without gstreamer pipeline
 * @see	  http://github.com/nnstreamer/nnstreamer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug	  No known bugs except for NYI items
 */

#ifndef __G_TENSOR_FILTER_SINGLE_H__
#define __G_TENSOR_FILTER_SINGLE_H__

#include <stdint.h>
#include <glib-object.h>

#include <nnstreamer/tensor_filter/tensor_filter_common.h>

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
  size_t total_output_size;     /**< Total size of the output tensor data */
  size_t output_offset[NNS_TENSOR_SIZE_LIMIT];  /**< Offset of each output from base memory */

  GstTensorFilterPrivate priv; /**< Internal properties for tensor-filter */
};

/**
 * @brief GTensorFilterSingleClass inherits GObjectClass.
 */
struct _GTensorFilterSingleClass
{
  GObjectClass parent; /**< inherits GObjectClass */

  /** Invoke the filter for execution. */
  gboolean (*invoke) (GTensorFilterSingle * self, const GstTensorMemory * input,
      GstTensorMemory * output);
  /** Start the filter, must be called before invoke. */
  gboolean (*start) (GTensorFilterSingle * self);
  /** Stop the filter.*/
  gboolean (*stop) (GTensorFilterSingle * self);
  /** Check if the input is already configured */
  gboolean (*input_configured) (GTensorFilterSingle * self);
  /** Check if the output is already configured */
  gboolean (*output_configured) (GTensorFilterSingle * self);
  /** Set the info about the input tensor */
  gint (*set_input_info) (GTensorFilterSingle * self,
      const GstTensorsInfo * in_info, GstTensorsInfo * out_info);
};

/**
 * @brief Get Type function required for gst elements
 */
GType g_tensor_filter_single_get_type (void);

G_END_DECLS
#endif /* __G_TENSOR_FILTER_SINGLE_H__ */

/**
 * GStreamer Tensor_Filter
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	tensor_filter.h
 * @date	24 May 2018
 * @brief	GStreamer plugin to use general neural network frameworks as filters
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * @todo TBD: Should we disable "in-place" mode? (what if output size > input size?)
 */

#ifndef __GST_TENSOR_FILTER_H__
#define __GST_TENSOR_FILTER_H__

#include <stdint.h>
#include <gst/gst.h>
#include <gst/gstinfo.h>
#include <gst/base/gstbasetransform.h>
#include <tensor_common.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_FILTER \
  (gst_tensor_filter_get_type())
#define GST_TENSOR_FILTER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_FILTER,GstTensorFilter))
#define GST_TENSOR_FILTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_FILTER,GstTensorFilterClass))
#define GST_IS_TENSOR_FILTER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_FILTER))
#define GST_IS_TENSOR_FILTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_FILTER))
#define GST_TENSOR_FILTER_CAST(obj)  ((GstTensorFilter *)(obj))

typedef struct _GstTensorFilter GstTensorFilter;
typedef struct _GstTensorFilterClass GstTensorFilterClass;
typedef struct _GstTensorFilterFramework GstTensorFilterFramework;

extern const char *nnfw_names[];

/**
 * @brief Internal data structure for tensor_filter instances.
 */
struct _GstTensorFilter
{
  GstBaseTransform element;     /**< This is the parent object */

  void *privateData; /**< NNFW plugin's private data is stored here */
  GstTensorFilterProperties prop; /**< NNFW plugin's properties */
  GstTensorFilterFramework *fw; /**< The implementation core of the NNFW. NULL if not configured */

  /** internal properties for tensor-filter */
  int silent; /**< Verbose mode if FALSE. int instead of gboolean for non-glib custom plugins */
  gboolean configured; /**< True if already successfully configured tensor metadata */
  GstTensorsConfig in_config; /**< input tensor info */
  GstTensorsConfig out_config; /**< output tensor info */
};

/**
 * @brief Location of GstTensorFilter from privateData
 * @param p the "privateData" pointer of GstTensorFilter
 * @return the pointer to GstTensorFilter containing p as privateData
 */
#define GstTensorFilter_of_privateData(p) ({ \
    const void **__mptr = (const void **)(p); \
    (GstTensorFilter *)( (char *)__mptr - offsetof(GstTensorFilter, privateData) );})

/**
 * @brief GstTensorFilterClass inherits GstBaseTransformClass.
 *
 * Referring another child (sibiling), GstVideoFilter (abstract class) and
 * its child (concrete class) GstVideoConverter.
 * Note that GstTensorFilterClass is a concrete class; thus we need to look at both.
 */
struct _GstTensorFilterClass
{
  GstBaseTransformClass parent_class;   /**< Inherits GstBaseTransformClass */
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_filter_get_type (void);

/**
 * @brief Subplugin definition
 *
 * Common callback parameters:
 * filter Filter properties. Read Only
 * private_data Subplugin's private data. Set this (*private_data = XXX) if you want to change filter->private_data
 */
struct _GstTensorFilterFramework
{
  gchar *name; /**< Name of the neural network framework, searchable by FRAMEWORK property */
  gboolean allow_in_place; /**< TRUE if InPlace transfer of input-to-output is allowed. Not supported in main, yet */
  gboolean allocate_in_invoke; /**< TRUE if invoke_NN is going to allocate outputptr by itself and return the address via outputptr. Do not change this value after cap negotiation is complete (or the stream has been started). */

  int (*invoke_NN) (const GstTensorFilter * filter, void **private_data,
      const GstTensorMemory * input, GstTensorMemory * output);
      /**< Mandatory callback. Invoke the given network model.
       *
       * @param[in] filter "this" pointer. Use this to read property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[in] input The array of input tensors. Allocated and filled by tensor_filter/main
       * @param[out] output The array of output tensors. Allocated by tensor_filter/main and to be filled by invoke_NN. If allocate_in_invoke is TRUE, sub-plugin should allocate the memory block for output tensor. (data in GstTensorMemory)
       * @return 0 if OK. non-zero if error.
       */

  int (*getInputDimension) (const GstTensorFilter * filter,
      void **private_data, GstTensorsInfo * info);
      /**< Optional. Set NULL if not supported. Get dimension of input tensor
       * If getInputDimension is NULL, setInputDimension must be defined.
       * If getInputDimension is defined, it is recommended to define getOutputDimension
       *
       * @param[in] filter "this" pointer. Use this to read property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[out] info structure of tensor info (return value)
       * @return the size of input tensors
       */

  int (*getOutputDimension) (const GstTensorFilter * filter,
      void **private_data, GstTensorsInfo * info);
      /**< Optional. Set NULL if not supported. Get dimension of output tensor
       * If getInputDimension is NULL, setInputDimension must be defined.
       * If getInputDimension is defined, it is recommended to define getOutputDimension
       *
       * @param[in] filter "this" pointer. Use this to read property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[out] info structure of tensor info (return value)
       * @return the size of output tensors
       */

  int (*setInputDimension) (const GstTensorFilter * filter,
      void **private_data, const GstTensorsInfo * in_info,
      GstTensorsInfo * out_info);
      /**< Optional. Set Null if not supported. Tensor_filter::main will
       * configure input dimension from pad-cap in run-time for the sub-plugin.
       * Then, the sub-plugin is required to return corresponding output dimension
       * If this is NULL, both getInput/OutputDimension must be non-NULL.
       *
       * When you use this, do NOT allocate or fix internal data structure based on it
       * until invoke is called. Gstreamer may try different dimensions before
       * settling down.
       *
       * @param[in] filter "this" pointer. Use this to read property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[in] in_info structure of input tensor info
       * @param[out] out_info structure of output tensor info (return value)
       * @return 0 if OK. non-zero if error.
       */

  int (*open) (const GstTensorFilter * filter, void **private_data);
      /**< Optional. tensor_filter.c will call this before any of other callbacks and will call once before calling close
       *
       * @param[in] filter "this" pointer. Use this to read property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer. Normally, open() allocates memory for private_data.
       * @return 0 if ok. < 0 if error.
       */

  void (*close) (const GstTensorFilter * filter, void **private_data);
      /**< Optional. tensor_filter.c will not call other callbacks after calling close. Free-ing private_data is this function's responsibility. Set NULL after that.
       *
       * @param[in] filter "this" pointer. Use this to read property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer. Normally, close() frees private_data and set NULL.
       */
};

extern GstTensorFilterFramework NNS_support_tensorflow_lite;
extern GstTensorFilterFramework NNS_support_tensorflow;
extern GstTensorFilterFramework NNS_support_custom;

extern GstTensorFilterFramework *tensor_filter_supported[];

G_END_DECLS

#endif /* __GST_TENSOR_FILTER_H__ */

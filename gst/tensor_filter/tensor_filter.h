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
 * @see		http://github.com/TO-BE-DETERMINED-SOON
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * @todo TBD: Should we disable "in-place" mode? (what if output size > input size?)
 */

#ifndef __GST_TENSOR_FILTER_H__
#define __GST_TENSOR_FILTER_H__

#include <stdint.h>
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <tensor_common.h>

G_BEGIN_DECLS
/* #defines don't like whitespacey bits */
#define GST_TYPE_TENSOR_FILTER \
  (gst_tensor_filter_get_type())
#define GST_TENSOR_FILTER(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_FILTER,GstTensor_Filter))
#define GST_TENSOR_FILTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_FILTER,GstTensor_FilterClass))
#define GST_IS_TENSOR_FILTER(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_FILTER))
#define GST_IS_TENSOR_FILTER_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_FILTER))
#define GST_TENSOR_FILTER_CAST(obj)  ((GstTensor_Filter *)(obj))
typedef struct _GstTensor_Filter GstTensor_Filter;

typedef struct _GstTensor_FilterClass GstTensor_FilterClass;

extern const char *nnfw_names[];

/**
 * @brief Internal data structure for tensor_filter instances.
 */
struct _GstTensor_Filter
{
  GstBaseTransform element;     /**< This is the parent object */

  void *privateData; /**< NNFW plugin's private data is stored here */

  GstTensor_Filter_Properties prop;
};

/** @brief Location of GstTensor_Filter from privateData
 *  @param p the "privateData" pointer of GstTensor_Filter
 *  @return the pointer to GstTensor_Filter containing p as privateData
 */
#define GstTensor_Filter_of_privateData(p) ({ \
    const void **__mptr = (const void **)(p); \
    (GstTensor_Filter *)( (char *)__mptr - offsetof(GstTensor_Filter, privateData) );})

/**
 * @brief GstTensor_FilterClass inherits GstBaseTransformClass.
 *
 * Referring another child (sibiling), GstVideoFilter (abstract class) and
 * its child (concrete class) GstVideoConverter.
 * Note that GstTensor_FilterClass is a concrete class; thus we need to look at both.
 */
struct _GstTensor_FilterClass
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
struct _GstTensor_Filter_Framework
{
  gchar *name; /**< Name of the neural network framework, searchable by FRAMEWORK property */
  gboolean allow_in_place; /**< TRUE if InPlace transfer of input-to-output is allowed. Not supported in main, yet */
  gboolean allocate_in_invoke; /**< TRUE if invoke_NN is going to allocate outputptr by itself and return the address via outputptr. Do not change this value after cap negotiation is complete (or the stream has been started). */

  uint8_t *(*invoke_NN) (const GstTensor_Filter * filter, void **private_data,
      const uint8_t * inputptr, uint8_t * outputptr);
      /**< Mandatory callback. Invoke the given network model.
       *
       * @param[in] filter "this" pointer. Use this to read property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[in] inputptr Input tensor. Allocated and filled by tensor_filter/main
       * @param[out] outputptr Output tensor. Allocated by tensor_filter/main and to be filled by invoke_NN. N/C if allocate_in_invoke is TRUE.
       * @return outputptr if allocate_in_invoke = 00 if OK. non-zero if error.
       */

  int (*getInputDimension) (const GstTensor_Filter * filter,
      void **private_data, GstTensor_TensorsMeta * meta);
      /**< Optional. Set NULL if not supported. Get dimension of input tensor
       * If getInputDimension is NULL, setInputDimension must be defined.
       * If getInputDimension is defined, it is recommended to define getOutputDimension
       *
       * @param[in] filter "this" pointer. Use this to read property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[out] inputDimension dimension of input tensor (return value)
       * @param[out] type type of input tensor element (return value)
       * @return the size of input tensors
       */
  int (*getOutputDimension) (const GstTensor_Filter * filter,
      void **private_data, GstTensor_TensorsMeta * meta);
      /**< Optional. Set NULL if not supported. Get dimension of output tensor
       * If getInputDimension is NULL, setInputDimension must be defined.
       * If getInputDimension is defined, it is recommended to define getOutputDimension
       *
       * @param[in] filter "this" pointer. Use this to read property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer.
       * @param[out] outputDimension dimension of output tensor (return value)
       * @param[out] type type of output tensor element (return value)
       * @return the size of output tensors
       */
  int (*setInputDimension) (const GstTensor_Filter * filter,
      void **private_data, const GstTensor_TensorsMeta * inputMeta, GstTensor_TensorsMeta * outputMeta);
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
       * @param[in] inputDimension dimension of input tensor
       * @param[in] inputType type of input tensor element
       * @param[out] outputDimension dimension of output tensor (return value)
       * @param[out] outputType type of output tensor element (return value)
       * @return 0 if OK. non-zero if error.
       */

  void (*open) (const GstTensor_Filter * filter, void **private_data);
      /**< Optional. tensor_filter.c will call this before any of other callbacks and will call once before calling close
       *
       * @param[in] filter "this" pointer. Use this to read property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer. Normally, open() allocates memory for private_data.
       */
  void (*close) (const GstTensor_Filter * filter, void **private_data);
      /**< Optional. tensor_filter.c will not call other callbacks after calling close. Free-ing private_data is this function's responsibility. Set NULL after that.
       *
       * @param[in] filter "this" pointer. Use this to read property values
       * @param[in/out] private_data A subplugin may save its internal private data here. The subplugin is responsible for alloc/free of this pointer. Normally, close() frees private_data and set NULL.
       */
};

extern GstTensor_Filter_Framework NNS_support_tensorflow_lite;
extern GstTensor_Filter_Framework NNS_support_tensorflow;
extern GstTensor_Filter_Framework NNS_support_custom;

extern GstTensor_Filter_Framework *tensor_filter_supported[];

G_END_DECLS
#endif /* __GST_TENSOR_FILTER_H__ */

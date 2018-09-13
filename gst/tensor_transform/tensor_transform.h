/**
 * GStreamer
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
 * @file	tensor_transform.c
 * @date	10 Jul 2018
 * @brief	GStreamer plugin to transform other/tensor dimensions
 *
 * @see		http://github.com/nnsuite/nnstreamer
 * @see		https://github.sec.samsung.net/STAR/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs.
 *
 */
/**
 * SECTION:element-tensor_transform
 *
 * A filter that converts other/tensor formats
 * The input/output is always in the format of other/tensor
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! tensor_transform mode=dimchg option=0:2 ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifndef __GST_TENSOR_TRANSFORM_H__
#define __GST_TENSOR_TRANSFORM_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <tensor_common.h>

G_BEGIN_DECLS

/* #defines don't like whitespacey bits */
#define GST_TYPE_TENSOR_TRANSFORM \
  (gst_tensor_transform_get_type())
#define GST_TENSOR_TRANSFORM(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_TRANSFORM,GstTensor_Transform))
#define GST_TENSOR_TRANSFORM_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_TRANSFORM,GstTensor_TransformClass))
#define GST_IS_TENSOR_TRANSFORM(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_TRANSFORM))
#define GST_IS_TENSOR_TRANSFORM_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_TRANSFORM))
#define GST_TENSOR_TRANSFORM_CAST(obj)  ((GstTensor_Transform *)(obj))

typedef struct _GstTensor_Transform GstTensor_Transform;

typedef struct _GstTensor_TransformClass GstTensor_TransformClass;

typedef enum
{
  GTT_DIMCHG = 0,               /* Dimension Change. "dimchg" */
  GTT_TYPECAST = 1,             /* Type change. "typecast" */
  GTT_ARITHMETIC = 2,           /* Type change. "typecast" */
  GTT_TRANSPOSE = 3,            /* Transpose. "transpose" */

  GTT_END,
} tensor_transform_mode;

typedef enum
{
  ARITH_ADD = 0,
  ARITH_MUL = 1,
  ARITH_END,
} tensor_transform_arith_mode;

/**
 * @brief Internal data structure for dimchg mode.
 */
typedef struct _tensor_transform_dimchg {
  int from;
  int to;
} tensor_transform_dimchg;

/**
 * @brief Internal data structure for typecast mode.
 */
typedef struct _tensor_transform_typecast {
  tensor_type to; /**< tensor_type after cast. _NNS_END if unknown */
} tensor_transform_typecast;

/**
 * @brief Internal data structure for arithmetic mode.
 */
typedef struct _tensor_transform_arithmetic {
  tensor_transform_arith_mode mode;
  /* TODO : Better to use union for various type*/
  int64_t value;
} tensor_transform_arithmetic;

/**
 * @brief Internal data structure for transpose mode.
 */
typedef struct _tensor_transform_transpose {
  uint8_t trans_order[NNS_TENSOR_RANK_LIMIT];
} tensor_transform_transpose;

/**
 * @brief Internal data structure for tensor_transform instances.
 */
struct _GstTensor_Transform
{
  GstBaseTransform element;	/**< This is the parent object */

  gboolean silent;	/**< True if logging is minimized */
  tensor_transform_mode mode; /**< Transform mode. GTT_END if invalid */
  gchar *option; /**< Stored option value */
  union {
    tensor_transform_dimchg data_dimchg; /**< Parsed option value for "dimchg" mode */
    tensor_transform_typecast data_typecast; /**< Parsed option value for "typecast" mode. */
    tensor_transform_arithmetic data_arithmetic; /**< Parsed option value for "arithmetic" mode. */
    tensor_transform_transpose data_transpose; /**< Parsed option value for "transpose" mode. */
  };
  gboolean loaded; /**< TRUE if mode & option are loaded */

  tensor_dim fromDim; /**< Input dimension */
  tensor_dim toDim; /**< Output dimension */
  tensor_type type; /**< tensor_type of input. Most transform share the same type for both input and output. However, this does not hold for typecast. */
};

/**
 * @brief GstTensor_TransformClass inherits GstBaseTransformClass.
 *
 * Referring another child (sibiling), GstVideoFilter (abstract class) and
 * its child (concrete class) GstVideoTransform.
 * Note that GstTensor_TransformClass is a concrete class; thus we need to look at both.
 */
struct _GstTensor_TransformClass
{
  GstBaseTransformClass parent_class;	/**< Inherits GstBaseTransformClass */
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_transform_get_type (void);

G_END_DECLS

#endif /* __GST_TENSOR_TRANSFORM_H__ */

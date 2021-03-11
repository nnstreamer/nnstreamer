/**
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
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
 * @file	tensor_transform.h
 * @date	10 Jul 2018
 * @brief	GStreamer plugin to transform tensor dimension or type
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs.
 *
 */

#ifndef __GST_TENSOR_TRANSFORM_H__
#define __GST_TENSOR_TRANSFORM_H__

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <tensor_common.h>
#include <tensor_data.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_TRANSFORM \
  (gst_tensor_transform_get_type())
#define GST_TENSOR_TRANSFORM(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_TRANSFORM,GstTensorTransform))
#define GST_TENSOR_TRANSFORM_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_TRANSFORM,GstTensorTransformClass))
#define GST_IS_TENSOR_TRANSFORM(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_TRANSFORM))
#define GST_IS_TENSOR_TRANSFORM_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_TRANSFORM))
#define GST_TENSOR_TRANSFORM_CAST(obj)  ((GstTensorTransform *)(obj))

typedef struct _GstTensorTransform GstTensorTransform;
typedef struct _GstTensorTransformClass GstTensorTransformClass;

typedef enum _tensor_transform_mode
{
  GTT_DIMCHG = 0,     /* Dimension Change. "dimchg" */
  GTT_TYPECAST,       /* Type change. "typecast" */
  GTT_ARITHMETIC,     /* Arithmetic. "arithmetic" */
  GTT_TRANSPOSE,      /* Transpose. "transpose" */
  GTT_STAND,          /* Standardization. "stand" */
  GTT_CLAMP,          /* Clamp, "clamp" */

  GTT_UNKNOWN = -1,   /* Unknown/Not-implemented-yet Mode. "unknown" */
} tensor_transform_mode;

typedef enum
{
  GTT_OP_TYPECAST = 0,
  GTT_OP_ADD = 1,
  GTT_OP_MUL = 2,
  GTT_OP_DIV = 3,

  GTT_OP_UNKNOWN
} tensor_transform_operator;

typedef enum
{
  STAND_DEFAULT = 0,
  STAND_END,
} tensor_transform_stand_mode;

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
 * @brief Internal data structure for operator of arithmetic mode.
 */
typedef struct
{
  tensor_transform_operator op;
  tensor_data_s value;
} tensor_transform_operator_s;

/**
 * @brief Internal data structure for arithmetic mode.
 */
typedef struct _tensor_transform_arithmetic {
  tensor_type out_type;
} tensor_transform_arithmetic;

/**
 * @brief Internal data structure for transpose mode.
 */
typedef struct _tensor_transform_transpose {
  uint8_t trans_order[NNS_TENSOR_RANK_LIMIT];
} tensor_transform_transpose;

/**
 * @brief Internal data structure for stand mode.
 */
typedef struct _tensor_transform_stand {
  tensor_transform_stand_mode mode;
  tensor_type out_type;
} tensor_transform_stand;

/**
 * @brief Internal data structure for clamp mode.
 */
typedef struct _tensor_transform_clamp {
  double min, max;
} tensor_transform_clamp;

/**
 * @brief Internal data structure for tensor_transform instances.
 */
struct _GstTensorTransform
{
  GstBaseTransform element;	/**< This is the parent object */

  gboolean silent;	/**< True if logging is minimized */
  tensor_transform_mode mode; /**< Transform mode. GTT_UNKNOWN if invalid. */
  gchar *option; /**< Stored option value */
  union {
    tensor_transform_dimchg data_dimchg; /**< Parsed option value for "dimchg" mode */
    tensor_transform_typecast data_typecast; /**< Parsed option value for "typecast" mode. */
    tensor_transform_arithmetic data_arithmetic; /**< Parsed option value for "arithmetic" mode. */
    tensor_transform_transpose data_transpose; /**< Parsed option value for "transpose" mode. */
    tensor_transform_stand data_stand; /**< Parsed option value for "stand" mode. */
    tensor_transform_clamp data_clamp; /**< Parsed option value for "clamp" mode. */
  };
  gboolean loaded; /**< TRUE if mode & option are loaded */
  gboolean acceleration; /**< TRUE to set orc acceleration */
#ifdef HAVE_ORC
  gboolean orc_supported; /**< TRUE if orc supported */
#endif
  GSList *operators; /**< operators list */

  GstTensorConfig in_config; /**< input tensor info */
  GstTensorConfig out_config; /**< output tensor info */
};

/**
 * @brief GstTensorTransformClass inherits GstBaseTransformClass.
 *
 * Referring another child (sibiling), GstVideoFilter (abstract class) and
 * its child (concrete class) GstVideoTransform.
 * Note that GstTensorTransformClass is a concrete class; thus we need to look at both.
 */
struct _GstTensorTransformClass
{
  GstBaseTransformClass parent_class;	/**< Inherits GstBaseTransformClass */
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_transform_get_type (void);

G_END_DECLS

#endif /* __GST_TENSOR_TRANSFORM_H__ */

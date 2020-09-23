/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer/NNStreamer Tensor-IF
 * Copyright (C) 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file	gsttensorif.h
 * @date	08 April 2020
 * @brief	GStreamer plugin to control flow based on tensor values
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * @todo Add "event/signal" to reload FILL_WITH_FILE* file??? (TBD)
 *
 * @details
 *	The output dimension is SAME with input dimension.
 */
#ifndef __GST_TENSOR_IF_H__
#define __GST_TENSOR_IF_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <tensor_common.h>

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_IF (gst_tensor_if_get_type ())
#define GST_TENSOR_IF(obj) (G_TYPE_CHECK_INSTANCE_CAST ((obj), GST_TYPE_TENSOR_IF, GstTensorIf))
#define GST_TENSOR_IF_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST ((klass), GST_TYPE_TENSOR_IF, GstTensorIfClass))
#define GST_TENSOR_IF_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS ((obj), GST_TYPE_TENSOR_IF, GstTensorIfClass))
#define GST_IS_TENSOR_IF(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_IF))
#define GST_IS_TENSOR_IF_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_IF))
#define GST_TENSOR_IF_CAST(obj)((GstTensorIf*)(obj))
typedef struct _GstTensorIf GstTensorIf;
typedef struct _GstTensorIfClass GstTensorIfClass;

/**
 * @brief Compared_Value
 */
typedef enum {
  TIFCV_A_VALUE = 0,	/**< Decide based on a single scalar value
			     of tensors */
  TIFCV_TENSOR_TOTAL_VALUE = 1,	/**< Decide based on a total (sum) value of
				     a specific tensor */
  TIFCV_ALL_TENSORS_TOTAL_VALUE = 2,	/**< Decide based on a total (sum) value of
					     of tensors or a specific tensor */
  TIFCV_TENSOR_AVERAGE_VALUE = 3,	/**< Decide based on a average value of
				     a specific tensor */
  TIFCV_ALL_TENSORS_AVERAGE_VALUE = 4,	/**< Decide based on a average value of
					     tensors or a specific tensor */
  TIFCV_END,
} tensor_if_compared_value;

/**
 * @brief OPERAND
 */
typedef enum {
  TIFOP_EQ = 0,	/**< == */
  TIFOP_NE,	/**< != */
  TIFOP_GT,	/**< > */
  TIFOP_GE,	/**< >= */
  TIFOP_LT,	/**< < */
  TIFOP_LE,	/**< <= */
  TIFOP_RANGE_INCLUSIVE,	/**< in [min, max] */
  TIFOP_RANGE_EXCLUSIVE,	/**< in (min, max) */
  TIFOP_NOT_IN_RANGE_INCLUSIVE,	/**< not in [min, max] */
  TIFOP_NOT_IN_RANGE_EXCLUSIVE, /**< not in (min, max) */
  TIFOP_END,
} tensor_if_operator;

/**
 * @brief Behaviors that may fit in THEN and ELSE
 * @details FILL_WITH_FILE, FILL_WITH_FILE_RPT, and REPEAT_PREVIOUS_FRAME caches an output frame
 *          and thus, may consume additional memory and incur an additional memcpy.
 */
typedef enum {
  TIFB_PASSTHROUGH = 0,	/**< The input frame becomes the output frame */
  TIFB_SKIP,	/**< Do not generate output frame (frame skip) */
  TIFB_FILL_ZERO,	/**< Fill output frame with zeros */
  TIFB_FILL_VALUES,	/**< Fill output frame with a user given value */
  TIFB_FILL_WITH_FILE,	/**< Fill output frame with a user given file (a raw data of tensor/tensors)
			     If the filesize is smaller, the reset is filled with 0 */
  TIFB_FILL_WITH_FILE_RPT,	/**< Fill output frame with a user given file (a raw data of tensor/tensors)
				     If the filesize is smally, the file is repeatedly used */
  TIFB_REPEAT_PREVIOUS_FRAME,	/**< Resend the previous output frame. If this is the first, send ZERO values. */
  TIFB_TENSORPICK, /**< Choose nth tensor (or tensors) among tensors */
  TIFB_END,
} tensor_if_behavior;

/**
 * @brief Internal data structure for value
 */
typedef struct
{
  tensor_type type;
  tensor_element data;
} tensor_if_data_s;

/**
 * @brief Internal data structure for supplied value
 */
typedef struct
{
  guint32 num;
  tensor_type type;
  tensor_element data[2];
} tensor_if_sv_s;

/**
 * @brief Tensor If data structure
 */
struct _GstTensorIf
{
  GstBaseTransform element;     /**< This is the parent object */
  GstPad *sinkpad;
  GSList *srcpads;
  gboolean silent;

  GstTensorsConfig in_config; /**< input tensor info */
  GstTensorsConfig out_config; /**< output tensor info */
  guint32 num_srcpads;
  gboolean have_group_id;
  guint group_id;

  tensor_if_compared_value cv; /**< compared value */
  tensor_if_operator op;
  tensor_if_behavior act_then;
  tensor_if_behavior act_else;
  tensor_if_sv_s sv[2];
  GList *cv_option;
  GList *then_option;
  GList *else_option;
};

/**
 * @brief GstTensorIfClass inherits GstElementClass
 */
struct _GstTensorIfClass
{
  GstBaseTransformClass parent_class;   /**< Inherits GstBaseTransformClass */
};

/**
 * @brief Get Type function required for gst elements
 */
GType gst_tensor_if_get_type (void);

G_END_DECLS

#endif /* __GST_TENSOR_IF_H__ */

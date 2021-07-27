/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd.
 *
 * @file	tensor_sparse_enc.h
 * @date	27 Jul 2021
 * @brief	GStreamer element to encode sparse tensors into dense tensors
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_SPARSE_ENC_H__
#define __GST_TENSOR_SPARSE_ENC_H__

#include <gst/gst.h>
#include <tensor_common.h>
#include "tensor_sparse_util.h"

G_BEGIN_DECLS

#define GST_TYPE_TENSOR_SPARSE_ENC \
  (gst_tensor_sparse_enc_get_type())
#define GST_TENSOR_SPARSE_ENC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_TENSOR_SPARSE_ENC,GstTensorSparseEnc))
#define GST_TENSOR_SPARSE_ENC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_TENSOR_SPARSE_ENC,GstTensorSparseEncClass))
#define GST_IS_TENSOR_SPARSE_ENC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_TENSOR_SPARSE_ENC))
#define GST_IS_TENSOR_SPARSE_ENC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_TENSOR_SPARSE_ENC))

typedef struct _GstTensorSparseEnc GstTensorSparseEnc;
typedef struct _GstTensorSparseEncClass GstTensorSparseEncClass;

/**
 * @brief GstTensorSparseEnc data structure.
 */
struct _GstTensorSparseEnc
{
  GstElement element; /**< parent object */
  GstPad *sinkpad; /**< sink pad */
  GstPad *srcpad; /**< src pad */

  /* <private> */
  GstTensorsConfig in_config; /**< input tensors config */
  gboolean silent; /**< true to print minimized log */
};

/**
 * @brief GstTensorSparseClass data structure.
 */
struct _GstTensorSparseEncClass
{
  GstElementClass parent_class; /**< parent class */
};

/**
 * @brief Function to get type of tensor_sparse.
 */
GType gst_tensor_sparse_enc_get_type (void);

G_END_DECLS

#endif /* __GST_TENSOR_SPARSE_ENC_H__ */

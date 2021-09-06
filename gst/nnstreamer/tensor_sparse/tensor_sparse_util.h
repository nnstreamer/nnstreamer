/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer Sparse Tensor support
 * Copyright (C) 2021 Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 */
/**
 * @file	tensor_sparse_util.h
 * @date	06 Jul 2021
 * @brief	Util functions for tensor_sparse encoder and decoder.
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __GST_TENSOR_SPARSE_UTIL_H__
#define __GST_TENSOR_SPARSE_UTIL_H__

#include <gst/gst.h>
#include <tensor_typedef.h>

G_BEGIN_DECLS

/**
 * @brief Make dense tensor with input sparse tensor.
 * @param[in,out] meta tensor meta structure to be updated
 * @param[in] mem gst-memory of sparse tensor data
 * @return pointer of GstMemory with dense tensor data or NULL on error. Caller should handle this newly allocated memory.
 */
extern GstMemory *
gst_tensor_sparse_to_dense (GstTensorMetaInfo * meta, GstMemory * mem);

/**
 * @brief Make sparse tensor with input dense tensor.
 * @param[in,out] meta tensor meta structure to be updated
 * @param[in] mem gst-memory of dense tensor data
 * @return pointer of GstMemory with sparse tensor data or NULL on error. Caller should handle this newly allocated memory.
 */
extern GstMemory *
gst_tensor_sparse_from_dense (GstTensorMetaInfo * meta, GstMemory * mem);

G_END_DECLS
#endif /* __GST_TENSOR_SPARSE_UTIL_H__ */

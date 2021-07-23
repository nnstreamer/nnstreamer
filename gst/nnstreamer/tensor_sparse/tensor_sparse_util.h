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
#include <tensor_common.h>

/**
 * @brief NYI. Make dense tensor with input sparse tensor.
 * @param[in,out] meta tensor meta structure to be updated
 * @param[in] in pointer of input sparse tensor data
 * @param[out] out pointer of output dense tensor data. Assume that it's already allocated.
 * @return TRUE if no error
 */
extern gboolean
gst_tensor_sparse_to_dense (GstTensorMetaInfo * meta, gpointer in, gpointer out);

/**
 * @brief NYI. Make sparse tensor with input dense tensor.
 * @param[in,out] meta tensor meta structure to be updated
 * @param[in] in pointer of input dense tensor data
 * @param[out] out pointer of output sparse tensor data. Assume that it's already allocated.
 * @param TRUE if no error
 */
extern gboolean
gst_tensor_sparse_from_dense (GstTensorMetaInfo * meta, gpointer in, gpointer out);

#endif /* __GST_TENSOR_SPARSE_UTIL_H__ */

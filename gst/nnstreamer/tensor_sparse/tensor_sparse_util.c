/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer Sparse Tensor support
 * Copyright (C) 2021 Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 */
/**
 * @file	tensor_sparse_util.c
 * @date	27 Jul 2021
 * @brief	Util functions for tensor_sparse encoder and decoder.
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <tensor_common.h>
#include <tensor_data.h>
#include "tensor_sparse_util.h"

/**
 * @brief Make dense tensor with input sparse tensor.
 * @param[in,out] meta tensor meta structure to be updated
 * @param[in] in pointer of input sparse tensor data
 * @return pointer of GstMemory with dense tensor data or NULL on error. Caller should handle this newly allocated memory.
 */
GstMemory *
gst_tensor_sparse_to_dense (GstTensorMetaInfo * meta, gpointer in)
{
  guint i, nnz;
  guint8 *output, *input;
  guint *indices;
  gsize output_size, element_size;

  meta->format = _NNS_TENSOR_FORMAT_STATIC;

  element_size = gst_tensor_get_element_size (meta->type);
  output_size = gst_tensor_meta_info_get_data_size (meta);

  if (element_size == 0 || output_size == 0) {
    nns_loge ("Got invalid meta info");
    return NULL;
  }

  output = (guint8 *) g_malloc0 (output_size);

  nnz = meta->sparse_info.nnz;
  input = (guint8 *) in + gst_tensor_meta_info_get_header_size (meta);
  indices = ((guint *) ((guint8 *) input + element_size * nnz));

  for (i = 0; i < nnz; ++i) {
    switch (meta->type) {
      case _NNS_INT32:
        ((int32_t *) output)[indices[i]] = ((int32_t *) input)[i];
        break;
      case _NNS_UINT32:
        ((uint32_t *) output)[indices[i]] = ((uint32_t *) input)[i];
        break;
      case _NNS_INT16:
        ((int16_t *) output)[indices[i]] = ((int16_t *) input)[i];
        break;
      case _NNS_UINT16:
        ((uint16_t *) output)[indices[i]] = ((uint16_t *) input)[i];
        break;
      case _NNS_INT8:
        ((int8_t *) output)[indices[i]] = ((int8_t *) input)[i];
        break;
      case _NNS_UINT8:
        ((uint8_t *) output)[indices[i]] = ((uint8_t *) input)[i];
        break;
      case _NNS_FLOAT64:
        ((double *) output)[indices[i]] = ((double *) input)[i];
        break;
      case _NNS_FLOAT32:
        ((float *) output)[indices[i]] = ((float *) input)[i];
        break;
      case _NNS_INT64:
        ((int64_t *) output)[indices[i]] = ((int64_t *) input)[i];
        break;
      case _NNS_UINT64:
        ((uint64_t *) output)[indices[i]] = ((uint64_t *) input)[i];
        break;
      default:
        nns_loge ("Error occured during get tensor value");
        g_free (output);

        return NULL;
    }
  }

  return gst_memory_new_wrapped (0, output, output_size, 0,
      output_size, output, g_free);
}

/**
 * @brief Make sparse tensor with input dense tensor.
 * @param[in,out] meta tensor meta structure to be updated
 * @param[in] in pointer of input dense tensor data
 * @return pointer of GstMemory with sparse tensor data or NULL on error. Caller should handle this newly allocated memory.
 */
GstMemory *
gst_tensor_sparse_from_dense (GstTensorMetaInfo * meta, gpointer in)
{
  guint i, nnz = 0;
  guint8 *output;
  tensor_type data_type;
  void *values;
  guint *indices;
  gsize output_size;
  gsize header_size = gst_tensor_meta_info_get_header_size (meta);
  gsize element_size = gst_tensor_get_element_size (meta->type);
  gulong element_count = gst_tensor_get_element_count (meta->dimension);

  if (element_size == 0 || element_count == 0) {
    nns_loge ("Got invalid meta info");
    return NULL;
  }

  /** alloc maximum possible size of memory */
  values = g_malloc0 (element_size * element_count);
  indices = g_malloc0 (sizeof (guint) * element_count);

  data_type = meta->type;

  /** Consider using macro to reduce loc and readability */
  for (i = 0; i < element_count; ++i) {
    switch (data_type) {
      case _NNS_INT32:
        if (((int32_t *) in)[i] != 0) {
          ((int32_t *) values)[nnz] = ((int32_t *) in)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_UINT32:
        if (((uint32_t *) in)[i] != 0) {
          ((uint32_t *) values)[nnz] = ((uint32_t *) in)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_INT16:
        if (((int16_t *) in)[i] != 0) {
          ((int16_t *) values)[nnz] = ((int16_t *) in)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_UINT16:
        if (((uint16_t *) in)[i] != 0) {
          ((uint16_t *) values)[nnz] = ((uint16_t *) in)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_INT8:
        if (((int8_t *) in)[i] != 0) {
          ((int8_t *) values)[nnz] = ((int8_t *) in)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_UINT8:
        if (((uint8_t *) in)[i] != 0) {
          ((uint8_t *) values)[nnz] = ((uint8_t *) in)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_FLOAT64:
        if (((double *) in)[i] != 0) {
          ((double *) values)[nnz] = ((double *) in)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_FLOAT32:
        if (((float *) in)[i] != 0) {
          ((float *) values)[nnz] = ((float *) in)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_INT64:
        if (((int64_t *) in)[i] != 0) {
          ((int64_t *) values)[nnz] = ((int64_t *) in)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_UINT64:
        if (((uint64_t *) in)[i] != 0) {
          ((uint64_t *) values)[nnz] = ((uint64_t *) in)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      default:
        nns_loge ("Error occured during get tensor value");
        g_free (values);
        g_free (indices);

        return NULL;
    }
  }

  /** update meta nnz info */
  meta->sparse_info.nnz = nnz;

  /** write to output buffer */
  output_size = element_size * nnz + sizeof (guint) * nnz;

  /** add meta info header */
  output_size += header_size;
  output = g_malloc0 (output_size);

  gst_tensor_meta_info_update_header (meta, output);

  memcpy (output + header_size, values, element_size * nnz);
  memcpy (output + header_size + (element_size * nnz),
      indices, sizeof (guint) * nnz);

  g_free (values);
  g_free (indices);

  return gst_memory_new_wrapped (0, output, output_size, 0,
      output_size, output, g_free);
}

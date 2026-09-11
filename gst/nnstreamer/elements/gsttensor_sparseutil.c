/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer / NNStreamer Sparse Tensor support
 * Copyright (C) 2021 Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 */
/**
 * @file	gsttensor_sparseutil.c
 * @date	27 Jul 2021
 * @brief	Util functions for tensor_sparse encoder and decoder.
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <string.h>
#include <tensor_common.h>
#include <tensor_data.h>
#include "gsttensor_sparseutil.h"

/**
 * @brief Make dense tensor with input sparse tensor.
 * @param[in,out] meta tensor meta structure to be updated
 * @param[in] mem gst-memory of sparse tensor data
 * @return pointer of GstMemory with dense tensor data or NULL on error. Caller should handle this newly allocated memory.
 */
GstMemory *
gst_tensor_sparse_to_dense (GstTensorMetaInfo * meta, GstMemory * mem)
{
  GstMemory *dense = NULL;
  GstMapInfo map;
  guint i, nnz;
  guint8 *output, *input;
  guint *indices;
  gsize output_size, element_size, header_size;
  gulong element_count;

  if (!gst_memory_map (mem, &map, GST_MAP_READ)) {
    nns_loge ("Failed to map given memory");
    return NULL;
  }

  gst_tensor_meta_info_init (meta);
  if (map.size < gst_tensor_meta_info_get_header_size (meta)) {
    nns_loge ("Given memory is too small to hold a tensor meta header");
    goto done;
  }

  if (!gst_tensor_meta_info_parse_header (meta, map.data)) {
    nns_loge ("Failed to parse meta info from given memory");
    goto done;
  }

  nnz = meta->sparse_info.nnz;
  header_size = gst_tensor_meta_info_get_header_size (meta);

  if (header_size == 0) {
    nns_loge ("Cannot get the header size of the meta info, version 0x%x",
        meta->version);
    goto done;
  }

  meta->format = _NNS_TENSOR_FORMAT_STATIC;

  element_size = gst_tensor_get_element_size (meta->type);
  element_count = gst_tensor_get_element_count (meta->dimension);
  output_size = gst_tensor_meta_info_get_data_size (meta);

  if (element_size == 0 || output_size == 0) {
    nns_loge ("Got invalid meta info");
    goto done;
  }

  if (element_count != output_size / element_size) {
    nns_loge ("The %lu elements of the meta info do not fit the %"
        G_GSIZE_FORMAT " bytes of the tensor they describe", element_count,
        output_size);
    goto done;
  }

  if (nnz > (map.size - header_size) / (element_size + sizeof (guint))) {
    nns_loge ("Given memory holds %" G_GSIZE_FORMAT
        " bytes, too small for the %u non-zero elements of the meta info",
        map.size, nnz);
    goto done;
  }

  input = map.data + header_size;
  indices = (guint *) (input + element_size * nnz);

  output = (guint8 *) g_malloc0 (output_size);

  for (i = 0; i < nnz; ++i) {
    guint index;

    /* one read: unaligned for some types, and check and use must agree */
    memcpy (&index, (guint8 *) indices + i * sizeof (guint), sizeof (guint));

    if (index >= element_count) {
      nns_loge ("Sparse tensor index %u is out of the %lu elements of the meta"
          " info", index, element_count);
      g_free (output);
      goto done;
    }

    switch (meta->type) {
      case _NNS_INT32:
        ((int32_t *) output)[index] = ((int32_t *) input)[i];
        break;
      case _NNS_UINT32:
        ((uint32_t *) output)[index] = ((uint32_t *) input)[i];
        break;
      case _NNS_INT16:
        ((int16_t *) output)[index] = ((int16_t *) input)[i];
        break;
      case _NNS_UINT16:
        ((uint16_t *) output)[index] = ((uint16_t *) input)[i];
        break;
      case _NNS_INT8:
        ((int8_t *) output)[index] = ((int8_t *) input)[i];
        break;
      case _NNS_UINT8:
        ((uint8_t *) output)[index] = ((uint8_t *) input)[i];
        break;
      case _NNS_FLOAT64:
        ((double *) output)[index] = ((double *) input)[i];
        break;
      case _NNS_FLOAT32:
        ((float *) output)[index] = ((float *) input)[i];
        break;
      case _NNS_INT64:
        ((int64_t *) output)[index] = ((int64_t *) input)[i];
        break;
      case _NNS_UINT64:
        ((uint64_t *) output)[index] = ((uint64_t *) input)[i];
        break;
      default:
        nns_loge ("Error occurred during get tensor value");
        g_free (output);
        goto done;
    }
  }

  dense = gst_memory_new_wrapped (0, output, output_size, 0, output_size,
      output, g_free);

done:
  gst_memory_unmap (mem, &map);
  return dense;
}

/**
 * @brief Make sparse tensor with input dense tensor.
 * @param[in,out] meta tensor meta structure to be updated
 * @param[in] mem gst-memory of dense tensor data
 * @return pointer of GstMemory with sparse tensor data or NULL on error. Caller should handle this newly allocated memory.
 */
GstMemory *
gst_tensor_sparse_from_dense (GstTensorMetaInfo * meta, GstMemory * mem)
{
  GstMemory *sparse = NULL;
  GstMapInfo map;
  guint i, nnz = 0;
  guint8 *output;
  tensor_type data_type;
  void *values;
  guint *indices;
  gsize output_size, header_size, element_size;
  gulong element_count;

  if (!gst_memory_map (mem, &map, GST_MAP_READ)) {
    nns_loge ("Failed to map given memory");
    return NULL;
  }

  header_size = gst_tensor_meta_info_get_header_size (meta);
  element_size = gst_tensor_get_element_size (meta->type);
  element_count = gst_tensor_get_element_count (meta->dimension);

  if (element_size == 0 || element_count == 0) {
    nns_loge ("Got invalid meta info");
    goto done;
  }

  if (element_count > map.size / element_size
      || element_count > G_MAXSIZE / sizeof (guint)) {
    nns_loge ("Given memory holds %" G_GSIZE_FORMAT
        " bytes, too small for the %lu elements of the meta info",
        map.size, element_count);
    goto done;
  }

  /** alloc maximum possible size of memory */
  values = g_malloc0 (element_size * element_count);
  indices = g_malloc0 (sizeof (guint) * element_count);

  data_type = meta->type;

  /** Consider using macro to reduce loc and readability */
  for (i = 0; i < element_count; ++i) {
    switch (data_type) {
      case _NNS_INT32:
        if (((int32_t *) map.data)[i] != 0) {
          ((int32_t *) values)[nnz] = ((int32_t *) map.data)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_UINT32:
        if (((uint32_t *) map.data)[i] != 0) {
          ((uint32_t *) values)[nnz] = ((uint32_t *) map.data)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_INT16:
        if (((int16_t *) map.data)[i] != 0) {
          ((int16_t *) values)[nnz] = ((int16_t *) map.data)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_UINT16:
        if (((uint16_t *) map.data)[i] != 0) {
          ((uint16_t *) values)[nnz] = ((uint16_t *) map.data)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_INT8:
        if (((int8_t *) map.data)[i] != 0) {
          ((int8_t *) values)[nnz] = ((int8_t *) map.data)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_UINT8:
        if (((uint8_t *) map.data)[i] != 0) {
          ((uint8_t *) values)[nnz] = ((uint8_t *) map.data)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_FLOAT64:
        if (((double *) map.data)[i] != 0) {
          ((double *) values)[nnz] = ((double *) map.data)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_FLOAT32:
        if (((float *) map.data)[i] != 0) {
          ((float *) values)[nnz] = ((float *) map.data)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_INT64:
        if (((int64_t *) map.data)[i] != 0) {
          ((int64_t *) values)[nnz] = ((int64_t *) map.data)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      case _NNS_UINT64:
        if (((uint64_t *) map.data)[i] != 0) {
          ((uint64_t *) values)[nnz] = ((uint64_t *) map.data)[i];
          indices[nnz] = i;
          nnz += 1;
        }
        break;
      default:
        nns_loge ("Error occurred during get tensor value");
        g_free (values);
        g_free (indices);
        goto done;
    }
  }

  /** update meta nnz info */
  meta->format = _NNS_TENSOR_FORMAT_SPARSE;
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

  sparse = gst_memory_new_wrapped (0, output, output_size, 0, output_size,
      output, g_free);

done:
  gst_memory_unmap (mem, &map);
  return sparse;
}

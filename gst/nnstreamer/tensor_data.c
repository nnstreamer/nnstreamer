/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	tensor_data.c
 * @date	10 Mar 2021
 * @brief	Internal functions to handle various tensor type and value.
 * @see	http://github.com/nnstreamer/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug	No known bugs except for NYI items
 */

#include "tensor_data.h"
#include "nnstreamer_log.h"
#include "nnstreamer_plugin_api.h"

/**
 * @brief Macro to set data in struct.
 */
#define td_set_data(td,v,dtype) do { \
    (td)->data._##dtype = *((dtype *) v); \
  } while (0)

/**
 * @brief Macro to get data from struct.
 */
#define td_get_data(td,v,dtype) do { \
    *((dtype *) v) = (td)->data._##dtype; \
  } while (0)

/**
 * @brief Macro for typecast.
 */
#define td_typecast_to(td,itype,otype) do { \
    itype in_val = (td)->data._##itype; \
    otype out_val = (otype) in_val; \
    (td)->data._##otype = out_val; \
  } while (0)

#define td_typecast(td,otype) do { \
    switch ((td)->type) { \
      case _NNS_INT32: td_typecast_to (td, int32_t, otype); break; \
      case _NNS_UINT32: td_typecast_to (td, uint32_t, otype); break; \
      case _NNS_INT16: td_typecast_to (td, int16_t, otype); break; \
      case _NNS_UINT16:  td_typecast_to (td, uint16_t, otype); break; \
      case _NNS_INT8: td_typecast_to (td, int8_t, otype); break; \
      case _NNS_UINT8: td_typecast_to (td, uint8_t, otype); break; \
      case _NNS_FLOAT64: td_typecast_to (td, double, otype); break; \
      case _NNS_FLOAT32: td_typecast_to (td, float, otype); break; \
      case _NNS_INT64: td_typecast_to (td, int64_t, otype); break; \
      case _NNS_UINT64: td_typecast_to (td, uint64_t, otype); break; \
      default: g_assert (0); break; \
    } \
  } while (0)

/**
 * @brief Set tensor element data with given type.
 * @param td struct for tensor data
 * @param type tensor type
 * @param value pointer of tensor element value
 * @return TRUE if no error
 */
gboolean
gst_tensor_data_set (tensor_data_s * td, tensor_type type, gpointer value)
{
  g_return_val_if_fail (td != NULL, FALSE);
  g_return_val_if_fail (value != NULL, FALSE);

  td->data._int64_t = 0;
  td->type = _NNS_END;

  switch (type) {
    case _NNS_INT32:
      td_set_data (td, value, int32_t);
      break;
    case _NNS_UINT32:
      td_set_data (td, value, uint32_t);
      break;
    case _NNS_INT16:
      td_set_data (td, value, int16_t);
      break;
    case _NNS_UINT16:
      td_set_data (td, value, uint16_t);
      break;
    case _NNS_INT8:
      td_set_data (td, value, int8_t);
      break;
    case _NNS_UINT8:
      td_set_data (td, value, uint8_t);
      break;
    case _NNS_FLOAT64:
      td_set_data (td, value, double);
      break;
    case _NNS_FLOAT32:
      td_set_data (td, value, float);
      break;
    case _NNS_INT64:
      td_set_data (td, value, int64_t);
      break;
    case _NNS_UINT64:
      td_set_data (td, value, uint64_t);
      break;
    default:
      nns_logw ("Unknown tensor type %d", type);
      return FALSE;
  }

  td->type = type;
  return TRUE;
}

/**
 * @brief Get tensor element value.
 * @param td struct for tensor data
 * @param value pointer of tensor element value
 * @return TRUE if no error
 */
gboolean
gst_tensor_data_get (tensor_data_s * td, gpointer value)
{
  g_return_val_if_fail (td != NULL, FALSE);
  g_return_val_if_fail (value != NULL, FALSE);

  switch (td->type) {
    case _NNS_INT32:
      td_get_data (td, value, int32_t);
      break;
    case _NNS_UINT32:
      td_get_data (td, value, uint32_t);
      break;
    case _NNS_INT16:
      td_get_data (td, value, int16_t);
      break;
    case _NNS_UINT16:
      td_get_data (td, value, uint16_t);
      break;
    case _NNS_INT8:
      td_get_data (td, value, int8_t);
      break;
    case _NNS_UINT8:
      td_get_data (td, value, uint8_t);
      break;
    case _NNS_FLOAT64:
      td_get_data (td, value, double);
      break;
    case _NNS_FLOAT32:
      td_get_data (td, value, float);
      break;
    case _NNS_INT64:
      td_get_data (td, value, int64_t);
      break;
    case _NNS_UINT64:
      td_get_data (td, value, uint64_t);
      break;
    default:
      nns_logw ("Unknown tensor type %d", td->type);
      return FALSE;
  }

  return TRUE;
}

/**
 * @brief Typecast tensor element data.
 * @param td struct for tensor data
 * @param type tensor type to be transformed
 * @return TRUE if no error
 */
gboolean
gst_tensor_data_typecast (tensor_data_s * td, tensor_type type)
{
  gboolean is_float;

  g_return_val_if_fail (td != NULL, FALSE);

  /* do nothing when transform to same type */
  if (td->type != type) {
    is_float = (td->type == _NNS_FLOAT32 || td->type == _NNS_FLOAT64);

    switch (type) {
      case _NNS_INT32:
        td_typecast (td, int32_t);
        break;
      case _NNS_UINT32:
        if (is_float) {
          td_typecast (td, int32_t);
          td->type = _NNS_INT32;
        }
        td_typecast (td, uint32_t);
        break;
      case _NNS_INT16:
        td_typecast (td, int16_t);
        break;
      case _NNS_UINT16:
        if (is_float) {
          td_typecast (td, int16_t);
          td->type = _NNS_INT16;
        }
        td_typecast (td, uint16_t);
        break;
      case _NNS_INT8:
        td_typecast (td, int8_t);
        break;
      case _NNS_UINT8:
        if (is_float) {
          td_typecast (td, int8_t);
          td->type = _NNS_INT8;
        }
        td_typecast (td, uint8_t);
        break;
      case _NNS_FLOAT64:
        td_typecast (td, double);
        break;
      case _NNS_FLOAT32:
        td_typecast (td, float);
        break;
      case _NNS_INT64:
        td_typecast (td, int64_t);
        break;
      case _NNS_UINT64:
        if (is_float) {
          td_typecast (td, int64_t);
          td->type = _NNS_INT64;
        }
        td_typecast (td, uint64_t);
        break;
      default:
        nns_logw ("Unknown tensor type %d", type);
        return FALSE;
    }

    td->type = type;
  }

  return TRUE;
}

/**
 * @brief Typecast tensor element value.
 * @param input pointer of input tensor data
 * @param in_type input tensor type
 * @param output pointer of output tensor data
 * @param out_type output tensor type
 * @return TRUE if no error
 */
gboolean
gst_tensor_data_raw_typecast (gpointer input, tensor_type in_type,
    gpointer output, tensor_type out_type)
{
  tensor_data_s td;

  g_return_val_if_fail (input != NULL, FALSE);
  g_return_val_if_fail (output != NULL, FALSE);
  g_return_val_if_fail (in_type != _NNS_END, FALSE);
  g_return_val_if_fail (out_type != _NNS_END, FALSE);

  gst_tensor_data_set (&td, in_type, input);
  gst_tensor_data_typecast (&td, out_type);
  gst_tensor_data_get (&td, output);
  return TRUE;
}

/**
 * @brief Calculate average value of the tensor.
 * @param raw pointer of raw tensor data
 * @param length byte size of raw tensor data
 * @param type tensor type
 * @return average value
 */
gdouble
gst_tensor_data_raw_average (gpointer raw, gsize length, tensor_type type)
{
  gdouble value, average;
  gulong i, num;
  gsize element_size;
  guint8 *data;

  g_return_val_if_fail (raw != NULL, 0.0);
  g_return_val_if_fail (length > 0, 0.0);
  g_return_val_if_fail (type != _NNS_END, 0.0);

  element_size = gst_tensor_get_element_size (type);
  num = length / element_size;

  average = 0.0;
  for (i = 0; i < num; ++i) {
    /* extract value and typecast to double */
    data = (guint8 *) raw + element_size * i;
    gst_tensor_data_raw_typecast (data, type, &value, _NNS_FLOAT64);

    average = (value - average) / (i + 1) + average;
  }

  return average;
}

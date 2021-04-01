/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	tensor_data.h
 * @date	10 Mar 2021
 * @brief	Internal functions to handle various tensor type and value.
 * @see	http://github.com/nnstreamer/nnstreamer
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug	No known bugs except for NYI items
 */

#ifndef __NNS_TENSOR_DATA_H__
#define __NNS_TENSOR_DATA_H__

#include <glib.h>
#include <tensor_typedef.h>

G_BEGIN_DECLS
/**
 * @brief Structure for tensor data.
 */
typedef struct
{
  tensor_type type;
  tensor_element data;
} tensor_data_s;

/**
 * @brief Set tensor element data with given type.
 * @param td struct for tensor data
 * @param type tensor type
 * @param value pointer of tensor element value
 * @return TRUE if no error
 */
extern gboolean
gst_tensor_data_set (tensor_data_s * td, tensor_type type, gpointer value);

/**
 * @brief Get tensor element value.
 * @param td struct for tensor data
 * @param value pointer of tensor element value
 * @return TRUE if no error
 */
extern gboolean
gst_tensor_data_get (tensor_data_s * td, gpointer value);

/**
 * @brief Typecast tensor element data.
 * @param td struct for tensor data
 * @param type tensor type to be transformed
 * @return TRUE if no error
 */
extern gboolean
gst_tensor_data_typecast (tensor_data_s * td, tensor_type type);

/**
 * @brief Typecast tensor element value.
 * @param input pointer of input tensor data
 * @param in_type input tensor type
 * @param output pointer of output tensor data
 * @param out_type output tensor type
 * @return TRUE if no error
 */
extern gboolean
gst_tensor_data_raw_typecast (gpointer input, tensor_type in_type,
    gpointer output, tensor_type out_type);

/**
 * @brief Calculate average value of the tensor.
 * @param raw pointer of raw tensor data
 * @param length byte size of raw tensor data
 * @param type tensor type
 * @param result double pointer for average value of given tensor. Caller should release allocated memory.
 * @return TRUE if no error
 */
extern gboolean
gst_tensor_data_raw_average (gpointer raw, gsize length, tensor_type type,
    gdouble ** result);

/**
 * @brief Calculate average value of the tensor per channel (the first dim).
 * @param raw pointer of raw tensor data
 * @param length byte size of raw tensor data
 * @param type tensor type
 * @param tensor_dim tensor dimension
 * @param results double array contains average values of each channel. Caller should release allocated array.
 * @return TRUE if no error
 */
extern gboolean
gst_tensor_data_raw_average_per_channel (gpointer raw, gsize length,
    tensor_type type, tensor_dim dim, gdouble ** results);

/**
 * @brief Calculate standard deviation of the tensor.
 * @param raw pointer of raw tensor data
 * @param length byte size of raw tensor data
 * @param type tensor type
 * @param average average value of given tensor
 * @param result double pointer for standard deviation of given tensor. Caller should release allocated memory.
 * @return TRUE if no error
 */
extern gboolean
gst_tensor_data_raw_std (gpointer raw, gsize length, tensor_type type,
    gdouble * average, gdouble ** result);

/**
 * @brief Calculate standard deviation of the tensor per channel (the first dim).
 * @param raw pointer of raw tensor data
 * @param length byte size of raw tensor data
 * @param type tensor type
 * @param tensor_dim tensor dimension
 * @param averages average values of given tensor per-channel
 * @param results double array contains standard deviation of each channel. Caller should release allocated array.
 * @return TRUE if no error
 */
extern gboolean
gst_tensor_data_raw_std_per_channel (gpointer raw, gsize length, 
    tensor_type type, tensor_dim dim, gdouble * averages, gdouble ** results);

G_END_DECLS
#endif /* __NNS_TENSOR_DATA_H__ */

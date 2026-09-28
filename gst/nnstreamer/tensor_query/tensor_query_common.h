/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * Copyright (C) 2021 Gichan Jang <gichan2.jang@samsung.com>
 *
 * @file   tensor_query_common.h
 * @date   09 July 2021
 * @brief  Utility functions for tensor query
 * @see    https://github.com/nnstreamer/nnstreamer
 * @author Gichan Jang <gichan2.jang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __TENSOR_QUERY_COMMON_H__
#define __TENSOR_QUERY_COMMON_H__

#include "tensor_typedef.h"
#include "tensor_common.h"
#include "tensor_meta.h"
#include <nnstreamer-edge.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Default timeout, in seconds.
 */
#define QUERY_DEFAULT_TIMEOUT_SEC 10

/**
 * @brief protocol options for tensor query.
 */
#define DEFAULT_HOST "localhost"
#define DEFAULT_CONNECT_TYPE (NNS_EDGE_CONNECT_TYPE_TCP)
#define GST_TYPE_QUERY_CONNECT_TYPE (gst_tensor_query_get_connect_type ())

/**
 * @brief Register GEnumValue array for query connect-type property.
 */
GType
gst_tensor_query_get_connect_type (void);

/**
 * @brief Get the tensors config to check the data from a remote peer with, from the caps of the receiving pad.
 */
gboolean
gst_tensor_query_config_from_caps (GstCaps * caps, GstTensorsConfig * config);

/**
 * @brief Check the memories of edge data received from a remote peer against the negotiated tensors config.
 */
gboolean
gst_tensor_query_validate_edge_data (nns_edge_data_h data_h, GstTensorsConfig * config);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* __TENSOR_QUERY_COMMON_H__ */

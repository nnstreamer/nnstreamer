/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        vsi_nn_pub.h
 * @date        22 Sep 2026
 * @brief       Minimal stand-in for the ovxlib public header, for testing only
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 *
 * @details
 * The vivante tensor-filter sub-plugin includes <ovx/vsi_nn_pub.h>, which is
 * shipped with the proprietary Vivante/acuity SDK and is not available to the
 * CI machines. This header declares only the handful of types, constants and
 * functions that the sub-plugin actually refers to, so that the sub-plugin can
 * be compiled and exercised against a mock ovxlib.
 *
 * The declarations follow the original ovxlib headers, which are published as
 * part of VeriSilicon/TIM-VX (src/tim/vx/internal/include/: vsi_nn_pub.h,
 * vsi_nn_types.h, vsi_nn_tensor.h and vsi_nn_graph.h). Everything the
 * sub-plugin does not use is omitted: contexts, nodes, operations, quantization
 * parameters, the remaining members of the tensor attributes and of the graph,
 * and every other data type of the SDK. The sub-plugin only ever handles the
 * two structures through pointers returned by the SDK, so this header is not
 * required to be ABI-compatible with the real one.
 *
 * Note that `vsi_size_t` of the original header is either `size_t` or
 * `uint32_t` depending on a build option of the SDK. It is declared as
 * `uint32_t` here because the sub-plugin assigns the tensor sizes to the
 * `uint32_t` dimension array of GstTensorInfo.
 */

#ifndef __NNS_TEST_OVX_VSI_NN_PUB_H__
#define __NNS_TEST_OVX_VSI_NN_PUB_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VSI_NN_MAX_DIM_NUM (8)

#define VSI_SUCCESS (0)
#define VSI_FAILURE (-1)

typedef int32_t vsi_status;
typedef int32_t vsi_bool;
typedef uint32_t vsi_size_t;
typedef uint32_t vsi_nn_tensor_id_t;

/**
 * @brief The data type of a tensor, a subset of the original enumeration.
 */
typedef enum
{
  VSI_NN_TYPE_NONE = 0,
  VSI_NN_TYPE_INT8,
  VSI_NN_TYPE_INT16,
  VSI_NN_TYPE_INT32,
  VSI_NN_TYPE_INT64,
  VSI_NN_TYPE_UINT8,
  VSI_NN_TYPE_UINT16,
  VSI_NN_TYPE_UINT32,
  VSI_NN_TYPE_UINT64,
  VSI_NN_TYPE_FLOAT16,
  VSI_NN_TYPE_FLOAT32,
  VSI_NN_TYPE_FLOAT64
} vsi_nn_type_e;

/**
 * @brief The data type description of a tensor.
 */
typedef struct
{
  vsi_nn_type_e vx_type; /**< The data type of each element */
} vsi_nn_dtype_t;

/**
 * @brief The attributes of a tensor.
 */
typedef struct
{
  vsi_size_t size[VSI_NN_MAX_DIM_NUM]; /**< The size of each dimension */
  uint32_t dim_num; /**< The number of the valid dimensions */
  vsi_nn_dtype_t dtype; /**< The data type description */
} vsi_nn_tensor_attr_t;

/**
 * @brief A tensor of a neural network graph.
 */
typedef struct
{
  vsi_nn_tensor_attr_t attr; /**< The attributes of this tensor */
} vsi_nn_tensor_t;

/**
 * @brief The list of the tensor identifiers of a graph.
 */
typedef struct
{
  vsi_nn_tensor_id_t *tensors; /**< The identifiers of the tensors */
  uint32_t num; /**< The number of the tensors */
} vsi_nn_graph_tensors_t;

/**
 * @brief A neural network graph.
 */
typedef struct
{
  vsi_nn_graph_tensors_t input; /**< The input tensors of this graph */
  vsi_nn_graph_tensors_t output; /**< The output tensors of this graph */
} vsi_nn_graph_t;

/**
 * @brief Get the tensor of the given identifier from the given graph.
 */
extern vsi_nn_tensor_t *vsi_nn_GetTensor (const vsi_nn_graph_t *graph,
    vsi_nn_tensor_id_t id);

/**
 * @brief Copy the contents of the given tensor into the given buffer.
 */
extern vsi_status vsi_nn_CopyTensorToBuffer (const vsi_nn_graph_t *graph,
    vsi_nn_tensor_t *tensor, void *buffer);

#ifdef __cplusplus
}
#endif

#endif /* __NNS_TEST_OVX_VSI_NN_PUB_H__ */

/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    qnn_mock.h
 * @date    22 Nov 2026
 * @brief   Controls of the mock QNN backend and model for the sub-plugin tests
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * The qnn sub-plugin opens two libraries named by its model property: a model
 * library (QnnModel_composeGraphs, QnnModel_freeGraphsInfo) and a backend
 * library (QnnInterface_getProviders). libqnn_mock_model and
 * libqnn_mock_backend stand in for both.
 *
 * The default model has one graph with one input and two outputs:
 *   in0  uint8 [1, 4, 4, 3]
 *   out0 uint8 [1, 4, 4, 3], element i is (in0[i] + 1) mod 256
 *   sum  int32 [1], the sum of every byte of in0
 * Dimensions are in the QNN order (outermost first). The model can also be
 * told to quantize in0, to give in0 or out0 more dimensions of 1 in front, or
 * to have more int32 [1] inputs ("extra_in<n>") or outputs ("extra<n>"); the
 * backend executes only the default shapes, and checks that a quantized in0
 * still carries the parameters below.
 */
#ifndef __QNN_MOCK_H__
#define __QNN_MOCK_H__

#include <stdbool.h>

#include <QnnInterface.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Errors of the mock, in the range the mock owns. */
#define QNN_MOCK_ERROR_INJECTED 0x7a01
#define QNN_MOCK_ERROR_INVALID_HANDLE 0x7a02
#define QNN_MOCK_ERROR_INVALID_TENSOR 0x7a03
#define QNN_MOCK_ERROR_NOT_FINALIZED 0x7a04

/** @brief The scale and offset of a scale-offset quantized in0. */
#define QNN_MOCK_SCALE (0.25f)
#define QNN_MOCK_OFFSET (7)
/** @brief The scale and offset of axis @a i of a per-axis quantized in0. */
#define QNN_MOCK_AXIS_SCALE(i) (0.5f * (float) ((i) + 1))
#define QNN_MOCK_AXIS_OFFSET(i) ((int32_t) (i))
/** @brief The axis of a per-axis quantized in0. */
#define QNN_MOCK_AXIS (3)
/** @brief The largest rank of in0 the model can describe. */
#define QNN_MOCK_MAX_RANK (32)

/** @brief The errors QnnModel_composeGraphs and QnnModel_freeGraphsInfo return. */
#define QNN_MOCK_MODEL_NO_ERROR 0
#define QNN_MOCK_MODEL_TENSOR_ERROR 1
#define QNN_MOCK_MODEL_GRAPH_ERROR 4

/**
 * @brief A composed graph, laid out as the GraphInfo_t of the QNN SDK
 *        (QnnWrapperUtils.hpp) that model libraries return.
 */
typedef struct {
  Qnn_GraphHandle_t graph;
  char *graphName;
  Qnn_Tensor_t *inputTensors;
  uint32_t numInputTensors;
  Qnn_Tensor_t *outputTensors;
  uint32_t numOutputTensors;
} QnnMockGraphInfo_t;

/**
 * @brief Clear the injected failures, the provider count and the call counters
 *        of the backend.
 */
void qnn_mock_backend_reset (void);

/**
 * @brief Make every later call of the named backend function fail with
 *        QNN_MOCK_ERROR_INJECTED. "getProviders" names
 *        QnnInterface_getProviders; the others are the interface members
 *        (e.g., "backendCreate", "graphExecute").
 */
void qnn_mock_backend_fail (const char *function);

/**
 * @brief Make QnnInterface_getProviders report @a num providers (default 1).
 */
void qnn_mock_backend_set_num_providers (uint32_t num);

/**
 * @brief The number of calls of the named backend function since the last reset.
 */
int qnn_mock_backend_calls (const char *function);

/**
 * @brief The number of backends, contexts and graphs not freed yet.
 */
int qnn_mock_backend_live_objects (void);

/**
 * @brief Restore the default model.
 */
void qnn_mock_model_reset (void);

/**
 * @brief Make QnnModel_composeGraphs fail with QNN_MOCK_MODEL_GRAPH_ERROR.
 */
void qnn_mock_model_fail_compose (void);

/**
 * @brief Make QnnModel_composeGraphs compose @a count copies of the graph.
 */
void qnn_mock_model_set_graph_count (uint32_t count);

/**
 * @brief Set the data type of in0.
 */
void qnn_mock_model_set_input_type (Qnn_DataType_t type);

/**
 * @brief Describe every tensor as a version 2 tensor, with dynamic-dimension
 *        flags (the outermost dimension dynamic) or with none (NULL).
 */
void qnn_mock_model_use_tensor_v2 (bool with_dynamic_flags);

/**
 * @brief Quantize in0: QNN_QUANTIZATION_ENCODING_SCALE_OFFSET with
 *        QNN_MOCK_SCALE and QNN_MOCK_OFFSET, or
 *        QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET on QNN_MOCK_AXIS with
 *        @a num pairs of QNN_MOCK_AXIS_SCALE () and QNN_MOCK_AXIS_OFFSET ().
 */
void qnn_mock_model_set_quantization (Qnn_QuantizationEncoding_t encoding, uint32_t num);

/**
 * @brief Give in0 and out0 @a in_rank and @a out_rank dimensions (4 to
 *        QNN_MOCK_MAX_RANK).
 */
void qnn_mock_model_set_ranks (uint32_t in_rank, uint32_t out_rank);

/**
 * @brief Give the graph @a inputs inputs (1 or more) and @a outputs outputs
 *        (2 or more).
 */
void qnn_mock_model_set_num_tensors (uint32_t inputs, uint32_t outputs);

/**
 * @brief The number of composed graphs not freed by QnnModel_freeGraphsInfo yet.
 */
int qnn_mock_model_live_graphs (void);

#ifdef __cplusplus
}
#endif

#endif /* __QNN_MOCK_H__ */

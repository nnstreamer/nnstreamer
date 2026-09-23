/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        mock_ovxlib.h
 * @date        22 Sep 2026
 * @brief       Control and observation interface of the mock ovxlib
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 *
 * @details
 * The mock ovxlib replaces the Vivante SDK so that the vivante tensor-filter
 * sub-plugin can be built and exercised without an NPU. It is shared by the
 * unit test and by the mock model shared library that the sub-plugin dlopen()s,
 * so a test may configure the behaviour of the model through this interface.
 */

#ifndef __NNS_TEST_MOCK_OVXLIB_H__
#define __NNS_TEST_MOCK_OVXLIB_H__

#include <ovx/vsi_nn_pub.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Restore every knob to its default and clear every counter.
 */
extern void mock_ovxlib_reset (void);

/**
 * @brief Set the number of the input and the output tensors of the model.
 */
extern void mock_ovxlib_set_tensor_num (unsigned int num_input,
    unsigned int num_output);

/**
 * @brief Set the data type that the tensors of the model report.
 */
extern void mock_ovxlib_set_tensor_type (vsi_nn_type_e type);

/**
 * @brief Set the number of the dimensions that the tensors of the model have.
 */
extern void mock_ovxlib_set_tensor_rank (unsigned int rank);

/**
 * @brief Make the tensors claim more dimensions than they can hold.
 * @details
 * The attributes of a tensor hold VSI_NN_MAX_DIM_NUM sizes, so a rank above
 * that describes a tensor the SDK itself could not have stored. Only the rank
 * the tensors report is raised; their sizes and contents stay as they are.
 * Pass 0 to report the rank set by mock_ovxlib_set_tensor_rank() again.
 */
extern void mock_ovxlib_set_tensor_rank_overflow (unsigned int rank);

/**
 * @brief Make vnn_CreateNeuralNetwork() of the model fail.
 */
extern void mock_ovxlib_set_create_fail (int fail);

/**
 * @brief Make vsi_nn_GetTensor() fail for the tensor of the given identifier.
 * @param[in] id The tensor identifier to fail for, or -1 to fail for none.
 */
extern void mock_ovxlib_set_get_tensor_fail (int id);

/**
 * @brief Make vsi_nn_RunGraph() of the model fail.
 */
extern void mock_ovxlib_set_run_fail (int fail);

/**
 * @brief Make the post-process of the model fail.
 */
extern void mock_ovxlib_set_post_process_fail (int fail);

/**
 * @brief Get the number of the graphs that have been created and not released.
 */
extern int mock_ovxlib_get_live_graph (void);

/**
 * @brief Get the number of the times the mock model has been unloaded.
 */
extern int mock_ovxlib_get_model_unload (void);

/**
 * @brief Get the number of the times the post-process of the model has run.
 */
extern int mock_ovxlib_get_post_process (void);

/**
 * @brief Create a graph. To be called by the mock model only.
 */
extern vsi_nn_graph_t *mock_ovxlib_create_graph (const char *model_path);

/**
 * @brief Release a graph. To be called by the mock model only.
 */
extern void mock_ovxlib_release_graph (vsi_nn_graph_t *graph);

/**
 * @brief Fill an input tensor. To be called by the mock model only.
 */
extern vsi_status mock_ovxlib_copy_data_to_tensor (vsi_nn_graph_t *graph,
    vsi_nn_tensor_t *tensor, void *data);

/**
 * @brief Run a graph. To be called by the mock model only.
 */
extern vsi_status mock_ovxlib_run_graph (vsi_nn_graph_t *graph);

/**
 * @brief Run the post-process of a graph. To be called by the mock model only.
 */
extern vsi_status mock_ovxlib_post_process (vsi_nn_graph_t *graph);

/**
 * @brief Record that the mock model has been unloaded.
 */
extern void mock_ovxlib_note_model_unload (void);

#ifdef __cplusplus
}
#endif

#endif /* __NNS_TEST_MOCK_OVXLIB_H__ */

/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        mock_vivante_model.c
 * @date        22 Sep 2026
 * @brief       A mock of the model library that the vivante sub-plugin dlopens
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 *
 * @details
 * The vivante sub-plugin takes a network binary and a shared library generated
 * by the acuity tool for it, and calls the model through the symbols it
 * resolves from that library. This is a stand-in for such a library; it
 * delegates every call to the mock ovxlib, which a test configures. Built
 * without NNS_MOCK_MODEL_FULL it leaves out the symbols that the sub-plugin
 * resolves last, which makes it a model library the sub-plugin has to reject.
 */

#include "mock_ovxlib.h"

void mock_vivante_model_unload (void) __attribute__ ((destructor));

vsi_status vsi_nn_CopyDataToTensor (vsi_nn_graph_t * graph,
    vsi_nn_tensor_t * tensor, void *data);
void vnn_ReleaseNeuralNetwork (vsi_nn_graph_t * graph);
#ifdef NNS_MOCK_MODEL_FULL
vsi_status vsi_nn_RunGraph (vsi_nn_graph_t * graph);
vsi_status vnn_PostProcessNeuralNetwork (vsi_nn_graph_t * graph);
vsi_nn_graph_t *vnn_CreateNeuralNetwork (const char *model_path);
#endif

/** @brief Record the unloading of this library, to observe a dlclose call. */
void
mock_vivante_model_unload (void)
{
  mock_ovxlib_note_model_unload ();
}

/** @brief Copy the given data into the given tensor of the given graph. */
vsi_status
vsi_nn_CopyDataToTensor (vsi_nn_graph_t * graph, vsi_nn_tensor_t * tensor,
    void *data)
{
  return mock_ovxlib_copy_data_to_tensor (graph, tensor, data);
}

/** @brief Release the given neural network. */
void
vnn_ReleaseNeuralNetwork (vsi_nn_graph_t * graph)
{
  mock_ovxlib_release_graph (graph);
}

#ifdef NNS_MOCK_MODEL_FULL
/** @brief Run the given neural network. */
vsi_status
vsi_nn_RunGraph (vsi_nn_graph_t * graph)
{
  return mock_ovxlib_run_graph (graph);
}

/** @brief Post-process the output of the given neural network. */
vsi_status
vnn_PostProcessNeuralNetwork (vsi_nn_graph_t * graph)
{
  return mock_ovxlib_post_process (graph);
}

/** @brief Create a neural network from the given network binary. */
vsi_nn_graph_t *
vnn_CreateNeuralNetwork (const char *model_path)
{
  return mock_ovxlib_create_graph (model_path);
}
#endif /* NNS_MOCK_MODEL_FULL */

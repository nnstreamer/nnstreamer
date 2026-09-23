/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        mock_ovxlib.c
 * @date        22 Sep 2026
 * @brief       A mock of the Vivante ovxlib, for testing the vivante sub-plugin
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs
 *
 * @details
 * A graph of this mock holds (num_input + num_output) tensors, identified by
 * their index: the input tensors come first. Every tensor of a graph has the
 * same data type, and running a graph copies each input tensor into the output
 * tensor of the same index, incrementing every byte by one, so that a test may
 * tell the output of an invocation from its input.
 */

#include <glib.h>
#include <string.h>

#include "mock_ovxlib.h"

#define MOCK_TENSOR_DIM_NUM (4U)

/**
 * @brief A tensor of the mock, holding its contents besides its attributes.
 */
typedef struct
{
  vsi_nn_tensor_t pub; /**< The tensor as the sub-plugin sees it */
  guint8 *data; /**< The contents of this tensor */
  gsize bytes; /**< The size of the contents in bytes */
} mock_tensor_s;

/**
 * @brief A graph of the mock, holding its tensors.
 */
typedef struct
{
  vsi_nn_graph_t pub; /**< The graph as the sub-plugin sees it */
  vsi_nn_tensor_id_t *input_ids; /**< The identifiers of the input tensors */
  vsi_nn_tensor_id_t *output_ids; /**< The identifiers of the output tensors */
  mock_tensor_s *tensors; /**< The input tensors followed by the output ones */
  guint num_tensors; /**< The number of the tensors of this graph */
} mock_graph_s;

static guint mock_num_input = 1U;
static guint mock_num_output = 1U;
static vsi_nn_type_e mock_type = VSI_NN_TYPE_UINT8;
static guint mock_rank = MOCK_TENSOR_DIM_NUM;
static guint mock_rank_claimed = 0;
static int mock_create_fail = 0;
static int mock_get_tensor_fail = -1;
static int mock_run_fail = 0;
static int mock_post_process_fail = 0;
static int mock_live_graph = 0;
static int mock_model_unload = 0;
static int mock_post_process = 0;

/** @brief Restore every knob to its default and clear every counter. */
void
mock_ovxlib_reset (void)
{
  mock_num_input = 1U;
  mock_num_output = 1U;
  mock_type = VSI_NN_TYPE_UINT8;
  mock_rank = MOCK_TENSOR_DIM_NUM;
  mock_rank_claimed = 0;
  mock_create_fail = 0;
  mock_get_tensor_fail = -1;
  mock_run_fail = 0;
  mock_post_process_fail = 0;
  mock_live_graph = 0;
  mock_model_unload = 0;
  mock_post_process = 0;
}

/** @brief Set the number of the input and the output tensors of the model. */
void
mock_ovxlib_set_tensor_num (unsigned int num_input, unsigned int num_output)
{
  mock_num_input = num_input;
  mock_num_output = num_output;
}

/** @brief Set the data type that the tensors of the model report. */
void
mock_ovxlib_set_tensor_type (vsi_nn_type_e type)
{
  mock_type = type;
}

/** @brief Set the number of the dimensions that the tensors have. */
void
mock_ovxlib_set_tensor_rank (unsigned int rank)
{
  mock_rank = MIN (rank, VSI_NN_MAX_DIM_NUM);
}

/** @brief Make the tensors claim more dimensions than they can hold. */
void
mock_ovxlib_set_tensor_rank_overflow (unsigned int rank)
{
  mock_rank_claimed = rank;
}

/** @brief Make vnn_CreateNeuralNetwork of the model fail. */
void
mock_ovxlib_set_create_fail (int fail)
{
  mock_create_fail = fail;
}

/** @brief Make vsi_nn_GetTensor fail for the tensor of the given identifier. */
void
mock_ovxlib_set_get_tensor_fail (int id)
{
  mock_get_tensor_fail = id;
}

/** @brief Make vsi_nn_RunGraph of the model fail. */
void
mock_ovxlib_set_run_fail (int fail)
{
  mock_run_fail = fail;
}

/** @brief Make the post-process of the model fail. */
void
mock_ovxlib_set_post_process_fail (int fail)
{
  mock_post_process_fail = fail;
}

/** @brief Get the number of the graphs created and not released. */
int
mock_ovxlib_get_live_graph (void)
{
  return mock_live_graph;
}

/** @brief Get the number of the times the mock model has been unloaded. */
int
mock_ovxlib_get_model_unload (void)
{
  return mock_model_unload;
}

/** @brief Get the number of the times the post-process of the model has run. */
int
mock_ovxlib_get_post_process (void)
{
  return mock_post_process;
}

/** @brief Record that the mock model has been unloaded. */
void
mock_ovxlib_note_model_unload (void)
{
  mock_model_unload++;
}

/** @brief Get the size of an element of the given data type in bytes. */
static gsize
mock_type_size (vsi_nn_type_e type)
{
  switch (type) {
    case VSI_NN_TYPE_INT8:
    case VSI_NN_TYPE_UINT8:
      return 1;
    case VSI_NN_TYPE_INT16:
    case VSI_NN_TYPE_UINT16:
    case VSI_NN_TYPE_FLOAT16:
      return 2;
    case VSI_NN_TYPE_INT32:
    case VSI_NN_TYPE_UINT32:
    case VSI_NN_TYPE_FLOAT32:
      return 4;
    default:
      break;
  }
  return 8;
}

/** @brief Give the tensor of the given index its attributes and contents. */
static void
mock_tensor_init (mock_tensor_s * tensor, guint index)
{
  guint i;

  tensor->pub.attr.dim_num = mock_rank_claimed ? mock_rank_claimed : mock_rank;
  tensor->pub.attr.size[0] = 2U + index;
  for (i = 1; i < mock_rank; i++)
    tensor->pub.attr.size[i] =
        (i < MOCK_TENSOR_DIM_NUM) ? MOCK_TENSOR_DIM_NUM - i : 1U;
  tensor->pub.attr.dtype.vx_type = mock_type;

  tensor->bytes = mock_type_size (mock_type);
  for (i = 0; i < mock_rank; i++)
    tensor->bytes *= tensor->pub.attr.size[i];

  tensor->data = g_malloc0 (tensor->bytes);
}

/** @brief Create a graph. To be called by the mock model only. */
vsi_nn_graph_t *
mock_ovxlib_create_graph (const char *model_path)
{
  mock_graph_s *graph;
  guint i;

  if (mock_create_fail || model_path == NULL)
    return NULL;

  graph = g_new0 (mock_graph_s, 1);
  graph->num_tensors = mock_num_input + mock_num_output;
  graph->tensors = g_new0 (mock_tensor_s, graph->num_tensors);
  graph->input_ids = g_new0 (vsi_nn_tensor_id_t, mock_num_input);
  graph->output_ids = g_new0 (vsi_nn_tensor_id_t, mock_num_output);

  for (i = 0; i < graph->num_tensors; i++) {
    guint index = (i < mock_num_input) ? i : i - mock_num_input;

    mock_tensor_init (&graph->tensors[i], index);
  }

  for (i = 0; i < mock_num_input; i++)
    graph->input_ids[i] = i;
  for (i = 0; i < mock_num_output; i++)
    graph->output_ids[i] = mock_num_input + i;

  graph->pub.input.tensors = graph->input_ids;
  graph->pub.input.num = mock_num_input;
  graph->pub.output.tensors = graph->output_ids;
  graph->pub.output.num = mock_num_output;

  mock_live_graph++;
  return &graph->pub;
}

/** @brief Release a graph. To be called by the mock model only. */
void
mock_ovxlib_release_graph (vsi_nn_graph_t * graph)
{
  mock_graph_s *self = (mock_graph_s *) graph;
  guint i;

  if (self == NULL)
    return;

  for (i = 0; i < self->num_tensors; i++)
    g_free (self->tensors[i].data);

  g_free (self->tensors);
  g_free (self->input_ids);
  g_free (self->output_ids);
  g_free (self);
  mock_live_graph--;
}

/** @brief Get the tensor of the given identifier from the given graph. */
vsi_nn_tensor_t *
vsi_nn_GetTensor (const vsi_nn_graph_t * graph, vsi_nn_tensor_id_t id)
{
  mock_graph_s *self = (mock_graph_s *) graph;

  if (self == NULL || id >= self->num_tensors)
    return NULL;
  if (mock_get_tensor_fail >= 0
      && (vsi_nn_tensor_id_t) mock_get_tensor_fail == id)
    return NULL;

  return &self->tensors[id].pub;
}

/** @brief Copy the contents of the given tensor into the given buffer. */
vsi_status
vsi_nn_CopyTensorToBuffer (const vsi_nn_graph_t * graph,
    vsi_nn_tensor_t * tensor, void *buffer)
{
  mock_tensor_s *self = (mock_tensor_s *) tensor;

  if (graph == NULL || self == NULL || buffer == NULL)
    return VSI_FAILURE;

  memcpy (buffer, self->data, self->bytes);
  return VSI_SUCCESS;
}

/** @brief Fill an input tensor. To be called by the mock model only. */
vsi_status
mock_ovxlib_copy_data_to_tensor (vsi_nn_graph_t * graph,
    vsi_nn_tensor_t * tensor, void *data)
{
  mock_tensor_s *self = (mock_tensor_s *) tensor;

  if (graph == NULL || self == NULL || data == NULL)
    return VSI_FAILURE;

  memcpy (self->data, data, self->bytes);
  return VSI_SUCCESS;
}

/** @brief Run a graph. To be called by the mock model only. */
vsi_status
mock_ovxlib_run_graph (vsi_nn_graph_t * graph)
{
  mock_graph_s *self = (mock_graph_s *) graph;
  guint i, j;

  if (self == NULL || mock_run_fail)
    return VSI_FAILURE;

  for (i = 0; i < self->pub.output.num; i++) {
    mock_tensor_s *out = &self->tensors[self->pub.output.tensors[i]];
    mock_tensor_s *in = &self->tensors[i % self->pub.input.num];

    for (j = 0; j < out->bytes; j++)
      out->data[j] = (guint8) (in->data[j % in->bytes] + 1U);
  }

  return VSI_SUCCESS;
}

/** @brief Run the post-process of a graph. To be called by the model only. */
vsi_status
mock_ovxlib_post_process (vsi_nn_graph_t * graph)
{
  if (graph == NULL)
    return VSI_FAILURE;

  mock_post_process++;
  if (mock_post_process_fail)
    return VSI_FAILURE;

  return VSI_SUCCESS;
}

/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file    qnn_mock_backend.cc
 * @date    22 Nov 2026
 * @brief   A mock QNN backend library for the qnn sub-plugin tests
 * @see     http://github.com/nnstreamer/nnstreamer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 *
 * The backend runs the graph described in qnn_mock.h on the CPU and checks
 * that every handle and buffer it gets is one it handed out and fits the
 * graph, so that a misuse by the sub-plugin fails the call instead of
 * passing silently.
 */
#include <map>
#include <set>
#include <string>

#include <string.h>

#include "qnn_mock.h"

namespace
{
/** @brief A context and the graphs created in it. */
struct mock_context {
  Qnn_BackendHandle_t backend;
  std::set<void *> graphs;
};

/** @brief A graph and whether it is finalized. */
struct mock_graph {
  mock_context *context;
  bool finalized;
};

std::set<void *> backends;
std::set<void *> contexts;
std::set<void *> graphs;
std::set<std::string> failing;
std::map<std::string, int> calls;
uint32_t num_providers = 1;

/** @brief Count a call and tell whether it is to fail. */
bool
enter (const char *function)
{
  calls[function]++;
  return failing.count (function) > 0;
}

/** @brief Free a graph. */
void
free_graph (void *graph)
{
  graphs.erase (graph);
  delete static_cast<mock_graph *> (graph);
}

/** @brief The size in bytes of an element of a data type, 0 if unknown. */
uint32_t
element_size (Qnn_DataType_t type)
{
  switch (type) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
      return 1;
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return 2;
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      return 4;
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
    case QNN_DATATYPE_FLOAT_64:
      return 8;
    default:
      return 0;
  }
}

/** @brief Check that a tensor carries a raw buffer of exactly its size. */
bool
valid_buffer (const Qnn_Tensor_t *t, Qnn_DataType_t type, uint32_t elements)
{
  return t->v1.dataType == type && t->v1.memType == QNN_TENSORMEMTYPE_RAW
         && t->v1.clientBuf.data != nullptr
         && t->v1.clientBuf.dataSize == element_size (type) * elements;
}

/**
 * @brief Check that the quantization parameters and the dynamic-dimension
 *        flags of in0 are those the mock model describes (see qnn_mock.h).
 */
bool
valid_input_description (const Qnn_Tensor_t *t)
{
  const Qnn_QuantizeParams_t *q = &t->v1.quantizeParams;

  switch (q->quantizationEncoding) {
    case QNN_QUANTIZATION_ENCODING_UNDEFINED:
      break;
    case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
      if (q->encodingDefinition != QNN_DEFINITION_DEFINED
          || q->scaleOffsetEncoding.scale != QNN_MOCK_SCALE
          || q->scaleOffsetEncoding.offset != QNN_MOCK_OFFSET)
        return false;
      break;
    case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET:
      if (q->encodingDefinition != QNN_DEFINITION_DEFINED
          || q->axisScaleOffsetEncoding.axis != QNN_MOCK_AXIS)
        return false;
      if (q->axisScaleOffsetEncoding.numScaleOffsets == 0)
        return q->axisScaleOffsetEncoding.scaleOffset == nullptr;
      for (uint32_t i = 0; i < q->axisScaleOffsetEncoding.numScaleOffsets; i++) {
        const Qnn_ScaleOffset_t *so = &q->axisScaleOffsetEncoding.scaleOffset[i];
        if (so->scale != QNN_MOCK_AXIS_SCALE (i) || so->offset != QNN_MOCK_AXIS_OFFSET (i))
          return false;
      }
      break;
    default:
      return false;
  }

  if (t->version == QNN_TENSOR_VERSION_2 && t->v2.isDynamicDimensions) {
    for (uint32_t i = 0; i < t->v2.rank; i++) {
      if (t->v2.isDynamicDimensions[i] != (i == 0 ? 1 : 0))
        return false;
    }
  }
  return true;
}

/** @brief Implements errorGetMessage. */
Qnn_ErrorHandle_t
mock_error_get_message (Qnn_ErrorHandle_t error, const char **message)
{
  enter ("errorGetMessage");
  if (!message)
    return QNN_MOCK_ERROR_INVALID_HANDLE;

  switch (error) {
    case QNN_MOCK_ERROR_INJECTED:
      *message = "injected failure";
      break;
    case QNN_MOCK_ERROR_INVALID_HANDLE:
      *message = "invalid handle";
      break;
    case QNN_MOCK_ERROR_INVALID_TENSOR:
      *message = "invalid tensor";
      break;
    case QNN_MOCK_ERROR_NOT_FINALIZED:
      *message = "graph is not finalized";
      break;
    default:
      return QNN_MOCK_ERROR_INVALID_HANDLE;
  }
  return QNN_SUCCESS;
}

/** @brief Implements backendCreate. */
Qnn_ErrorHandle_t
mock_backend_create (Qnn_LogHandle_t logger, const QnnBackend_Config_t **config,
    Qnn_BackendHandle_t *backend)
{
  (void) logger;
  (void) config;

  if (enter ("backendCreate"))
    return QNN_MOCK_ERROR_INJECTED;
  if (!backend)
    return QNN_MOCK_ERROR_INVALID_HANDLE;

  *backend = new int (0);
  backends.insert (*backend);
  return QNN_SUCCESS;
}

/** @brief Implements backendFree. */
Qnn_ErrorHandle_t
mock_backend_free (Qnn_BackendHandle_t backend)
{
  enter ("backendFree");
  if (backends.erase (backend) == 0)
    return QNN_MOCK_ERROR_INVALID_HANDLE;

  delete static_cast<int *> (backend);
  return QNN_SUCCESS;
}

/** @brief Implements contextCreate. */
Qnn_ErrorHandle_t
mock_context_create (Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device,
    const QnnContext_Config_t **config, Qnn_ContextHandle_t *context)
{
  (void) device;
  (void) config;

  if (enter ("contextCreate"))
    return QNN_MOCK_ERROR_INJECTED;
  if (backends.count (backend) == 0 || !context)
    return QNN_MOCK_ERROR_INVALID_HANDLE;

  mock_context *c = new mock_context ();
  c->backend = backend;
  *context = c;
  contexts.insert (c);
  return QNN_SUCCESS;
}

/** @brief Implements contextFree; it frees the graphs of the context too. */
Qnn_ErrorHandle_t
mock_context_free (Qnn_ContextHandle_t context, Qnn_ProfileHandle_t profile)
{
  (void) profile;

  enter ("contextFree");
  if (contexts.erase (context) == 0)
    return QNN_MOCK_ERROR_INVALID_HANDLE;

  mock_context *c = static_cast<mock_context *> (context);
  for (void *g : c->graphs)
    free_graph (g);
  delete c;
  return QNN_SUCCESS;
}

/** @brief Implements graphCreate. */
Qnn_ErrorHandle_t
mock_graph_create (Qnn_ContextHandle_t context, const char *name,
    const QnnGraph_Config_t **config, Qnn_GraphHandle_t *graph)
{
  (void) config;

  if (enter ("graphCreate"))
    return QNN_MOCK_ERROR_INJECTED;
  if (contexts.count (context) == 0 || !name || !graph)
    return QNN_MOCK_ERROR_INVALID_HANDLE;

  mock_context *c = static_cast<mock_context *> (context);
  mock_graph *g = new mock_graph{ c, false };
  c->graphs.insert (g);
  graphs.insert (g);
  *graph = g;
  return QNN_SUCCESS;
}

/** @brief Implements graphFinalize. */
Qnn_ErrorHandle_t
mock_graph_finalize (Qnn_GraphHandle_t graph, Qnn_ProfileHandle_t profile, Qnn_SignalHandle_t signal)
{
  (void) profile;
  (void) signal;

  if (enter ("graphFinalize"))
    return QNN_MOCK_ERROR_INJECTED;
  if (graphs.count (graph) == 0)
    return QNN_MOCK_ERROR_INVALID_HANDLE;

  static_cast<mock_graph *> (graph)->finalized = true;
  return QNN_SUCCESS;
}

/** @brief Implements graphExecute for the graph of qnn_mock.h. */
Qnn_ErrorHandle_t
mock_graph_execute (Qnn_GraphHandle_t graph, const Qnn_Tensor_t *inputs,
    uint32_t num_inputs, Qnn_Tensor_t *outputs, uint32_t num_outputs,
    Qnn_ProfileHandle_t profile, Qnn_SignalHandle_t signal)
{
  const uint32_t elements = 1 * 4 * 4 * 3;

  (void) profile;
  (void) signal;

  if (enter ("graphExecute"))
    return QNN_MOCK_ERROR_INJECTED;
  if (graphs.count (graph) == 0)
    return QNN_MOCK_ERROR_INVALID_HANDLE;
  if (!static_cast<mock_graph *> (graph)->finalized)
    return QNN_MOCK_ERROR_NOT_FINALIZED;
  if (num_inputs != 1 || num_outputs != 2 || !inputs || !outputs
      || !valid_buffer (&inputs[0], QNN_DATATYPE_UINT_8, elements)
      || !valid_input_description (&inputs[0])
      || !valid_buffer (&outputs[0], QNN_DATATYPE_UINT_8, elements)
      || !valid_buffer (&outputs[1], QNN_DATATYPE_INT_32, 1))
    return QNN_MOCK_ERROR_INVALID_TENSOR;

  const uint8_t *in = static_cast<const uint8_t *> (inputs[0].v1.clientBuf.data);
  uint8_t *out = static_cast<uint8_t *> (outputs[0].v1.clientBuf.data);
  int32_t sum = 0;

  for (uint32_t i = 0; i < elements; i++) {
    out[i] = (uint8_t) (in[i] + 1);
    sum += in[i];
  }
  memcpy (outputs[1].v1.clientBuf.data, &sum, sizeof (sum));
  return QNN_SUCCESS;
}

QnnInterface_t provider;
const QnnInterface_t *provider_list[] = { &provider };

} /* namespace */

extern "C" {

/**
 * @brief The entry point the sub-plugin resolves in a backend library.
 */
Qnn_ErrorHandle_t
QnnInterface_getProviders (const QnnInterface_t ***list, uint32_t *num)
{
  if (enter ("getProviders"))
    return QNN_MOCK_ERROR_INJECTED;
  if (!list || !num)
    return QNN_MOCK_ERROR_INVALID_HANDLE;

  provider.backendId = 0x7a;
  provider.providerName = "qnn_mock_backend";
  provider.apiVersion = { { 2, 0, 0 }, { 2, 0, 0 } };
  provider.QNN_INTERFACE_VER_NAME.errorGetMessage = mock_error_get_message;
  provider.QNN_INTERFACE_VER_NAME.backendCreate = mock_backend_create;
  provider.QNN_INTERFACE_VER_NAME.backendFree = mock_backend_free;
  provider.QNN_INTERFACE_VER_NAME.contextCreate = mock_context_create;
  provider.QNN_INTERFACE_VER_NAME.contextFree = mock_context_free;
  provider.QNN_INTERFACE_VER_NAME.graphCreate = mock_graph_create;
  provider.QNN_INTERFACE_VER_NAME.graphFinalize = mock_graph_finalize;
  provider.QNN_INTERFACE_VER_NAME.graphExecute = mock_graph_execute;

  *list = provider_list;
  *num = num_providers;
  return QNN_SUCCESS;
}

/** @brief Clear the injected failures and the counters. */
void
qnn_mock_backend_reset (void)
{
  failing.clear ();
  calls.clear ();
  num_providers = 1;
}

/** @brief Make the named function fail. */
void
qnn_mock_backend_fail (const char *function)
{
  failing.insert (function);
}

/** @brief Set the number of providers. */
void
qnn_mock_backend_set_num_providers (uint32_t num)
{
  num_providers = num;
}

/** @brief The number of calls of the named function. */
int
qnn_mock_backend_calls (const char *function)
{
  auto it = calls.find (function);
  return (it == calls.end ()) ? 0 : it->second;
}

/** @brief The number of objects not freed yet. */
int
qnn_mock_backend_live_objects (void)
{
  return (int) (backends.size () + contexts.size () + graphs.size ());
}

} /* extern "C" */

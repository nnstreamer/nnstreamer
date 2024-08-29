/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer tensor_filter, sub-plugin for QNN
 * Copyright (C) 2024 Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 */
/**
 * @file      tensor_filter_qnn.cc
 * @date      27 Aug 2024
 * @brief     NNStreamer tensor-filter sub-plugin for QNN (Qualcomm® AI Engine Direct)
 * @see       http://github.com/nnstreamer/nnstreamer
              https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
 * @author    Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug       No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (QNN) for tensor_filter.
 */

#include <string>
#include <unordered_map>
#include <vector>

#include <glib.h>
#include <gmodule.h>

#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

#include <QnnInterface.h>


/**
 * @brief Get the corresponding nns tensor_type for qnn data type.
 */
static tensor_type
qnnTensorType (Qnn_DataType_t data_type)
{
  const static std::unordered_map<Qnn_DataType_t, tensor_type> data_type_to_nns_type = {
    { QNN_DATATYPE_INT_8, _NNS_INT8 },
    { QNN_DATATYPE_INT_16, _NNS_INT16 },
    { QNN_DATATYPE_INT_32, _NNS_INT32 },
    { QNN_DATATYPE_INT_64, _NNS_INT64 },
    { QNN_DATATYPE_UINT_8, _NNS_UINT8 },
    { QNN_DATATYPE_UINT_16, _NNS_UINT16 },
    { QNN_DATATYPE_UINT_32, _NNS_UINT32 },
    { QNN_DATATYPE_UINT_64, _NNS_UINT64 },
    { QNN_DATATYPE_FLOAT_16, _NNS_FLOAT16 },
    { QNN_DATATYPE_FLOAT_32, _NNS_FLOAT32 },
    { QNN_DATATYPE_FLOAT_64, _NNS_FLOAT64 },
    { QNN_DATATYPE_SFIXED_POINT_8, _NNS_INT8 },
    { QNN_DATATYPE_SFIXED_POINT_16, _NNS_INT16 },
    { QNN_DATATYPE_SFIXED_POINT_32, _NNS_INT32 },
    { QNN_DATATYPE_UFIXED_POINT_8, _NNS_UINT8 },
    { QNN_DATATYPE_UFIXED_POINT_16, _NNS_UINT16 },
    { QNN_DATATYPE_UFIXED_POINT_32, _NNS_UINT32 },
    { QNN_DATATYPE_BOOL_8, _NNS_UINT8 },
  };

  auto um = data_type_to_nns_type.find (data_type);
  if (um == data_type_to_nns_type.end ()) {
    nns_loge ("Unknown QNN data type");
    return _NNS_END;
  }

  return um->second;
}

/**
 * @brief Get the data size of Qnn_Tensor_t.
 */
static uint32_t
qnnTensorDataSize (const Qnn_Tensor_t *tensor)
{
  uint32_t data_size = gst_tensor_get_element_size (qnnTensorType (tensor->v1.dataType));
  for (uint32_t i = 0; i < tensor->v1.rank; i++) {
    data_size *= tensor->v1.dimensions[i];
  }

  return data_size;
}

/**
 * @brief Deep copy given Qnn_Tensor_t
 */
static bool
deepCopyQnnTensor (Qnn_Tensor_t *dst, const Qnn_Tensor_t *src)
{
  if (nullptr == dst || nullptr == src) {
    nns_loge ("Received nullptr");
    return false;
  }

  dst->version = src->version;

  const char *tensorName = src->v1.name;
  dst->v1.name = g_strdup (tensorName);
  dst->v1.id = src->v1.id;
  dst->v1.type = src->v1.type;
  dst->v1.dataFormat = src->v1.dataFormat;
  dst->v1.dataType = src->v1.dataType;

  const Qnn_QuantizeParams_t &srcQParams = src->v1.quantizeParams;
  Qnn_QuantizeParams_t &qParams = dst->v1.quantizeParams;
  qParams = QNN_QUANTIZE_PARAMS_INIT;
  qParams.encodingDefinition = srcQParams.encodingDefinition;
  qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  if (srcQParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
    qParams.quantizationEncoding = srcQParams.quantizationEncoding;
    qParams.scaleOffsetEncoding = srcQParams.scaleOffsetEncoding;
  } else if (srcQParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
    qParams.quantizationEncoding = srcQParams.quantizationEncoding;
    qParams.axisScaleOffsetEncoding.axis = srcQParams.axisScaleOffsetEncoding.axis;
    qParams.axisScaleOffsetEncoding.numScaleOffsets
        = srcQParams.axisScaleOffsetEncoding.numScaleOffsets;
    if (srcQParams.axisScaleOffsetEncoding.numScaleOffsets > 0) {
      qParams.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t *) g_malloc0 (
          srcQParams.axisScaleOffsetEncoding.numScaleOffsets * sizeof (Qnn_ScaleOffset_t));
      if (qParams.axisScaleOffsetEncoding.scaleOffset) {
        for (size_t idx = 0;
             idx < srcQParams.axisScaleOffsetEncoding.numScaleOffsets; idx++) {
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale
              = srcQParams.axisScaleOffsetEncoding.scaleOffset[idx].scale;
          qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset
              = srcQParams.axisScaleOffsetEncoding.scaleOffset[idx].offset;
        }
      }
    }
  }

  uint32_t rank = src->v1.rank;
  dst->v1.rank = rank;
  dst->v1.dimensions = nullptr;
  if (rank > 0) {
    dst->v1.dimensions = (uint32_t *) malloc (rank * sizeof (uint32_t));
    if (dst->v1.dimensions) {
      memcpy (dst->v1.dimensions, src->v1.dimensions, rank * sizeof (uint32_t));
    }
  }

  /* only for v2, do not use QNN_TENSOR_MEMBER macro */
  if (src->version == QNN_TENSOR_VERSION_2) {
    if (rank > 0) {
      if (src->v2.isDynamicDimensions) {
        dst->v2.isDynamicDimensions = (uint8_t *) malloc (rank * sizeof (uint8_t));
        memcpy (dst->v2.isDynamicDimensions, src->v2.isDynamicDimensions,
            rank * sizeof (uint8_t));
      }
    }
    dst->v2.sparseParams = src->v2.sparseParams;
  }

  return true;
}


namespace nnstreamer
{
namespace tensor_filter_qnn
{
extern "C" {
void init_filter_qnn (void) __attribute__ ((constructor));
void fini_filter_qnn (void) __attribute__ ((destructor));
}

/** @brief Wrapper class for QNN APIs */
class QnnManager
{
  public:
  bool init (const char *model_lib_path, const char *backend_lib_path);
  void fini ();
  const char *qnnErrorHandleToString (Qnn_ErrorHandle_t err);
  Qnn_ErrorHandle_t executeGraph ();

  std::vector<Qnn_Tensor_t> input_tensors_;
  std::vector<Qnn_Tensor_t> output_tensors_;

  private:
  GModule *backend_module = nullptr;
  GModule *model_module = nullptr;
  QNN_INTERFACE_VER_TYPE qnn_interface_ = QNN_INTERFACE_VER_TYPE_INIT;
  Qnn_BackendHandle_t backend_handle_ = nullptr;
  Qnn_ContextHandle_t context_handle_ = nullptr;
  typedef enum ModelError {
    MODEL_NO_ERROR = 0,
    MODEL_TENSOR_ERROR = 1,
    MODEL_PARAMS_ERROR = 2,
    MODEL_NODES_ERROR = 3,
    MODEL_GRAPH_ERROR = 4,
    MODEL_CONTEXT_ERROR = 5,
    MODEL_GENERATION_ERROR = 6,
    MODEL_SETUP_ERROR = 7,
    MODEL_INVALID_ARGUMENT_ERROR = 8,
    MODEL_FILE_ERROR = 9,
    MODEL_MEMORY_ALLOCATE_ERROR = 10,
    MODEL_UNKNOWN_ERROR = 0x7FFFFFFF
  } ModelError_t;
  typedef struct GraphConfigInfo {
    char *graphName;
    const QnnGraph_Config_t **graphConfigs;
  } GraphConfigInfo_t;
  typedef struct GraphInfo {
    Qnn_GraphHandle_t graph;
    char *graphName;
    Qnn_Tensor_t *inputTensors;
    uint32_t numInputTensors;
    Qnn_Tensor_t *outputTensors;
    uint32_t numOutputTensors;
  } GraphInfo_t;
  GraphInfo_t **graphs_info_ = nullptr;

  using QnnInterfaceGetProvidersFn = decltype (QnnInterface_getProviders);
  using ComposeGraphsFn = ModelError_t (Qnn_BackendHandle_t, QNN_INTERFACE_VER_TYPE,
      Qnn_ContextHandle_t, const GraphConfigInfo_t **, const uint32_t,
      GraphInfo_t ***, uint32_t *, bool, QnnLog_Callback_t, QnnLog_Level_t);
  using FreeGraphsInfoFn = ModelError_t (GraphInfo_t ***, uint32_t);
};

/**
 * @brief Return error handle string.
 */
const char *
QnnManager::qnnErrorHandleToString (Qnn_ErrorHandle_t err)
{
  const char *error_msg = nullptr;
  if (qnn_interface_.errorGetMessage (err, &error_msg) == QNN_SUCCESS) {
    return error_msg;
  }

  return "Unknown";
}

/**
 * @brief Execute the graph.
 */
Qnn_ErrorHandle_t
QnnManager::executeGraph ()
{
  /* Execute graph */
  Qnn_ErrorHandle_t ret = qnn_interface_.graphExecute (graphs_info_[0]->graph,
      input_tensors_.data (), input_tensors_.size (), output_tensors_.data (),
      output_tensors_.size (), nullptr, nullptr);

  return ret;
}

/**
 * @brief Init QnnManager with given model and backend.
 */
bool
QnnManager::init (const char *model_path, const char *backend_path)
{
  /* Load backend shared library */
  backend_module = g_module_open (backend_path, G_MODULE_BIND_LOCAL);
  if (!backend_module) {
    nns_loge ("Failed to open backend library %s", backend_path);
    return false;
  }

  /* Resolve required symbols from backend library */
  QnnInterfaceGetProvidersFn *get_providers;
  if (!g_module_symbol (backend_module, "QnnInterface_getProviders", (void **) &get_providers)) {
    nns_loge ("Failed to load symbol from backend library: %s", g_module_error ());
    return false;
  }

  /* Get QnnInterface_t */
  uint32_t num_providers;
  QnnInterface_t **provider_list{ nullptr };
  Qnn_ErrorHandle_t ret
      = get_providers ((const QnnInterface_t ***) &provider_list, &num_providers);
  if (ret != QNN_SUCCESS) {
    nns_loge ("Qnn Interface failed to get providers: %s", qnnErrorHandleToString (ret));
    return false;
  }

  if (num_providers == 0U) {
    nns_loge ("No Qnn providers found.");
    return false;
  }

  /** @todo handle multiple providers, use the first one for now. */
  nns_logi ("use provider %s", provider_list[0]->providerName);

  /* Set qnn_interface_ */
  qnn_interface_ = provider_list[0]->QNN_INTERFACE_VER_NAME;

  /* Initialize backend */
  ret = qnn_interface_.backendCreate (nullptr, nullptr, &backend_handle_);
  if (ret != QNN_SUCCESS) {
    nns_loge ("Failed to create backend: %s", qnnErrorHandleToString (ret));
    return false;
  }

  /* Create Context */
  QnnContext_Config_t qnn_context_config = QNN_CONTEXT_CONFIG_INIT;
  qnn_context_config.option = QNN_CONTEXT_CONFIG_OPTION_PRIORITY;
  qnn_context_config.priority = QNN_PRIORITY_NORMAL;
  const QnnContext_Config_t *context_configs[] = { &qnn_context_config, nullptr };

  ret = qnn_interface_.contextCreate (
      backend_handle_, nullptr, context_configs, &context_handle_);
  if (ret != QNN_SUCCESS) {
    nns_loge ("Failed to create context: %s", qnnErrorHandleToString (ret));
    return false;
  }

  /* Prepare graphs from model.so */
  model_module = g_module_open (model_path, G_MODULE_BIND_LOCAL);
  if (!model_module) {
    g_warning ("Failed to open module: %s", g_module_error ());
    return false;
  }

  /* Compose graph */
  ComposeGraphsFn *compose_graphs = nullptr;
  if (!g_module_symbol (model_module, "QnnModel_composeGraphs", (void **) &compose_graphs)) {
    nns_loge ("Failed to get symbol 'QnnModel_composeGraphs' from the model: %s",
        g_module_error ());
  }

  uint32_t graph_count = 0U;
  ModelError_t mrt = compose_graphs (backend_handle_, qnn_interface_, context_handle_,
      nullptr, 1, &graphs_info_, &graph_count, false, nullptr, QNN_LOG_LEVEL_ERROR);
  if (mrt != MODEL_NO_ERROR) {
    nns_loge ("Failed to compose graphs. Error: %d", (int) mrt);
    return false;
  }

  /** @todo Support multiple graphs. Assume graph_count == 1 for now. */
  if (graph_count != 1U) {
    nns_loge ("Only support single graph for now. graph_count: %u", graph_count);
    return false;
  }

  GraphInfo_t *graph_info = graphs_info_[0];

  /* Parse input tensors */
  uint32_t num_input_tensors = graph_info->numInputTensors;
  input_tensors_.reserve (num_input_tensors);
  for (uint32_t j = 0; j < num_input_tensors; ++j) {
    Qnn_Tensor_t _t = QNN_TENSOR_INIT;
    if (!deepCopyQnnTensor (&_t, &graph_info->inputTensors[j])) {
      nns_loge ("Failed to copy tensor info.");
      return false;
    }
    _t.v1.clientBuf.dataSize = qnnTensorDataSize (&_t);
    input_tensors_.push_back (_t);
  }

  /* Parse output tensors */
  uint32_t num_output_tensors = graph_info->numOutputTensors;
  output_tensors_.reserve (num_output_tensors);
  for (uint32_t j = 0; j < num_output_tensors; ++j) {
    Qnn_Tensor_t _t = QNN_TENSOR_INIT;
    if (!deepCopyQnnTensor (&_t, &graph_info->outputTensors[j])) {
      nns_loge ("Failed to copy tensor info.");
      return false;
    }
    _t.v1.clientBuf.dataSize = qnnTensorDataSize (&_t);
    output_tensors_.push_back (_t);
  }

  /* Finalize Graphs */
  if (QNN_GRAPH_NO_ERROR
      != qnn_interface_.graphFinalize (graph_info->graph, nullptr, nullptr)) {
    nns_loge ("Failed to finalize QNN graph");
    return false;
  }

  return true;
}

/**
 * @brief Free QnnManager resources.
 */
void
QnnManager::fini ()
{
  auto _free_qnn_tensor = [] (Qnn_Tensor_t &tensor) {
    free ((void *) tensor.v1.name);
    free (tensor.v1.dimensions);
    if (tensor.version == QNN_TENSOR_VERSION_2) {
      if (tensor.v2.isDynamicDimensions) {
        free (tensor.v2.isDynamicDimensions);
      }
    }
  };

  /* release qnn tensors */
  for (auto &t : input_tensors_) {
    _free_qnn_tensor (t);
  }
  for (auto &t : output_tensors_) {
    _free_qnn_tensor (t);
  }

  input_tensors_.clear ();
  output_tensors_.clear ();

  /* free graphs from model */
  if (model_module) {
    FreeGraphsInfoFn *free_graphs;
    if (!g_module_symbol (model_module, "QnnModel_freeGraphsInfo", (void **) &free_graphs)) {
      nns_logw ("Failed to resolve symbol 'QnnModel_freeGraphsInfo' from model: %s",
          g_module_error ());
    } else {
      free_graphs (&graphs_info_, 1U);
    }

    g_module_close (model_module);
    model_module = nullptr;
  }

  /* release context */
  if (backend_module) {
    if (nullptr != qnn_interface_.contextFree) {
      qnn_interface_.contextFree (context_handle_, nullptr);
    }

    /* release backend */
    if (nullptr != qnn_interface_.backendFree) {
      qnn_interface_.backendFree (backend_handle_);
    }

    g_module_close (backend_module);
    backend_module = nullptr;
  }
}


/** @brief tensor-filter-subplugin concrete class for qnn */
class qnn_subplugin final : public tensor_filter_subplugin
{
  private:
  static qnn_subplugin *registeredRepresentation;
  static const GstTensorFilterFrameworkInfo framework_info;

  bool configured;
  void cleanup ();
  GstTensorsInfo inputInfo;
  GstTensorsInfo outputInfo;
  QnnManager qnn_manager;

  public:
  static void init_filter_qnn ();
  static void fini_filter_qnn ();

  qnn_subplugin ();
  ~qnn_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

/**
 * @brief Describe framework information.
 */
const GstTensorFilterFrameworkInfo qnn_subplugin::framework_info = { .name = "qnn",
  .allow_in_place = FALSE,
  .allocate_in_invoke = FALSE,
  .run_without_model = FALSE,
  .verify_model_path = TRUE,
  .hw_list = (const accl_hw[]){ ACCL_CPU },
  .num_hw = 1,
  .accl_auto = ACCL_CPU,
  .accl_default = ACCL_CPU,
  .statistics = nullptr };

/**
 * @brief Constructor for qnn_subplugin.
 */
qnn_subplugin::qnn_subplugin () : tensor_filter_subplugin (), configured (false)
{
}

/**
 * @brief Destructor for qnn subplugin.
 */
qnn_subplugin::~qnn_subplugin ()
{
  cleanup ();
}

/**
 * @brief Method to get empty object.
 */
tensor_filter_subplugin &
qnn_subplugin::getEmptyInstance ()
{
  return *(new qnn_subplugin ());
}

/**
 * @brief Method to cleanup qnn subplugin.
 */
void
qnn_subplugin::cleanup ()
{
  if (!configured)
    return;

  qnn_manager.fini ();

  gst_tensors_info_free (std::addressof (inputInfo));
  gst_tensors_info_free (std::addressof (outputInfo));

  configured = false;
}

/**
 * @brief Method to prepare/configure qnn instance.
 */
void
qnn_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  /* Already configured */
  if (configured) {
    cleanup ();
  }

  configured = true;

  gst_tensors_info_init (std::addressof (inputInfo));
  gst_tensors_info_init (std::addressof (outputInfo));

  if (prop->num_models != 2) {
    cleanup ();
    throw std::invalid_argument (
        "QNN subplugin requires two model files. Example) model=libQnnModel.so,libQnnBackend.so");
  }

  nns_logi ("model file: %s, backend lib file: %s", prop->model_files[0],
      prop->model_files[1]);

  if (!qnn_manager.init (prop->model_files[0], prop->model_files[1])) {
    cleanup ();
    throw std::invalid_argument ("Failed to prepare backend and model");
  }

  /* parse input tensors info */
  inputInfo.num_tensors = qnn_manager.input_tensors_.size ();
  for (size_t i = 0; i < inputInfo.num_tensors; ++i) {
    GstTensorInfo *info = gst_tensors_info_get_nth_info (std::addressof (inputInfo), i);
    Qnn_Tensor_t &tensor = qnn_manager.input_tensors_[i];

    info->name = g_strdup (tensor.v1.name);

    uint32_t rank = tensor.v1.rank;
    for (size_t j = 0; j < rank; ++j) {
      info->dimension[rank - 1 - j] = (tensor.v1.dimensions)[j];
    }

    info->type = qnnTensorType (tensor.v1.dataType);
    if (info->type == _NNS_END) {
      throw std::runtime_error ("Unsupported data type");
    }
  }

  /* parse output tensors info */
  outputInfo.num_tensors = qnn_manager.output_tensors_.size ();
  for (size_t i = 0; i < outputInfo.num_tensors; ++i) {
    GstTensorInfo *info = gst_tensors_info_get_nth_info (std::addressof (outputInfo), i);
    Qnn_Tensor_t &tensor = qnn_manager.output_tensors_[i];

    info->name = g_strdup (tensor.v1.name);

    uint32_t rank = tensor.v1.rank;
    for (size_t j = 0; j < rank; ++j) {
      info->dimension[rank - 1 - j] = (tensor.v1.dimensions)[j];
    }

    info->type = qnnTensorType (tensor.v1.dataType);
    if (info->type == _NNS_END) {
      throw std::runtime_error ("Unsupported data type");
    }
  }
}

/**
 * @brief Method to execute the model.
 */
void
qnn_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  if (!input)
    throw std::runtime_error ("Invalid input buffer, it is NULL.");
  if (!output)
    throw std::runtime_error ("Invalid output buffer, it is NULL.");

  for (size_t i = 0; i < inputInfo.num_tensors; i++) {
    qnn_manager.input_tensors_[i].v1.clientBuf.data = input[i].data;
  }

  for (size_t i = 0; i < outputInfo.num_tensors; i++) {
    qnn_manager.output_tensors_[i].v1.clientBuf.data = output[i].data;
  }

  Qnn_ErrorHandle_t ret = qnn_manager.executeGraph ();
  if (ret != QNN_SUCCESS) {
    throw std::runtime_error ("Failed to execute graph. "
                              + std::string (qnn_manager.qnnErrorHandleToString (ret)));
  }
}

/**
 * @brief Method to get the information of qnn subplugin.
 */
void
qnn_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info = framework_info;
}

/**
 * @brief Method to get the model information.
 */
int
qnn_subplugin::getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops == GET_IN_OUT_INFO) {
    gst_tensors_info_copy (std::addressof (in_info), std::addressof (inputInfo));
    gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputInfo));
    return 0;
  }

  return -ENOENT;
}

/**
 * @brief Method to handle events.
 */
int
qnn_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);

  return -ENOENT;
}

qnn_subplugin *qnn_subplugin::registeredRepresentation = nullptr;

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
qnn_subplugin::init_filter_qnn (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<qnn_subplugin> ();
}

/** @brief Destruct the subplugin */
void
qnn_subplugin::fini_filter_qnn (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/**
 * @brief Register the sub-plugin for qnn.
 */
void
init_filter_qnn ()
{
  qnn_subplugin::init_filter_qnn ();
}

/**
 * @brief Destruct the sub-plugin for qnn.
 */
void
fini_filter_qnn ()
{
  qnn_subplugin::fini_filter_qnn ();
}

} /* namespace tensor_filter_qnn */
} /* namespace nnstreamer */

/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer tensor_filter, sub-plugin for onnxruntime
 * Copyright (C) 2023 Suyeon Kim <suyeon5.kim@samsung.com>
 */
/**
 * @file        tensor_filter_onnxruntime.cc
 * @date        30 Oct 2023
 * @brief       NNStreamer tensor-filter sub-plugin for ONNXRuntime
 * @see         http://github.com/nnstreamer/nnstreamer
 * @see         https://onnxruntime.ai/
 * @author      Suyeon Kim <suyeon5.kim@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (onnxruntime) for tensor_filter.
 *
 * @todo Only float32 is allowed for input/output. Other types are NYI.
 * @todo Only CPU is supported. GPU and other hardware support is NYI.
 */

#include <iostream>
#include <set>
#include <string>

#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#include <nnstreamer_util.h>

#define ORT_API_MANUAL_INIT 1
#ifdef G_LOG_DOMAIN
#undef G_LOG_DOMAIN
#endif
#define G_LOG_DOMAIN "nnstreamer-onnxruntime"

#include <onnxruntime_cxx_api.h>

namespace nnstreamer
{
namespace tensor_filter_onnxruntime
{
extern "C" {
void init_filter_onnxruntime (void) __attribute__ ((constructor));
void fini_filter_onnxruntime (void) __attribute__ ((destructor));
}

static const gchar *onnx_accl_support[] = { ACCL_CPU_STR, ACCL_GPU_STR, NULL };

/** @brief tensor-filter-subplugin concrete class for onnxruntime */
class onnxruntime_subplugin final : public tensor_filter_subplugin
{
  private:
  /**
   * @brief Internal data structure for tensor information from ONNX.
   */
  typedef struct {
    std::size_t count;
    std::vector<Ort::AllocatedStringPtr> names_allocated_strings;
    std::vector<const char *> names;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<ONNXTensorElementDataType> types;
    std::vector<Ort::Value> tensors;
  } onnx_node_info_s;

  bool configured;
  ORTCHAR_T *model_path; /**< The model *.onnx file */

  Ort::Session session;
  Ort::SessionOptions sessionOptions;
  Ort::Env env;
  Ort::MemoryInfo memInfo;

  bool has_cuda;
  bool has_qnn;
  bool has_rocm;
  bool has_openvino;
  accl_hw has_accelerator;

  accl_hw use_accelerator;
  bool use_gpu;

  onnx_node_info_s inputNode;
  onnx_node_info_s outputNode;

  static const char *name;
  static onnxruntime_subplugin *registeredRepresentation;

  void cleanup ();
  void clearNodeInfo (onnx_node_info_s &node);
  void convertTensorInfo (onnx_node_info_s &node, GstTensorsInfo &info);
  int convertTensorDim (std::vector<int64_t> &shapes, tensor_dim &dim, bool &is_dynamic);
  int convertTensorType (ONNXTensorElementDataType _type, tensor_type &type);
  void setAccelerator (const char *accelerators);

  public:
  static void init_filter_onnxruntime ();
  static void fini_filter_onnxruntime ();

  onnxruntime_subplugin ();
  ~onnxruntime_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void invoke_dynamic (GstTensorFilterProperties *prop,
      const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
  int convertElementDataType (tensor_type type, ONNXTensorElementDataType &_type);
};

/**
 * @brief Constructor for onnxruntime_subplugin.
 */
onnxruntime_subplugin::onnxruntime_subplugin ()
    : configured{ false }, model_path{ nullptr }, session{ nullptr },
      sessionOptions{ nullptr }, env{ nullptr }, memInfo{ nullptr },
      has_cuda{ false }, has_qnn{ false },
      has_rocm{ false }, has_openvino{ false },
      has_accelerator{ ACCL_NONE },
      use_accelerator{ ACCL_NONE },
      use_gpu{ false }
{
  std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
  for (auto provider_name : availableProviders) {
    if (provider_name == "CUDAExecutionProvider") {
      has_cuda = true;
    } else if (provider_name == "ROCMExecutionProvider") {
      has_rocm = true;
    } else if (provider_name == "QNNExecutionProvider") {
      has_qnn = true;
    } else if (provider_name == "OpenVINOExecutionProvider") {
      has_openvino = true;
    }
  }
  nns_logi("onnxruntime provider: cuda=%d rocm=%d, qnn=%d, openvino=%d",
      has_cuda, has_rocm, has_qnn, has_openvino);
}

/**
 * @brief Destructor for onnxruntime_subplugin.
 */
onnxruntime_subplugin::~onnxruntime_subplugin ()
{
  cleanup ();
}

/** @brief cleanup resources used by onnxruntime subplugin */
void
onnxruntime_subplugin::cleanup ()
{
  if (!configured)
    return; /* Nothing to do if it is an empty model */

  session = Ort::Session{ nullptr };
  sessionOptions = Ort::SessionOptions{ nullptr };
  env = Ort::Env{ nullptr };
  memInfo = Ort::MemoryInfo{ nullptr };

  clearNodeInfo (inputNode);
  clearNodeInfo (outputNode);

  g_free (model_path);
  model_path = nullptr;
  configured = false;
}

/**
 * @brief Cleanup ONNX tensor information.
 */
void
onnxruntime_subplugin::clearNodeInfo (onnx_node_info_s &node)
{
  node.count = 0;
  node.names_allocated_strings.clear ();
  node.names.clear ();
  node.shapes.clear ();
  node.types.clear ();
  node.tensors.clear ();
}

/**
 * @brief Convert ONNX tensor information.
 */
void
onnxruntime_subplugin::convertTensorInfo (onnx_node_info_s &node, GstTensorsInfo &info)
{
  GstTensorInfo *_info;
  gst_tensors_info_init (std::addressof (info));
  info.num_tensors = (unsigned int) node.count;
  bool is_dynamic = false;
  for (guint i = 0; i < info.num_tensors; ++i) {
    _info = gst_tensors_info_get_nth_info (std::addressof (info), i);

    if (convertTensorType (node.types[i], _info->type) != 0)
      throw std::runtime_error ("Failed to convert ONNX data type.");

    if (convertTensorDim (node.shapes[i], _info->dimension, is_dynamic) != 0)
      throw std::runtime_error ("Failed to convert ONNX shape.");

    _info->name = g_strdup (node.names[i]);
  }
  if (is_dynamic) {
    info.format = _NNS_TENSOR_FORMAT_FLEXIBLE;
  }
}

/**
 * @brief Convert the shape of tensor.
 * @return 0 if OK. non-zero if error.
 */
int
onnxruntime_subplugin::convertTensorDim (std::vector<int64_t> &shapes, tensor_dim &dim, bool &is_dynamic)
{
  size_t i, rank;
  is_dynamic = false;
  rank = shapes.size ();
  if (rank <= 0 || rank > NNS_TENSOR_RANK_LIMIT) {
    nns_loge ("Invalid shape (rank %zu, max: %d)", rank, NNS_TENSOR_RANK_LIMIT);
    return -EINVAL;
  }

  /* the order of dimension is reversed at CAPS negotiation */
  for (i = 0; i < rank; i++) {
    /* free dimensions are treated as 1 if not overridden */
    if (shapes[rank - i - 1] < 0) {
      is_dynamic = true;
      dim[i] = 1;
    } else {
      dim[i] = shapes[rank - i - 1];
    }
  }

  /* fill remaining entries with 0 */
  for (i = rank; i < NNS_TENSOR_RANK_LIMIT; ++i) {
    dim[i] = 0;
  }

  return 0;
}

/**
 * @brief Convert the type of tensor.
 * @return 0 if OK. non-zero if error.
 */
int
onnxruntime_subplugin::convertTensorType (ONNXTensorElementDataType _type, tensor_type &type)
{
  switch (_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      type = _NNS_INT8;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      type = _NNS_UINT8;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      type = _NNS_INT16;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      type = _NNS_UINT16;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      type = _NNS_INT32;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      type = _NNS_UINT32;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      type = _NNS_INT64;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      type = _NNS_UINT64;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      type = _NNS_FLOAT32;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      type = _NNS_FLOAT64;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
#ifdef FLOAT16_SUPPORT
      type = _NNS_FLOAT16;
      break;
#endif
    default:
      nns_loge ("Tensor type not supported: %d", (gint) _type);
      type = _NNS_END;
      return -EINVAL;
  }

  return 0;
}

/**
 * @brief Convert the type of tensor.
 * @return 0 if OK. non-zero if error.
 */
int
onnxruntime_subplugin::convertElementDataType (tensor_type type, ONNXTensorElementDataType &_type)
{
  switch (type) {
    case _NNS_INT8:
      _type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
      break;
    case _NNS_UINT8:
      _type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
      break;
    case _NNS_INT16:
      _type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
      break;
    case _NNS_UINT16:
      _type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
      break;
    case _NNS_INT32:
      _type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
      break;
    case _NNS_UINT32:
      _type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
      break;
    case _NNS_INT64:
      _type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
      break;
    case _NNS_UINT64:
      _type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
      break;
    case _NNS_FLOAT32:
      _type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      break;
    case _NNS_FLOAT64:
      _type = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
      break;
    case _NNS_FLOAT16:
#ifdef FLOAT16_SUPPORT
      _type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
      break;
#endif
    default:
      nns_loge ("Element type not supported: %d", (gint) type);
      _type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      return -EINVAL;
  }

  return 0;
}

/**
 * @brief Method to get empty object.
 */
tensor_filter_subplugin &
onnxruntime_subplugin::getEmptyInstance ()
{
  return *(new onnxruntime_subplugin ());
}

/**
 * @brief Method to prepare/configure onnxruntime instance.
 */
void
onnxruntime_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  size_t i, num_inputs, num_outputs;

  if (configured) {
    /* Already opened */
    if (!prop->model_files[0] || prop->model_files[0][0] == '\0') {
      throw std::runtime_error ("Model path is not given.");
    }
    cleanup ();
  }
  nns_logi("num_hw: %d acc string: %s", prop->num_hw, prop->accl_str);
  for (int j = 0; j < prop->num_hw; j++) {
    nns_logi("prop->hw_list[i]: %d", prop->hw_list[j]);
  }
  setAccelerator(prop->accl_str);
  if (!g_file_test (prop->model_files[0], G_FILE_TEST_IS_REGULAR)) {
    const std::string err_msg
        = "Given file " + (std::string) prop->model_files[0] + " is not valid";
    cleanup ();
    throw std::runtime_error (err_msg);
  }

  // Handle Windows path conversion
#if (defined(_WIN32) || defined(__CYGWIN__))
  // TODO: add error checking and check type of model_files
  char *model_path_char = g_strdup (prop->model_files[0]);

  int wlen = mbstowcs(NULL, model_path_char, 0);
  model_path = (wchar_t*) malloc((wlen + 1) * sizeof(wchar_t));

  mbstowcs(model_path, model_path_char, wlen + 1);
  g_free(model_path_char);
#else
  model_path = g_strdup (prop->model_files[0]);
#endif

  /* Read a model */
  env = Ort::Env (ORT_LOGGING_LEVEL_WARNING, "nnstreamer_onnxruntime");
  session = Ort::Session (env, model_path, sessionOptions);

  num_inputs = session.GetInputCount ();
  if (num_inputs <= 0 || num_inputs > NNS_TENSOR_SIZE_LIMIT) {
    cleanup ();
    throw std::invalid_argument (
        std::string ("Too many input tensors: ") + std::to_string (num_inputs)
        + std::string ("max: ") + NNS_TENSOR_SIZE_LIMIT_STR);
  }

  num_outputs = session.GetOutputCount ();
  if (num_outputs <= 0 || num_outputs > NNS_TENSOR_SIZE_LIMIT) {
    cleanup ();
    throw std::invalid_argument (
        std::string ("Too many output tensors: ") + std::to_string (num_outputs)
        + std::string ("max: ") + NNS_TENSOR_SIZE_LIMIT_STR);
  }

  Ort::AllocatorWithDefaultOptions allocator;

  /* Initialize input info */
  inputNode.count = 0;

  for (i = 0; i < num_inputs; i++) {
    /* Get input name */
    auto input_name = session.GetInputNameAllocated (i, allocator);
    inputNode.names_allocated_strings.push_back (std::move (input_name));
    inputNode.names.push_back (inputNode.names_allocated_strings.back ().get ());

    /* Get input type and shape */
    Ort::TypeInfo type_info = session.GetInputTypeInfo (i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo ();
    inputNode.types.push_back (tensor_info.GetElementType ());
    inputNode.shapes.push_back (tensor_info.GetShape ());

    inputNode.count++;
  }

  /* Initialize output info */
  outputNode.count = 0;
  auto output_names = std::set<std::string>();
  for (i = 0; i < prop->output_meta.num_tensors; i++) {
    if (prop->output_meta.info[i].name) {
      output_names.insert (prop->output_meta.info[i].name);
    }
  }
  for (i = 0; i < num_outputs; i++) {
    /* Get output name */
    auto output_name = session.GetOutputNameAllocated (i, allocator);
    if (!output_names.empty() && output_names.count(output_name.get()) == 0) {
      g_info("skipping model output tensor %s, not pre-configured", output_name.get());
      continue;
    }
    if (outputNode.count == NNS_TENSOR_SIZE_LIMIT) {
      g_info("skipping model output tensor %s, max reached", output_name.get());
      continue;
    }
    outputNode.names_allocated_strings.push_back (std::move (output_name));
    outputNode.names.push_back (outputNode.names_allocated_strings.back ().get ());

    /* Get output type and shape */
    Ort::TypeInfo type_info = session.GetOutputTypeInfo (i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo ();
    outputNode.types.push_back (tensor_info.GetElementType ());
    outputNode.shapes.push_back (tensor_info.GetShape ());

    outputNode.count++;
  }

  memInfo = Ort::MemoryInfo::CreateCpu (
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  configured = true;
  allocator = Ort::AllocatorWithDefaultOptions{ nullptr }; /* delete unique_ptr */
}

/**
 * @brief	Set the accelerator for the onnx
 */
void
onnxruntime_subplugin::setAccelerator (const char *accelerators)
{
  use_gpu = TRUE;
  use_accelerator = parse_accl_hw (accelerators, onnx_accl_support, NULL, NULL);
  if ((use_accelerator & (ACCL_CPU)) != 0) {
    use_gpu = FALSE;
  } else {
    sessionOptions = Ort::SessionOptions();
    if (has_cuda) {
      auto api = Ort::GetApi();
      OrtCUDAProviderOptionsV2* options = nullptr;
      Ort::ThrowOnError(api.CreateCUDAProviderOptions(&options));
//      std::vector<const char*> keys{"enable_cuda_graph"};
//      std::vector<const char*> values{"1"};
//      Ort::ThrowOnError(api.UpdateCUDAProviderOptions(options, keys.data(), values.data(), 1));
      sessionOptions.AppendExecutionProvider_CUDA_V2(*options);
      api.ReleaseCUDAProviderOptions(options);
    } else if (has_qnn) {
#if (defined(_WIN32) || defined(__CYGWIN__))
      std::unordered_map<std::string, std::string> provider_options;
      provider_options["backend_path"] = "QnnHtp.dll";
      sessionOptions.AppendExecutionProvider("QNN", provider_options);
#else
      sessionOptions.AppendExecutionProvider("QNN");
#endif
    } else if (has_rocm) {
      auto api = Ort::GetApi();
      OrtROCMProviderOptions* options = nullptr;
      Ort::ThrowOnError(api.CreateROCMProviderOptions(&options));
      sessionOptions.AppendExecutionProvider_ROCM(*options);
      api.ReleaseROCMProviderOptions(options);
    } else if (has_openvino) {
      sessionOptions.AppendExecutionProvider("OpenVINO");
    }
  }
}

/**
 * @brief Method to execute the model with dynamic tensors.
 */
void
onnxruntime_subplugin::invoke_dynamic (GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  size_t i;
  g_assert (configured);

  inputNode.tensors.clear ();
  outputNode.tensors.clear ();

  if (!input)
    throw std::runtime_error ("Invalid input buffer, it is NULL.");
  if (!output)
    throw std::runtime_error ("Invalid output buffer, it is NULL.");

  if (prop == nullptr || prop->input_meta.format == _NNS_TENSOR_FORMAT_STATIC) {
    /* Set input to tensor */
    for (i = 0; i < inputNode.count; ++i) {
      auto shape = inputNode.shapes[i].data ();
      auto shape_size = inputNode.shapes[i].size ();
      inputNode.tensors.emplace_back (Ort::Value::CreateTensor (
          memInfo,
          input[i].data,
          input[i].size,
          shape,
          shape_size,
          inputNode.types[i]));
    }
  } else if (prop->input_meta.format == _NNS_TENSOR_FORMAT_FLEXIBLE) {
    inputNode.count = prop->input_meta.num_tensors;
    for (i = 0; i < prop->input_meta.num_tensors; i++) {
      auto element_data_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
      std::vector<int64_t> shape;
      /* revert order between onnxruntime <> nnstreamer dimensions */
      for (auto j = NNS_TENSOR_RANK_LIMIT-1; j >= 0 ; j--) {
        if (prop->input_meta.info[i].dimension[j] > 0) {
          shape.push_back(prop->input_meta.info[i].dimension[j]);
        } else {
          continue;
        }
      }
      convertElementDataType (
          static_cast<tensor_type> (prop->input_meta.info[i].type), element_data_type);
      inputNode.tensors.emplace_back (Ort::Value::CreateTensor (
          memInfo,
          input[i].data,
          input[i].size,
          shape.data(),
          shape.size(),
          element_data_type));
    }
  } else {
    const std::string err_msg
        = "ERROR running model inference: does not support tensor format " +
          (std::string) gst_tensor_get_format_string (prop->input_meta.format);
    throw std::runtime_error (err_msg);
  }

  if (prop == nullptr || (prop->output_meta.format == _NNS_TENSOR_FORMAT_STATIC && !prop->invoke_dynamic)) {
    /* Set output to tensor */
    for (i = 0; i < outputNode.count; ++i) {
      outputNode.tensors.emplace_back (Ort::Value::CreateTensor (memInfo,
          output[i].data, output[i].size, outputNode.shapes[i].data (),
          outputNode.shapes[i].size (), outputNode.types[i]));
    }
    try {
      /* call Run() to fill in the GstTensorMemory *output data with the probabilities of each */
      session.Run (Ort::RunOptions{ nullptr }, inputNode.names.data (),
          inputNode.tensors.data (), inputNode.count, outputNode.names.data (),
          outputNode.tensors.data (), outputNode.count);
    } catch (const Ort::Exception &exception) {
      const std::string err_msg
          = "ERROR running model inference: " + (std::string) exception.what ();
      throw std::runtime_error (err_msg);
    }
  } else if (prop->input_meta.format == _NNS_TENSOR_FORMAT_FLEXIBLE || prop->invoke_dynamic) {
    try {
      /* call Run() to fill in the GstTensorMemory *output data with the probabilities of each */
      auto outputTensors = session.Run (Ort::RunOptions{ nullptr }, inputNode.names.data (),
          inputNode.tensors.data (), inputNode.count, outputNode.names.data (),
          outputNode.count);

      gst_tensors_info_init (&prop->output_meta);
      prop->output_meta.num_tensors = outputTensors.size();
      prop->output_meta.format = _NNS_TENSOR_FORMAT_FLEXIBLE;

      for (i = 0; i < outputTensors.size(); i++) {
        size_t scalar_count = 1;
        auto outputInfo = outputTensors[i].GetTensorTypeAndShapeInfo();
        if (convertTensorType (outputInfo.GetElementType(), prop->output_meta.info[i].type) != 0) {
          throw std::runtime_error ("Failed to convert ONNX data type.");
        }
        /* revert order between onnxruntime <> nnstreamer dimensions */
        auto rank = outputInfo.GetShape().size();
        for (unsigned int shapeI = rank; shapeI > 0 ; shapeI--) {
          auto dim = outputInfo.GetShape()[shapeI - 1];
          prop->output_meta.info[i].dimension[rank - shapeI] = dim;
          scalar_count *= dim;
        }
        output[i].size = scalar_count * gst_tensor_get_element_size(prop->output_meta.info[i].type);
        output[i].data = g_memdup2(outputTensors[i].GetTensorRawData(), output[i].size);
      }
    } catch (const Ort::Exception &exception) {
      const std::string err_msg
          = "ERROR running model inference: " + (std::string) exception.what ();
      throw std::runtime_error (err_msg);
    }
  }
}


/**
 * @brief Method to execute the model.
 */
void
onnxruntime_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  invoke_dynamic(nullptr, input, output);
}

/**
 * @brief Method to get the information of onnxruntime subplugin.
 */
void
onnxruntime_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = 0;
  info.allocate_in_invoke = 0;
  info.run_without_model = 0;
  info.verify_model_path = 1;
  if (has_cuda || has_qnn || has_openvino || has_rocm) {
    has_accelerator = ACCL_GPU;
    info.num_hw = 1;
    info.hw_list = &has_accelerator;
    info.accl_auto = has_accelerator;
    info.accl_default = has_accelerator;
  }
}

/**
 * @brief Method to get the model information.
 */
int
onnxruntime_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops == GET_IN_OUT_INFO) {
    convertTensorInfo (inputNode, in_info);
    convertTensorInfo (outputNode, out_info);

    /* For debug, print input and output tensor information. */
    g_autofree gchar *instr = gst_tensors_info_to_string (std::addressof (in_info));
    g_autofree gchar *outstr = gst_tensors_info_to_string (std::addressof (out_info));
    nns_logd ("Input info: %s", instr);
    nns_logd ("Output info: %s", outstr);

    return 0;
  }

  return -ENOENT;
}

/**
 * @brief Method to handle events.
 */
int
onnxruntime_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);
  return -ENOENT;
}

const char *onnxruntime_subplugin::name = "onnxruntime";
onnxruntime_subplugin *onnxruntime_subplugin::registeredRepresentation = nullptr;

/** @brief Initialize this object for tensor_filter subplugin runtime register. */
void
onnxruntime_subplugin::init_filter_onnxruntime ()
{
  Ort::InitApi();
  std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
  for (auto f : availableProviders) {
    nns_logi("onnxruntime filter found provider: %s", f.c_str());
  }
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<onnxruntime_subplugin> ();
}

/** @brief Destruct the sub-plugin for onnxruntime. */
void
onnxruntime_subplugin::fini_filter_onnxruntime ()
{
  g_assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

/** @brief initializer */
void
init_filter_onnxruntime ()
{
  if (nnstreamer_filter_find ("onnx")) {
    nns_loge ("Cannot use onnxruntime and onnx both. Won't register this onnxruntime subplugin.");
    return;
  }

  onnxruntime_subplugin::init_filter_onnxruntime ();
}

/** @brief finalizer */
void
fini_filter_onnxruntime ()
{
  onnxruntime_subplugin::fini_filter_onnxruntime ();
}

} /* namespace tensor_filter_onnxruntime */
} /* namespace nnstreamer */

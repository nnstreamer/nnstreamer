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

#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include <tensor_common.h>

#include <onnxruntime_cxx_api.h>

namespace nnstreamer
{
namespace tensor_filter_onnxruntime
{
extern "C" {
void init_filter_onnxruntime (void) __attribute__ ((constructor));
void fini_filter_onnxruntime (void) __attribute__ ((destructor));
}

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
  char *model_path; /**< The model *.onnx file */

  Ort::Session session;
  Ort::SessionOptions sessionOptions;
  Ort::Env env;
  Ort::MemoryInfo memInfo;

  onnx_node_info_s inputNode;
  onnx_node_info_s outputNode;

  static const char *name;
  static onnxruntime_subplugin *registeredRepresentation;

  void cleanup ();
  void clearNodeInfo (onnx_node_info_s &node);
  void convertTensorInfo (onnx_node_info_s &node, GstTensorsInfo &info);
  int convertTensorDim (std::vector<int64_t> shapes, tensor_dim &dim);
  int convertTensorType (ONNXTensorElementDataType _type, tensor_type &type);

  public:
  static void init_filter_onnxruntime ();
  static void fini_filter_onnxruntime ();

  onnxruntime_subplugin ();
  ~onnxruntime_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

/**
 * @brief Constructor for onnxruntime_subplugin.
 */
onnxruntime_subplugin::onnxruntime_subplugin ()
    : configured{ false }, session{ nullptr }, sessionOptions{ nullptr }, env{ nullptr }, memInfo{ nullptr }
{
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

  for (guint i = 0; i < info.num_tensors; ++i) {
    _info = gst_tensors_info_get_nth_info (std::addressof (info), i);

    if (convertTensorType (node.types[i], _info->type) != 0)
      throw std::runtime_error ("Failed to convert ONNX data type.");

    if (convertTensorDim (node.shapes[i], _info->dimension) != 0)
      throw std::runtime_error ("Failed to convert ONNX shape.");

    _info->name = g_strdup (node.names[i]);
  }
}

/**
 * @brief Convert the shape of tensor.
 * @return 0 if OK. non-zero if error.
 */
int
onnxruntime_subplugin::convertTensorDim (std::vector<int64_t> shapes, tensor_dim &dim)
{
  size_t i, rank;

  rank = shapes.size ();
  if (rank <= 0 || rank > NNS_TENSOR_RANK_LIMIT) {
    nns_loge ("Invalid shape (rank %zu, max: %d)", rank, NNS_TENSOR_RANK_LIMIT);
    return -EINVAL;
  }

  /* the order of dimension is reversed at CAPS negotiation */
  for (i = 0; i < rank; i++) {
    /* free dimensions are treated as 1 if not overriden */
    dim[i] = (shapes[rank - i - 1] > 0) ? shapes[rank - i - 1] : 1;
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

  if (!g_file_test (prop->model_files[0], G_FILE_TEST_IS_REGULAR)) {
    const std::string err_msg
        = "Given file " + (std::string) prop->model_files[0] + " is not valid";
    cleanup ();
    throw std::runtime_error (err_msg);
  }

  model_path = g_strdup (prop->model_files[0]);

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
  inputNode.count = num_inputs;

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
  }

  /* Initialize output info */
  outputNode.count = num_outputs;

  for (i = 0; i < num_outputs; i++) {
    /* Get output name */
    auto output_name = session.GetOutputNameAllocated (i, allocator);
    outputNode.names_allocated_strings.push_back (std::move (output_name));
    outputNode.names.push_back (outputNode.names_allocated_strings.back ().get ());

    /* Get output type and shape */
    Ort::TypeInfo type_info = session.GetOutputTypeInfo (i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo ();
    outputNode.types.push_back (tensor_info.GetElementType ());
    outputNode.shapes.push_back (tensor_info.GetShape ());
  }

  memInfo = Ort::MemoryInfo::CreateCpu (
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  configured = true;
  allocator = Ort::AllocatorWithDefaultOptions{ nullptr }; /* delete unique_ptr */
}

/**
 * @brief Method to execute the model.
 */
void
onnxruntime_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  size_t i;
  g_assert (configured);

  inputNode.tensors.clear ();
  outputNode.tensors.clear ();

  if (!input)
    throw std::runtime_error ("Invalid input buffer, it is NULL.");
  if (!output)
    throw std::runtime_error ("Invalid output buffer, it is NULL.");

  /* Set input to tensor */
  for (i = 0; i < inputNode.count; ++i) {
    inputNode.tensors.emplace_back (Ort::Value::CreateTensor (memInfo,
        input[i].data, input[i].size, inputNode.shapes[i].data (),
        inputNode.shapes[i].size (), inputNode.types[i]));
  }

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

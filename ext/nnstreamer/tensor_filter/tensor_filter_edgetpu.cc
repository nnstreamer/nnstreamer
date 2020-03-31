// SPDX-License-Identifier: LGPL-2.1-only
/**
 * GStreamer Tensor_Filter, Edge-TPU Module
 * Copyright (C) 2019 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file	tensor_filter_edgetpu.c
 * @date	10 Dec 2019
 * @brief	Edge-TPU module for tensor_filter gstreamer plugin
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (Edge TPU) for tensor_filter.
 *
 * @todo This supports single-model-single-TPU only. Prep for multi-TPU.
 * @todo A lot of this duplicate tf-lite filter.
 *       We may be able to embed this code into tf-lite filter code.
 */
#include <iostream>
#include <string>
#include <stdexcept>

#include <stdint.h>

#include <nnstreamer_log.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <tensor_common.h>
#include <glib.h>

#include <edgetpu.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/builtin_op_data.h>
#include <tensorflow/lite/kernels/register.h>

using nnstreamer::tensor_filter_subplugin;
using edgetpu::EdgeTpuContext;

namespace nnstreamer {
namespace tensorfilter_edgetpu {

void _init_filter_edgetpu (void) __attribute__ ((constructor));
void _fini_filter_edgetpu (void) __attribute__ ((destructor));

class edgetpu_subplugin final : public tensor_filter_subplugin {
private:
  bool empty_model;
  char *model_path; /**< The model *.tflite file */
  GstTensorsInfo inputInfo; /**< Input tensors metadata */
  GstTensorsInfo outputInfo; /**< Output tensors metadata */

  /** Edge-TPU + TFLite Library Properties & Functions ******************/
  std::unique_ptr<tflite::Interpreter> model_interpreter;
      /**< TFLite interpreter */
  tflite::Interpreter *interpreter;
      /**< model_interpreter.get() */
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context;
      /**< EdgeTPU Device context */
  std::unique_ptr<tflite::FlatBufferModel> model;
      /**< Loaded TF Lite model (from model_path) */
  static std::unique_ptr<tflite::Interpreter>
      BuildEdgeTpuInterpreter(const tflite::FlatBufferModel &model,
          edgetpu::EdgeTpuContext* edgetpu_context);


  /** Internal Utility Functions & Properties ***************************/
  void cleanup ();
  static void setTensorProp (tflite::Interpreter *interpreter,
      const std::vector<int> &tensor_idx_list, GstTensorsInfo & tensorMeta);
  static int getTensorDim ( tflite::Interpreter *interpreter, int tensor_idx,
      tensor_dim dim);
  static tensor_type getTensorType (TfLiteType tfType);
  static const char *name;
  static const accl_hw hw_list[];
  static const int num_hw = 1;
  static edgetpu_subplugin *registeredRepresentation;

public:
  static void init_filter_edgetpu ();
  static void fini_filter_edgetpu ();

  edgetpu_subplugin ();
  ~edgetpu_subplugin ();

  tensor_filter_subplugin & getEmptyInstance();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

const char *edgetpu_subplugin::name = "edgetpu";
const accl_hw edgetpu_subplugin::hw_list[] = { ACCL_NPU_EDGE_TPU };

edgetpu_subplugin::edgetpu_subplugin () :
    tensor_filter_subplugin (),
    empty_model (true),
    model_path (nullptr),
    model_interpreter (nullptr),
    edgetpu_context (nullptr),
    model (nullptr)
{
  inputInfo.num_tensors = 0;
  outputInfo.num_tensors = 0;
  /** Nothing to do. Just let it have an empty instance */
}

void edgetpu_subplugin::cleanup ()
{
  if (empty_model)
    return; /* Nothing to do if it is an empty model */

  if (model_interpreter) {
    model_interpreter = nullptr; /* delete unique_ptr */
  }
  if (interpreter) {
    interpreter = nullptr; /* it's already freed with model_interpreter */
  }
  if (edgetpu_context) {
    edgetpu_context.reset();
    edgetpu_context = nullptr;
  }
  if (model) {
    model = nullptr; /* delete unique_ptr */
  }

  if (model_path)
    delete model_path;

  model_path = nullptr;
  inputInfo.num_tensors = 0;
  outputInfo.num_tensors = 0;
  empty_model = true;
}

edgetpu_subplugin::~edgetpu_subplugin ()
{
  cleanup ();
}

tensor_filter_subplugin & edgetpu_subplugin::getEmptyInstance ()
{
  return *(new edgetpu_subplugin());
}

void edgetpu_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  const std::string _model_path = prop->model_files[0];

  if (!empty_model) {
    /* Already opened */

    if (!prop->model_files[0] || prop->model_files[0][0] == '\0') {
      std::cerr << "Model path is not given." << std::endl;
      throw std::invalid_argument ("Model path is not given.");
    }

    cleanup();
  }

  assert (model_path == nullptr);

  model_path = g_strdup (prop->model_files[0]);

  /** Read a model */
  model = tflite::FlatBufferModel::BuildFromFile(_model_path.c_str());
  if (nullptr == model) {
    std::cerr << "Cannot load the model file: " << _model_path << std::endl;
    cleanup();
    throw std::invalid_argument ("Cannot load the given model file.");
  }

  /** Build an interpreter */
  edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  if (nullptr == edgetpu_context) {
    std::cerr << "Cannot open edge-TPU device." << std::endl;
    cleanup();
    throw std::system_error (ENODEV, std::system_category(), "Cannot open edge-TPU device.");
  }

  model_interpreter = BuildEdgeTpuInterpreter(*model, edgetpu_context.get());
  if (nullptr == model_interpreter) {
    std::cerr << "Edge-TPU device is opened, but cannot get its interpreter."
        << std::endl;
    cleanup();
    throw std::system_error (ENODEV, std::system_category(), "Edge-TPU device is opened, but cannot get its interpreter.");
  }

  interpreter = model_interpreter.get();

  try {
    setTensorProp (interpreter, interpreter->inputs (), inputInfo);
  } catch (const std::invalid_argument& ia) {
    std::cerr << "Invalid input tensor specification: " << ia.what() << '\n';
    cleanup();
    throw std::invalid_argument ("Input tensor of the given model is incompatible or invalid");
  }

  try {
    setTensorProp (interpreter, interpreter->outputs (), outputInfo);
  } catch (const std::invalid_argument& ia) {
    std::cerr << "Invalid output tensor specification: " << ia.what() << '\n';
    cleanup();
    throw std::invalid_argument ("Output tensor of the given model is incompatible or invalid");
  }

  empty_model = false;
}

void edgetpu_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  unsigned int i;

  std::vector <int> tensors_idx;
  int tensor_idx;
  TfLiteTensor *tensor_ptr;
  TfLiteStatus status;

  assert (!empty_model);
  assert (interpreter);

  /* Configure inputs */
  for (i = 0; i < inputInfo.num_tensors; i++) {
    tensor_idx = interpreter->inputs ()[i];
    tensor_ptr = interpreter->tensor (tensor_idx);

    assert (tensor_ptr->bytes == input[i].size);
    tensor_ptr->data.raw = (char *) input[i].data;
    tensors_idx.push_back (tensor_idx);
  }

  /* Configure outputs */
  for (i = 0; i < outputInfo.num_tensors; ++i) {
    tensor_idx = interpreter->outputs ()[i];
    tensor_ptr = interpreter->tensor (tensor_idx);

    assert (tensor_ptr->bytes == output[i].size);
    tensor_ptr->data.raw = (char *) output[i].data;
    tensors_idx.push_back (tensor_idx);
  }

  status = interpreter->Invoke ();

  /** if it is not `nullptr`, tensorflow makes `free()` the memory itself. */
  int tensorSize = tensors_idx.size ();
  for (int i = 0; i < tensorSize; ++i) {
    interpreter->tensor (tensors_idx[i])->data.raw = nullptr;
  }

  if (status != kTfLiteOk) {
    std::cerr << "Failed to invoke tensorflow-lite + edge-tpu." << std::endl;
    throw std::runtime_error ("Invoking tensorflow-lite with edge-tpu delgation failed.");
  }
}

void edgetpu_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info.name = name;
  info.allow_in_place = 0;
  info.allocate_in_invoke = 0;
  info.run_without_model = 0;
  info.verify_model_path = 1;
  info.hw_list = hw_list;
  info.num_hw = num_hw;
}

int edgetpu_subplugin::getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops == GET_IN_OUT_INFO) {
    gst_tensors_info_copy (std::addressof (in_info),
        std::addressof (inputInfo));
    gst_tensors_info_copy (std::addressof (out_info),
        std::addressof (outputInfo));
    return 0;
  }
  return -ENOENT;
}

int edgetpu_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  return -ENOENT;
}

/**
 * @brief Get TF-Lite interpreter w/ edgetpu context
 */
std::unique_ptr<tflite::Interpreter>
edgetpu_subplugin::BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel &model,
    edgetpu::EdgeTpuContext* edgetpu_context)
{
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) !=
      kTfLiteOk) {
    std::cerr << "Failed to build interpreter." << std::endl;
  }

  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }

  return interpreter;
}

/**
 * @brief from tflite-core
 * @todo Remove this func or make them shared
 */
int edgetpu_subplugin::getTensorDim (tflite::Interpreter *interpreter,
    int tensor_idx, tensor_dim dim)
{
  TfLiteIntArray *tensor_dims = interpreter->tensor (tensor_idx)->dims;
  int len = tensor_dims->size;
  g_assert (len <= NNS_TENSOR_RANK_LIMIT);

  /* the order of dimension is reversed at CAPS negotiation */
  std::reverse_copy (tensor_dims->data, tensor_dims->data + len, dim);

  /* fill the remnants with 1 */
  for (int i = len; i < NNS_TENSOR_RANK_LIMIT; ++i) {
    dim[i] = 1;
  }

  return 0;
}

/**
 * @brief From tflite-core.cc
 * @todo Remove this or make them shared
 */
tensor_type edgetpu_subplugin::getTensorType (TfLiteType tfType)
{
  switch (tfType) {
    case kTfLiteFloat32:
      return _NNS_FLOAT32;
    case kTfLiteUInt8:
      return _NNS_UINT8;
    case kTfLiteInt32:
      return _NNS_INT32;
    case kTfLiteBool:
      return _NNS_INT8;
    case kTfLiteInt64:
      return _NNS_INT64;
    case kTfLiteString:
    default:
      /** @todo Support other types */
      break;
  }

  return _NNS_END;
}

/**
 * @brief extract and store the information of given tensor list
 * @param interpreter The edgetpu tflite delegation interpreter
 * @param tensor_idx_list list of index of tensors in tflite interpreter
 * @param[out] tensorMeta tensors to set the info into
 * @throws std::invalid_argument if a given argument is not valid.
 */
void edgetpu_subplugin::setTensorProp (tflite::Interpreter *interpreter,
    const std::vector<int> &tensor_idx_list,
    GstTensorsInfo & tensorMeta)
{
  tensorMeta.num_tensors = tensor_idx_list.size ();
  if (tensorMeta.num_tensors > NNS_TENSOR_SIZE_LIMIT)
    throw std::invalid_argument ("The number of tensors required by the given model exceeds the nnstreamer tensor limit (16 by default).");

  for (unsigned int i = 0; i < tensorMeta.num_tensors; ++i) {
    if (getTensorDim (interpreter, tensor_idx_list[i], tensorMeta.info[i].dimension)) {
      std::cerr << "failed to get the dimension of tensors" << std::endl;
      throw std::invalid_argument ("Cannot get the dimensions of given tensors at the tensor ");
    }
    tensorMeta.info[i].type =
        getTensorType (interpreter->tensor (tensor_idx_list[i])->type);
    tensorMeta.info[i].name = nullptr; /** @todo tensor name is not retrieved */
  }
}

edgetpu_subplugin *edgetpu_subplugin::registeredRepresentation = nullptr;

/**@brief Initialize this object for tensor_filter subplugin runtime register */
void edgetpu_subplugin::init_filter_edgetpu (void)
{
  registeredRepresentation =
      tensor_filter_subplugin::register_subplugin<edgetpu_subplugin> ();
}

void _init_filter_edgetpu ()
{
  edgetpu_subplugin::init_filter_edgetpu();
}

/** @brief Destruct the subplugin */
void edgetpu_subplugin::fini_filter_edgetpu (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

void _fini_filter_edgetpu ()
{
  edgetpu_subplugin::fini_filter_edgetpu();
}

} /* namespace nnstreamer::tensorfilter_edgetpu */
} /* namespace nnstreamer */

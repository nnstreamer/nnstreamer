/* SPDX-License-Identifier: LGPL-2.1-only */
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
#include <fstream>
#include <iostream>
#include <sstream>
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

#if defined (TFLITE_VERSION)
constexpr char tflite_ver[] = G_STRINGIFY(TFLITE_VERSION);
#else
constexpr char tflite_ver[] = "1.x";
#endif /* defined (TFLITE_VER) */

namespace nnstreamer {
namespace tensorfilter_edgetpu {

void _init_filter_edgetpu (void) __attribute__ ((constructor));
void _fini_filter_edgetpu (void) __attribute__ ((destructor));

/** @brief enum for edgetpu device type */
enum class edgetpu_subplugin_device_type : uint32_t {
  USB = static_cast<uint32_t>(edgetpu::DeviceType::kApexUsb),
  PCI = static_cast<uint32_t>(edgetpu::DeviceType::kApexPci),
  DEFAULT = USB,
  DUMMY = 99,
};

/** @brief get device typename */
static const std::string edgetpu_subplugin_device_type_name (
    edgetpu_subplugin_device_type t)
{
    switch(t)
    {
        case edgetpu_subplugin_device_type::PCI:
          return "pci";
        case edgetpu_subplugin_device_type::DUMMY:
          return "dummy";
        case edgetpu_subplugin_device_type::USB:
        default:
          return "usb";
    }
}

/** @brief edgetpu subplugin class */
class edgetpu_subplugin final : public tensor_filter_subplugin {
private:
  bool empty_model;
  char *model_path; /**< The model *.tflite file */
  edgetpu_subplugin_device_type device_type; /**< The device type of Edge TPU */
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
          const edgetpu_subplugin_device_type dev_type,
          edgetpu::EdgeTpuContext* edgetpu_context = nullptr);

  /** Internal Utility Functions & Properties ***************************/
  void cleanup ();
  static void setTensorProp (tflite::Interpreter *interpreter,
      const std::vector<int> &tensor_idx_list, GstTensorsInfo & tensorMeta);
  static int getTensorDim ( tflite::Interpreter *interpreter, int tensor_idx,
      tensor_dim dim);
  static tensor_type getTensorType (TfLiteType tfType);
  static std::string str_tolower(std::string s);
  static edgetpu_subplugin_device_type parse_custom_prop (
      const char * custom_prop);
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

/** @brief edgetpu class constructor */
edgetpu_subplugin::edgetpu_subplugin () :
    tensor_filter_subplugin (),
    empty_model (true),
    model_path (nullptr),
    device_type (edgetpu_subplugin_device_type::DEFAULT),
    model_interpreter (nullptr),
    edgetpu_context (nullptr),
    model (nullptr)
{
  inputInfo.num_tensors = 0;
  outputInfo.num_tensors = 0;
  /** Nothing to do. Just let it have an empty instance */
}

/** @brief cleanup resources used by edgetpu subplugin */
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

/** @brief edgetpu class destructor */
edgetpu_subplugin::~edgetpu_subplugin ()
{
  cleanup ();
}

/** @brief get empty instance of edgetpu subplugin */
tensor_filter_subplugin & edgetpu_subplugin::getEmptyInstance ()
{
  return *(new edgetpu_subplugin());
}

/**
 * @brief Internal helper to get lowercase string of the given one.
 * @param s The given std:string value
 * @return A lowercase version of s
 */
std::string edgetpu_subplugin::str_tolower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
      [](unsigned char c){ return std::tolower(c); });

  return s;
}

/**
 * @brief Internal method to parse custom properties
 * @param custom_prop The given c_str value of the 'custom' property
 * @return A edgetpu_subplugin_device_type value for the given 'device_type'
 *        if the value of the 'custom' property. Otherwise,
 *        edgetpu_subplugin_device_type::USB, which is default, is returned.
 */
edgetpu_subplugin_device_type edgetpu_subplugin::parse_custom_prop (
    const char *custom_prop)
{
  if ((!custom_prop) || (strlen (custom_prop) == 0))
    return edgetpu_subplugin_device_type::DEFAULT;

  const std::string cprop (custom_prop);
  const std::string key ("device_type");
  const std::size_t max_vec_len = 2;
  std::vector<std::string> vec;
  std::stringstream cprop_ss;
  std::string token;
  std::size_t pos;

  pos = cprop.find_first_of (',');
  cprop_ss = std::stringstream (cprop.substr (0, pos));

  while (std::getline (cprop_ss, token, ':')) {
    vec.push_back (token);
    if (vec.size () > max_vec_len)
      break;
  }

  if (edgetpu_subplugin::str_tolower (vec[0]) != key)
    return edgetpu_subplugin_device_type::DEFAULT;

  std::string val = edgetpu_subplugin::str_tolower (vec[1]);
  edgetpu_subplugin_device_type ret;

  ret = edgetpu_subplugin_device_type::PCI;
  if (edgetpu_subplugin_device_type_name (ret) == val)
    return ret;

  ret = edgetpu_subplugin_device_type::DUMMY;
  if (edgetpu_subplugin_device_type_name (ret) == val)
    return ret;

  return edgetpu_subplugin_device_type::DEFAULT;
}

/** @brief configure edgetpu instance */
void edgetpu_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  const std::string _model_path = prop->model_files[0];

  this->device_type = this->parse_custom_prop (prop->custom_properties);

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
    cleanup();
    throw std::invalid_argument ("Cannot load the given model file.");
  }

  /** Build an interpreter */
  if (this->device_type != edgetpu_subplugin_device_type::DUMMY) {
    edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
    if (nullptr == edgetpu_context) {
      std::cerr << "Cannot open edge-TPU device." << std::endl;
      cleanup();
      throw std::system_error (ENODEV, std::system_category(),
          "Cannot open edge-TPU device.");
    }

    model_interpreter = BuildEdgeTpuInterpreter (*model, this->device_type,
        edgetpu_context.get ());
    if (nullptr == model_interpreter) {
      std::cerr << "Edge-TPU device is opened, but cannot get its interpreter."
          << std::endl;
      cleanup();
      throw std::system_error (ENODEV, std::system_category(),
          "Edge-TPU device is opened, but cannot get its interpreter.");
    }
  } else {
    /* If the device_type is 'dummy', work same as tflite using CPU */
    tflite::ops::builtin::BuiltinOpResolver resolver;

    edgetpu_context = nullptr;
    model_interpreter = BuildEdgeTpuInterpreter(*model, this->device_type);
    if (nullptr == model_interpreter) {
      cleanup();
      throw std::system_error (ENODEV, std::system_category(),
          "Failed to get the interpreter while trying to running dummy device mode of Edge-TPU.");
    }
  }

  interpreter = model_interpreter.get();

  try {
    setTensorProp (interpreter, interpreter->inputs (), inputInfo);
  } catch (const std::invalid_argument& ia) {
    cleanup();
    throw std::invalid_argument ("Input tensor of the given model is incompatible or invalid");
  }

  try {
    setTensorProp (interpreter, interpreter->outputs (), outputInfo);
  } catch (const std::invalid_argument& ia) {
    cleanup();
    throw std::invalid_argument ("Output tensor of the given model is incompatible or invalid");
  }

  empty_model = false;
}

/** @brief invoke using edgetpu */
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
    std::ifstream ifs;

    ifs.open (model_path);
    if (!ifs.fail ()) {
      /**
       * In the case of models compiled by edgetpu-compiler, there is a string,
       * 'edgetpu-custom-op', at the position 0x5C of the .tflite file.
       */
      const std::streampos compiled_model_id_pos (0x5C);
      const std::string compiled_model_id ("edgetpu-custom-op");
      std::vector<char> buf (compiled_model_id.size ());

      ifs.seekg (compiled_model_id_pos);
      ifs.read (buf.data (), compiled_model_id.size ());
      ifs.close ();

      if (compiled_model_id == std::string (buf.data ())) {
        /** The given model is a compiled model */
        nns_loge ("A compiled model by edgetpu-compiler has been given, but this extension might be statically linked with TensorFlow Lite v%s which does not support the model.",
            tflite_ver);
        nns_loge ("To use this model, we recommend to upgrade the tensorflow-lite package to version 1.15.2 and rebuild the subplugin with the upgraded tensorflow-lite.");
      }
    }

    std::cerr << "Failed to invoke tensorflow-lite + edge-tpu." << std::endl;
    throw std::runtime_error ("Invoking tensorflow-lite with edge-tpu delgation failed.");
  }
}

/** @brief Get framework information */
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

/** @brief Get model information */
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

/** @brief Event handler (TBD) */
int edgetpu_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  return -ENOENT;
}

/**
 * @brief Get TF-Lite interpreter w/ edgetpu context
 */
std::unique_ptr<tflite::Interpreter>
edgetpu_subplugin::BuildEdgeTpuInterpreter(const tflite::FlatBufferModel &model,
    const edgetpu_subplugin_device_type dev_type,
    edgetpu::EdgeTpuContext* edgetpu_context)
{
  tflite::ops::builtin::BuiltinOpResolver resolver;

  if (dev_type != edgetpu_subplugin_device_type::DUMMY)
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) !=
      kTfLiteOk) {
    nns_loge ("Failed to build interpreter.");
  }

  if (dev_type != edgetpu_subplugin_device_type::DUMMY)
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);

  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    nns_loge ("Failed to allocate tensors.");
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

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void edgetpu_subplugin::init_filter_edgetpu (void)
{
  registeredRepresentation =
      tensor_filter_subplugin::register_subplugin<edgetpu_subplugin> ();
}

/** @brief initializer */
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

/** @brief finalizer */
void _fini_filter_edgetpu ()
{
  edgetpu_subplugin::fini_filter_edgetpu();
}

} /* namespace nnstreamer::tensorfilter_edgetpu */
} /* namespace nnstreamer */

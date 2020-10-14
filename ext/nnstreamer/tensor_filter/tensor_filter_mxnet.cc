/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Tensor_Filter, MXNet Module
 * Copyright (C) 2020 Bumsik Kim <k.bumsik@gmail.com>
 */
/**
 * @file	tensor_filter_mxnet.cc
 * @date	5 Oct 2020
 * @brief	MXNet module for tensor_filter gstreamer plugin
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	Bumsik Kim <k.bumsik@gmail.com>
 * @bug		No known bugs except for NYI items
 *
 * This is the MXNet plugin for tensor_filter.
 *
 * @details Usage examples
 *
 *          Case 1: simple ImageNet example found in test case:
 *                  https://github.com/nnstreamer/nnstreamer/tree/main/tests/nnstreamer_filter_mxnet/simple_test_mxnet.cc
 *            gst-launch-1.0 \
 *               appsrc ! application/octet-stream \
 *               ! tensor_converter input-dim=1:3:224:224 input-type=float32 \
 *               ! tensor_filter \
 *                   framework=mxnet \
 *                   model=model/Inception-BN.json \
 *                   input=1:3:224:224 \
 *                   inputtype=float32 \
 *                   inputname=data \
 *                   output=1 \
 *                   outputtype=float32 \
 *                   outputname=argmax_channel \
 *                   custom=input_rank=4 \
 *                   accelerator=true:cpu,!npu,!gpu \
 *               ! appsink",
 *
 * @note Special considerations on props:
 *
 *   outputname:
 *     The output name should not be the name of output layer, but the name of
 *     MXNet NDArray operator passed to the output. For example, the most common
 *     name is "argmax_channel" for typical classification model. It also can be
 *     "sigmoid" or etc.
 *     If it is not meant to do any operations, "_copyto" operator may be used.
 *     The output will be passed as-is.
 *     The current limitation is that it only takes a unary operator, meaning it
 *     does not take additional parameters.
 *     The candidates of operators can be found here:
 *      - https://github.com/apache/incubator-mxnet/blob/1.7.0/benchmark/opperf/nd_operations/unary_operators.py#L24-L28
 *      - https://mxnet.apache.org/versions/1.7/api/python/docs/api/ndarray/op/index.html?highlight=argmax_channel#mxnet.ndarray.op.argmax_channel
 *
 *     If this prop is not configured appropriately, the application may get
 *     unexpected output values, or even segfaults without warning messages.
 *
 *   custom:
 *     Each entries are separated by ','
 *     Each entries have property_key=value format.
 *     There must be no speces.
 *
 *     Supported props:
 *       input_rank: (mandatory)
 *            Rank of each input tensors.
 *            Each ranks are separeted by ':'.
 *            The number of ranks must be the same as the number of input
 *            tensors.
 *
 *     Examples:
 *       tensor_filter framework=mxnet model=model/Inception-BN.json
 *                input=1:3:224:224
 *                inputname=data
 *                ...
 *                custom=input_rank=4
 *
 *        tensor_filter framework=mxnet model=model/Inception-BN.json
 *                input=1:3:224:224,1
 *                inputname=data1,data2
 *                ...
 *                custom=input_rank=4:1
 */
#include <nnstreamer_cppplugin_api_filter.hh>
#include <stdexcept>
#include <tensor_common.h>

#include <fcntl.h>
#include <glib.h>
#include <glib/gstdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <mxnet-cpp/MxNetCpp.h>

using nnstreamer::tensor_filter_subplugin;
using namespace mxnet::cpp;

namespace nnstreamer
{
namespace tensorfilter_mxnet
{

const static std::string kFileLocation = "/ext/nnstreamer/tensor_filter/tensor_filter_mxnet.cc";
const static std::string kFileUrl = "https://github.com/nnstreamer/nnstreamer/tree/main" + kFileLocation;

void init_filter_mxnet (void) __attribute__ ((constructor)); /**< Dynamic library contstructor */
void fini_filter_mxnet (void) __attribute__ ((destructor)); /**< Dynamic library desctructor */

class TensorFilterMXNet final : public tensor_filter_subplugin
{
  public:
  static void init_filter (); /**< Dynamic library contstructor helper */
  static void fini_filter (); /**< Dynamic library desctructor helper */

  TensorFilterMXNet ();
  ~TensorFilterMXNet ();

  /**< Implementations of tensor_filter_subplugin */
  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);

  static const std::string ext_symbol; /**< extension of model symbol (.json) */
  static const std::string ext_params; /**< extension of model parameters (.params) */

  /**< Data type for MXNet NDArray from <mshadow/base.h> which causes errors when included */
  enum TypeFlag : int {
    kFloat32 = 0,
    kFloat64 = 1,
    kFloat16 = 2,
    kUint8 = 3,
    kInt32 = 4,
    kInt8 = 5,
    kInt64 = 6,
    kBool = 7,
    kInt16 = 8,
    kUint16 = 9,
    kUint32 = 10,
    kUint64 = 11,
    kBfloat16 = 12
  };

  private:
  Shape tensorInfoToShape (GstTensorInfo &tensorinfo, int rank);
  TypeFlag tensorTypeToMXNet (tensor_type type);
  void parseCustomProperties (const GstTensorFilterProperties *prop);

  bool empty_model_; /**< Empty (not initialized) model flag */
  static const GstTensorFilterFrameworkInfo info_; /**< Framework info */
  GstTensorsInfo inputs_info_; /**< Input tensors metadata */
  GstTensorsInfo outputs_info_; /**< Output tensors metadata */

  // GstTensorInfo does not contain rank info, so extra fields needed */
  int input_ranks_[NNS_TENSOR_RANK_LIMIT]; /**< Rank info of input tensor */
  int output_ranks_[NNS_TENSOR_RANK_LIMIT]; /**< Rank info of output tensor */

  std::string model_symbol_path_; /**< The model symbol .json file */
  std::string model_params_path_; /**< The model paremeters .params file */

  Symbol net_; /**< Model symbol */
  std::unique_ptr<Executor> executor_; /**< Model executor */
  std::map<std::string, NDArray> args_map_; /**< arguments information of model, used internally by MXNet */
  std::map<std::string, NDArray> aux_map_; /**< auxiliary information of model, used internally by MXNet */
  Context ctx_; /**< Device type (CPU or GPU) */

  static TensorFilterMXNet *registeredRepresentation;
};

const std::string TensorFilterMXNet::ext_symbol = ".json";
const std::string TensorFilterMXNet::ext_params = ".params";

const GstTensorFilterFrameworkInfo TensorFilterMXNet::info_ = { .name = "mxnet",
  .allow_in_place = FALSE,
  .allocate_in_invoke = FALSE,
  .run_without_model = FALSE,
  .verify_model_path = TRUE,
  .hw_list = (const accl_hw[]){ ACCL_CPU },
  .num_hw = 1,
  .accl_auto = static_cast<accl_hw>(-1),
  .accl_default = static_cast<accl_hw>(-1),
  .statistics = nullptr };

TensorFilterMXNet::TensorFilterMXNet ()
    : tensor_filter_subplugin (), empty_model_ (true), ctx_ (Context::cpu ())
{
  /** Nothing to do. Just let it have an empty instance */
}

TensorFilterMXNet::~TensorFilterMXNet ()
{
  executor_.reset ();
}

tensor_filter_subplugin &
TensorFilterMXNet::getEmptyInstance ()
{
  return *(new TensorFilterMXNet ());
}

void
TensorFilterMXNet::configure_instance (const GstTensorFilterProperties *prop)
{
  if (prop->num_models != 1) {
    throw std::invalid_argument ("Multiple models is not supported.");
  }

  try {
    parseCustomProperties (prop);
  } catch (const std::invalid_argument &e) {
    throw std::invalid_argument ("Failed to parse \"custom\" prop:"
                                 + std::string (e.what ()) + "\n\tReference: " + kFileUrl);
  }

  // Validate model file paths and then assign
  std::tie (model_symbol_path_, model_params_path_) = [&] {
    const std::vector<std::string> extensions{ TensorFilterMXNet::ext_symbol,
      TensorFilterMXNet::ext_params };

    // trim extension like .json
    const std::string model_path = [&] {
      std::string path (prop->model_files[0]);
      for (const std::string &ext : extensions) {
        if (g_str_has_suffix (path.c_str (), ext.c_str ())) {
          return path.substr (0, path.length () - ext.length ());
        }
      }
      return path;
    }();

    // validate paths
    for (const std::string &ext : extensions) {
      std::string path (model_path + ext);
      if (g_access (path.c_str (), R_OK) != 0) {
        throw std::invalid_argument ("Failed to open the " + ext + " file, " + path);
      }
    }

    return std::make_tuple (model_path + TensorFilterMXNet::ext_symbol,
        model_path + TensorFilterMXNet::ext_params);
  }();

  // Read a model
  net_ = Symbol::Load (model_symbol_path_);

  // Load parameters into temporary array maps
  // The following loop split loaded param map into arg parm
  // and aux param with target context
  std::map<std::string, NDArray> parameters;
  NDArray::Load (model_params_path_, nullptr, &parameters);
  for (const auto &pair : parameters) {
    std::string type = pair.first.substr (0, 4);
    std::string name = pair.first.substr (4);
    if (type == "arg:") {
      args_map_[name] = pair.second.Copy (ctx_);
    } else if (type == "aux:") {
      aux_map_[name] = pair.second.Copy (ctx_);
    }
  }

  // WaitAll is need when we copy data between GPU and the main memory
  NDArray::WaitAll ();

  gst_tensors_info_copy (&inputs_info_, &prop->input_meta);
  gst_tensors_info_copy (&outputs_info_, &prop->output_meta);

  // Set ndarrays for the input layers
  for (unsigned int i = 0; i < inputs_info_.num_tensors; i++) {
    auto &input_tensor = inputs_info_.info[i];
    args_map_[input_tensor.name]
        = NDArray (tensorInfoToShape (input_tensor, input_ranks_[i]), ctx_,
            false, tensorTypeToMXNet (input_tensor.type));
  }

  // These are ndarrays where the execution engine runs
  std::vector<NDArray> arg_arrays;
  std::vector<NDArray> grad_arrays;
  std::vector<OpReqType> grad_reqs;
  std::vector<NDArray> aux_arrays;

  try {
    // infer and create ndarrays according to the given array maps.
    net_.InferExecutorArrays (ctx_, &arg_arrays, &grad_arrays, &grad_reqs,
        &aux_arrays, args_map_, std::map<std::string, NDArray> (),
        std::map<std::string, OpReqType> (), aux_map_);
    for (auto& i : grad_reqs) i = OpReqType::kNullOp;

    // Create a symbolic execution engine after binding the model to input parameters.
    executor_ = std::make_unique<Executor> (
        net_, ctx_, arg_arrays, grad_arrays, grad_reqs, aux_arrays);
  } catch (const std::exception &e) {
    throw std::system_error (ENODEV, std::system_category (),
        std::string (e.what ()) + "\nFailed to bind parameters to the model. This is mostly caused by wrong \"input\" and \"inputname\" information.");
  }
  empty_model_ = false;
}

void
TensorFilterMXNet::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  assert (!empty_model_);
  assert (executor_);

  // Copy input
  for (unsigned int i = 0; i < inputs_info_.num_tensors; i++) {
    auto &input_info = inputs_info_.info[i];
    auto &input_ndarray = args_map_[input_info.name];

    assert ((input_ndarray.Size () * sizeof (mx_float)) == input[i].size);
    input_ndarray.SyncCopyFromCPU (
        (const mx_float *)input[i].data, input_ndarray.Size ());
    NDArray::WaitAll ();
  }

  // Run forward pass
  executor_->Forward (false);
  NDArray::WaitAll ();

  // Copy output
  for (unsigned int i = 0; i < outputs_info_.num_tensors; i++) {
    auto &output_info = outputs_info_.info[i];
    NDArray result;

    // Warning: It will cause segfault if the operator name (output name) is different from expected.
    //          The user should know the name of the operator name.
    Operator (output_info.name) (executor_->outputs[0]).Invoke (result);
    NDArray::WaitAll ();

    assert ((result.Size () * sizeof (mx_float)) == output[i].size);
    result.SyncCopyToCPU ((mx_float *)output[i].data, result.Size ());
    NDArray::WaitAll ();
  }
}

void
TensorFilterMXNet::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  info = info_;
}

int
TensorFilterMXNet::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  switch (ops) {
  case GET_IN_OUT_INFO:
    gst_tensors_info_copy (std::addressof (in_info), std::addressof (inputs_info_));
    gst_tensors_info_copy (std::addressof (out_info), std::addressof (outputs_info_));
    break;
  case SET_INPUT_INFO:
  default:
    return -ENOENT;
  }
  return 0;
}

int
TensorFilterMXNet::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  return -ENOENT;
}

/**
 * @brief Convert GstTensorInfo to MXNet Shape
 */
Shape
TensorFilterMXNet::tensorInfoToShape (GstTensorInfo &tensorinfo, int rank)
{
  return Shape (std::vector<index_t> (tensorinfo.dimension, tensorinfo.dimension + rank));
}

/**
 * @brief Convert tensor_type to MXNet TypeFlag
 */
TensorFilterMXNet::TypeFlag
TensorFilterMXNet::tensorTypeToMXNet (tensor_type type)
{
  switch (type) {
  case _NNS_INT32:
    return kInt32;
  case _NNS_UINT32:
    return kUint32;
  case _NNS_INT16:
    return kInt16;
  case _NNS_UINT16:
    return kUint16;
  case _NNS_INT8:
    return kInt8;
  case _NNS_UINT8:
    return kUint8;
  case _NNS_FLOAT64:
    return kFloat64;
  case _NNS_FLOAT32:
    return kFloat32;
  case _NNS_INT64:
    return kInt64;
  case _NNS_UINT64:
    return kUint64;
  default:
    throw std::invalid_argument ("Unsupported data type.");
  }
}

/**
 * @brief Parse custom prop and set instance options accordingly.
 */
void
TensorFilterMXNet::parseCustomProperties (const GstTensorFilterProperties *prop)
{
  using unique_g_strv = std::unique_ptr<gchar *, std::function<void (gchar **)>>;
  bool is_input_rank_parsed = false;

  if (prop->custom_properties) {
    unique_g_strv options (g_strsplit (prop->custom_properties, ",", -1), g_strfreev);
    guint len = g_strv_length (options.get ());

    for (guint i = 0; i < len; i++) {
      unique_g_strv option (g_strsplit (options.get ()[i], "=", -1), g_strfreev);

      if (g_strv_length (option.get ()) > 1) {
        g_strstrip (option.get ()[0]);
        g_strstrip (option.get ()[1]);

        if (g_ascii_strcasecmp (option.get ()[0], "input_rank") == 0) {
          unique_g_strv ranks (g_strsplit (option.get ()[1], ":", -1), g_strfreev);
          guint num_outputs = g_strv_length (ranks.get ());

          if (num_outputs != prop->input_meta.num_tensors) {
            throw std::invalid_argument (
                "The number of ranks does not match the number of input tensors.");
          }

          for (guint i = 0; i < num_outputs; i++) {
            input_ranks_[i] = g_ascii_strtoull (ranks.get ()[i], nullptr, 10);
          }
          is_input_rank_parsed = true;
        } else {
          throw std::invalid_argument (
              "Unsupported custom property: " + std::string (option.get ()[0]) + ".");
        }
      }
    }
  }

  if (!is_input_rank_parsed) {
    throw std::invalid_argument ("\"input_rank\" must be set. e.g. custom=input_rank=4:1");
  }
  return;
}

TensorFilterMXNet *TensorFilterMXNet::registeredRepresentation = nullptr;

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
TensorFilterMXNet::init_filter (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<TensorFilterMXNet> ();
}

void
init_filter_mxnet ()
{
  TensorFilterMXNet::init_filter ();
}

/** @brief Destruct the subplugin */
void
TensorFilterMXNet::fini_filter (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

void
fini_filter_mxnet ()
{
  TensorFilterMXNet::fini_filter ();
}

} // namespace tensorfilter_mxnet
} /* namespace nnstreamer */

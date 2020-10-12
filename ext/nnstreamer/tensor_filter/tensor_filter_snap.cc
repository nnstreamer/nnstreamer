/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file	tensor_filter_snap.cc
 * @date	11 Oct 2019
 * @brief	NNStreamer tensor-filter sub-plugin for SNAP
 * @see		https://github.com/nnstreamer/nnstreamer
 * @see		https://developer.samsung.com/neural/overview.html
 * @author	Jaeyun Jung <jy1210.jung@samsung.com>
 * @bug		No known bugs
 *
 * SNAP (Samsung Neural Acceleration Platform) version 2.0, run only on Samsung devices.
 * Developer should download Samsung Neural SDK (https://developer.samsung.com/neural/overview.html).
 *
 * To construct a pipeline with SNAP, you should set the custom option string to specify the neural network and data format.
 * Custom options:
 *  - ModelFWType: the type of model (TensorFlow/Caffe)
 *  - ExecutionDataType: the execution data type for SNAP (default float32)
 *  - ComputingUnit: the computing unit to execute the model (default CPU)
 *  - CpuThreadCount: the number of CPU threads to be executed (optional, default 4 if ComputingUnit is CPU)
 *  - GpuCacheSource: the absolute path to GPU Kernel caching (mandatory if ComputingUnit is GPU)
 */

#if !defined(__ANDROID__)
#error "This sub-plugin only can be included in Android (ndk-r17c or higher, arm64-v8a only)."
#endif

#include <android/log.h>

#include <snap_sdk_interface.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_cppplugin_api_filter.hh>

#define TAG "NNStreamer-SNAP"
#define snap_logi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define snap_logw(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define snap_loge(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define snap_logd(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

using nnstreamer::tensor_filter_subplugin;

/**
 * @brief Options to open SNAP session.
 */
typedef struct
{
  snap_sdk::ModelFWType fw_type;
  snap_sdk::ExecutionDataType exec_data_type;
  snap_sdk::ComputingUnit computing_unit;
  std::vector<snap_sdk::DataFormat> input_format;
  std::vector<snap_sdk::DataFormat> output_format;
  std::string gpu_cache_src;
  int cpu_thread_count;
  bool model_encrypted;
} snap_option_s;

/**
 * @brief Data information to execute the model.
 */
typedef struct
{
  snap_sdk::DataType type;
  snap_sdk::DataFormat format;
  std::vector<int> shape;
} snap_data_info_s;

/**
 * @brief Class for SNAP subplugin.
 */
class tensor_filter_snap final : public tensor_filter_subplugin
{
public:
  tensor_filter_snap ();
  ~tensor_filter_snap ();

  tensor_filter_subplugin& getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);

  static void register_snap ();
  static void unregister_snap ();

private:
  bool open (const GstTensorFilterProperties *prop, snap_option_s &snap_option);
  void close ();
  bool validate (const GstTensorFilterProperties *prop, snap_option_s &snap_option);
  bool invoke_internal (const GstTensorFilterProperties *prop, const GstTensorMemory *input, GstTensorMemory *output, bool configure);
  bool parse_custom_prop (const char *custom_prop, snap_option_s &snap_option);
  bool configure_option (const GstTensorFilterProperties *prop, snap_option_s &snap_option);
  bool configure_input_meta (const GstTensorFilterProperties *prop, snap_option_s &snap_option);
  bool get_nns_type (snap_sdk::DataType snap_type, tensor_type &nns_type);
  bool convert_nns_type (tensor_type nns_type, snap_sdk::DataType &snap_type);
  bool get_nns_layout (snap_sdk::DataFormat snap_format, tensor_layout &nns_layout);
  bool get_snap_data_format (tensor_layout nns_layout, snap_sdk::DataFormat &snap_format);
  bool convert_nns_layouts (const tensors_layout nns_layout, unsigned int len, std::vector<snap_sdk::DataFormat> &format);
  bool parse_format_string (const gchar *format_str, bool is_input, std::vector<snap_sdk::DataFormat> &format);
  bool parse_dimension (const std::vector<int> &shape, tensor_dim dim);
  bool convert_names (const GstTensorsInfo *info, std::vector<std::string> &names);
  bool compare_meta (const GstTensorFilterProperties *prop, std::vector<snap_sdk::SnapData> &data, bool is_input);
  const char* error_string (snap_sdk::ErrCode status);

private:
  static tensor_filter_snap *instance_;
  snap_sdk::SnapSessionInterface *session_;
  std::vector<snap_data_info_s> in_info_;
  std::vector<snap_data_info_s> out_info_;
  GstTensorsInfo input_meta_;
  GstTensorsInfo output_meta_;
};

/**
 * @brief Internal instance for SNAP registration.
 */
tensor_filter_snap *tensor_filter_snap::instance_ = nullptr;

/**
 * @brief Constructor for SNAP subplugin.
 */
tensor_filter_snap::tensor_filter_snap () : session_(nullptr)
{
  in_info_.clear ();
  out_info_.clear ();

  gst_tensors_info_init (std::addressof (input_meta_));
  gst_tensors_info_init (std::addressof (output_meta_));
}

/**
 * @brief Destructor for SNAP subplugin.
 */
tensor_filter_snap::~tensor_filter_snap ()
{
  close ();

  gst_tensors_info_free (std::addressof (input_meta_));
  gst_tensors_info_free (std::addressof (output_meta_));
}

/**
 * @brief Mandatory method to get empty object.
 */
tensor_filter_subplugin&
tensor_filter_snap::getEmptyInstance ()
{
  return *(new tensor_filter_snap ());
}

/**
 * @brief Mandatory method to prepare in/out data information and open SNAP session.
 */
void
tensor_filter_snap::configure_instance (const GstTensorFilterProperties *prop)
{
  snap_option_s snap_option;
  std::string snap_ver = snap_sdk::GetSnapSDKVersionName ();

  snap_logi ("Start to open SNAP session (%s).", snap_ver.c_str ());

  if (!configure_option (prop, snap_option)) {
    snap_loge ("Failed to configure SNAP options.");
    return;
  }

  if (!open (prop, snap_option)) {
    snap_loge ("Failed to create SNAP session.");
    return;
  }

  /* validate in/out meta */
  if (!validate (prop, snap_option)) {
    snap_loge ("Failed to validate in/out meta.");
    close ();
    return;
  }

  gst_tensors_info_copy (std::addressof (input_meta_),
      std::addressof (prop->input_meta));
  gst_tensors_info_copy (std::addressof (output_meta_),
      std::addressof (prop->output_meta));
}

/**
 * @brief Mandatory method to execute the model.
 */
void
tensor_filter_snap::invoke (const GstTensorMemory *input,
    GstTensorMemory *output)
{
  invoke_internal (nullptr, input, output, false);
}

/**
 * @brief Mandatory method to get the base information of SNAP subplugin.
 */
void
tensor_filter_snap::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
{
  static const char filter_subplugin_name_snap[] = "snap";

  info.name = filter_subplugin_name_snap;
  info.allow_in_place = 0;
  info.allocate_in_invoke = 0;
  info.run_without_model = 0;
  info.verify_model_path = 1;
  info.hw_list = nullptr;
  info.num_hw = 0;
}

/**
 * @brief Mandatory method to get the model information.
 * Internally SNAP subplugin uses user-defined tensor information.
 */
int
tensor_filter_snap::getModelInfo (model_info_ops ops,
    GstTensorsInfo &in_info, GstTensorsInfo &out_info)
{
  if (ops == GET_IN_OUT_INFO) {
    gst_tensors_info_copy (std::addressof (in_info),
        std::addressof (input_meta_));
    gst_tensors_info_copy (std::addressof (out_info),
        std::addressof (output_meta_));
    return 0;
  }

  return -ENOENT;
}

/**
 * @brief Optional method to handle the event.
 */
int
tensor_filter_snap::eventHandler (event_ops ops,
    GstTensorFilterFrameworkEventData &data)
{
  return -ENOENT;
}

/**
 * @brief Registers SNAP subplugin.
 */
void
tensor_filter_snap::register_snap ()
{
  if (instance_ == nullptr) {
    instance_ =
        tensor_filter_subplugin::register_subplugin<tensor_filter_snap> ();
  }
}

/**
 * @brief Unregisters SNAP subplugin.
 */
void
tensor_filter_snap::unregister_snap ()
{
  if (instance_) {
    tensor_filter_subplugin::unregister_subplugin (instance_);
    instance_ = nullptr;
  }
}

/**
 * @brief Internal method to get the error string.
 */
const char*
tensor_filter_snap::error_string (snap_sdk::ErrCode status)
{
  static const char *err_string[] = {
    [static_cast<int>(snap_sdk::ErrCode::OK)] = "OK",
    [static_cast<int>(snap_sdk::ErrCode::ERR)] = "ERR",
    [static_cast<int>(snap_sdk::ErrCode::UNSUPPORTED_DEVICE)] = "UNSUPPORTED_DEVICE",
    [static_cast<int>(snap_sdk::ErrCode::UNSUPPORTED_MODELTYPE)] = "UNSUPPORTED_MODELTYPE",
    [static_cast<int>(snap_sdk::ErrCode::SNAPMODEL_NOTCREATED)] = "SNAPMODEL_NOTCREATED",
    [static_cast<int>(snap_sdk::ErrCode::MODELTYPE_NOTVALID)] = "MODELTYPE_NOTVALID",
    [static_cast<int>(snap_sdk::ErrCode::INPUTBUFFER_EMPTY)] = "INPUTBUFFER_EMPTY",
    [static_cast<int>(snap_sdk::ErrCode::UNSUPPORTED_EXEC_TYPE)] = "UNSUPPORTED_EXEC_TYPE",
    [static_cast<int>(snap_sdk::ErrCode::UNDEFINED_GPU_CACHEPATH)] = "UNDEFINED_GPU_CACHEPATH",
    [static_cast<int>(snap_sdk::ErrCode::INVALID_INPUT_PARAM)] = "INVALID_INPUT_PARAM",
  };

  return err_string[static_cast<int>(status)];
}

/**
 * @brief Internal method to open SNAP session.
 */
bool
tensor_filter_snap::open (const GstTensorFilterProperties *prop,
    snap_option_s &snap_option)
{
  snap_sdk::SnapModel model;
  snap_sdk::ExecutionOptions exec_options;
  snap_sdk::ErrCode status;
  std::string model_file;
  std::string weight_file;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  if (session_ != nullptr) {
    snap_logw ("SNAP session is already opened.");
    return true;
  }

  /* set model and weight files */
  if (prop->num_models < 1) {
    snap_loge ("Invalid model file, please set the model file.");
    return false;
  }

  model_file = std::string (prop->model_files[0]);
  if (prop->num_models > 1)
    weight_file = std::string (prop->model_files[1]);
  else
    weight_file = std::string (prop->model_files[0]);

  snap_logd ("model file (%s) weight file (%s)",
      model_file.c_str (), weight_file.c_str ());

  /* set tensor name */
  if (!convert_names (&prop->input_meta, input_names) ||
      !convert_names (&prop->output_meta, output_names)) {
    return false;
  }

  /* create model */
  status = snap_sdk::SnapModel::Create (snap_option.fw_type,
      input_names, output_names, weight_file, model_file, model);
  if (status != snap_sdk::ErrCode::OK) {
    snap_loge ("Failed to create snap model (%s).", error_string (status));
    goto done;
  }

  /** @note snap v2.0 only supports caffe model to set encrypted */
  if (snap_option.fw_type == snap_sdk::ModelFWType::CAFFE) {
    status = model.SetEncrypted (snap_option.model_encrypted);
    if (status != snap_sdk::ErrCode::OK) {
      snap_loge ("Failed to set encrypted (%s).", error_string (status));
      goto done;
    }
  }

  /* create execution option */
  status = snap_sdk::ExecutionOptions::Create (snap_option.computing_unit,
      snap_option.exec_data_type, model, exec_options);
  if (status != snap_sdk::ErrCode::OK) {
    snap_loge ("Failed to create execution option (%s).",
        error_string (status));
    goto done;
  }

  if (snap_option.computing_unit == snap_sdk::ComputingUnit::CPU) {
    if (snap_option.cpu_thread_count > 0) {
      status = exec_options.SetCpuThreadCount (snap_option.cpu_thread_count);
      if (status != snap_sdk::ErrCode::OK) {
        snap_loge ("Failed to set CPU thread count (%s).",
            error_string (status));
        goto done;
      }
    }
  } else if (snap_option.computing_unit == snap_sdk::ComputingUnit::GPU) {
    status = exec_options.SetGpuCacheSource (snap_option.gpu_cache_src);
    if (status != snap_sdk::ErrCode::OK) {
      snap_loge ("Failed to set GPU cache source (%s).", error_string (status));
      goto done;
    }
  }

  /* create session */
  status = snap_sdk::CreateSnapSession (&session_);
  if (status != snap_sdk::ErrCode::OK) {
    snap_loge ("Failed to create session (%s).", error_string (status));
    goto done;
  }

  status = session_->Open (model, exec_options);
  if (status != snap_sdk::ErrCode::OK) {
    snap_loge ("Failed to open session (%s).", error_string (status));
    goto done;
  }

done:
  if (status != snap_sdk::ErrCode::OK) {
    /* failed */
    close ();
    return false;
  }

  return true;
}

/**
 * @brief Internal method to close SNAP session.
 */
void
tensor_filter_snap::close ()
{
  /* close session */
  if (session_ != nullptr) {
    snap_sdk::ErrCode status;

    snap_logi ("Start to close SNAP session.");

    status = session_->Close ();
    if (status != snap_sdk::ErrCode::OK) {
      snap_loge ("Failed to close session (%s).", error_string (status));
    }

    status = snap_sdk::DestroySnapSession (session_);
    if (status != snap_sdk::ErrCode::OK) {
      snap_loge ("Failed to destroy session (%s).", error_string (status));
    }

    session_ = nullptr;
  }

  in_info_.clear ();
  out_info_.clear ();
}

/**
 * @brief Internal method to validate user-defined tensors information.
 */
bool
tensor_filter_snap::validate (const GstTensorFilterProperties *prop,
    snap_option_s &snap_option)
{
  GstTensorMemory in_tensors[NNS_TENSOR_SIZE_LIMIT] = { 0, };
  bool validated = false;
  guint i;

  /* Configure input info from properties and options. */
  if (!configure_input_meta (prop, snap_option)) {
    snap_loge ("Failed to configure input meta.");
    return false;
  }

  /* Invoke with dummy data to validate output meta. */
  for (i = 0; i < prop->input_meta.num_tensors; i++) {
    in_tensors[i].size = gst_tensor_info_get_size (&prop->input_meta.info[i]);
    in_tensors[i].data = g_malloc0 (in_tensors[i].size);
  }

  validated = invoke_internal (prop, in_tensors, nullptr, true);

  for (i = 0; i < prop->input_meta.num_tensors; i++)
    g_free (in_tensors[i].data);

  if (!validated)
    snap_loge ("Failed to validate tensor meta.");
  return validated;
}

/**
 * @brief Internal method to execute the model.
 */
bool
tensor_filter_snap::invoke_internal (const GstTensorFilterProperties *prop,
    const GstTensorMemory *input, GstTensorMemory *output, bool configure)
{
  snap_sdk::ErrCode status;
  std::vector<snap_sdk::SnapData> inputs;
  std::vector<snap_sdk::SnapData> outputs;
  guint i;

  if (session_ == nullptr) {
    snap_loge ("Invalid state, SNAP session is not opened.");
    return false;
  }

  /* input data */
  for (i = 0; i < in_info_.size (); ++i) {
    snap_sdk::SnapData input_data;

    input_data.SetData (input[i].data, in_info_[i].shape, in_info_[i].type,
        in_info_[i].format);
    inputs.push_back (input_data);
  }

  /* invoke */
  status = session_->Execute (inputs, &outputs);
  if (status != snap_sdk::ErrCode::OK) {
    snap_loge ("Failed to execute (%s).", error_string (status));
    return false;
  }

  /* validate and configure tensor meta */
  if (configure) {
    if (!compare_meta (prop, inputs, true)) {
      snap_loge ("The input meta is not matched with configured.");
      return false;
    }

    if (!compare_meta (prop, outputs, false)) {
      snap_loge ("The output meta is not matched with configured.");
      return false;
    }

    for (i = 0; i < outputs.size (); i++) {
      snap_data_info_s snap_info = {
        .type = outputs[i].GetType (),
        .format = outputs[i].GetFormat (),
        .shape = outputs[i].GetShapes ()
      };

      out_info_.push_back (snap_info);
    }
  }

  /* fill output data */
  if (output != nullptr) {
    for (i = 0; i < outputs.size (); ++i) {
      /** @todo is it possible to remove memcpy? */
      memcpy (output[i].data, outputs[i].GetBuffer (), output[i].size);
    }
  }

  return true;
}

/**
 * @brief Internal method to parse option string.
 * Developer should define proper option to open SNAP session.
 */
bool
tensor_filter_snap::parse_custom_prop (const char *custom_prop,
    snap_option_s &snap_option)
{
  gchar **options;
  guint op;
  bool invalid_option = false;

  if (custom_prop == nullptr) {
    snap_loge ("Cannot get the options, please add proper option string.");
    return false;
  }

  options = g_strsplit (custom_prop, ",", -1);

  for (op = 0; op < g_strv_length (options); ++op) {
    gchar **option = g_strsplit (options[op], ":", -1);

    if (g_strv_length (option) > 1) {
      g_strstrip (option[0]);
      g_strstrip (option[1]);

      if (g_ascii_strcasecmp (option[0], "ModelFWType") == 0) {
        if (g_ascii_strcasecmp (option[1], "TENSORFLOW") == 0) {
          snap_option.fw_type = snap_sdk::ModelFWType::TENSORFLOW;
        } else if (g_ascii_strcasecmp (option[1], "CAFFE") == 0) {
          snap_option.fw_type = snap_sdk::ModelFWType::CAFFE;
        } else {
          snap_logw ("Unknown FW type (%s).", options[op]);
          invalid_option = true;
        }
      } else if (g_ascii_strcasecmp (option[0], "ModelEncrypted") == 0) {
        /** @note snap v2.0 only supports caffe model to set encrypted */
        if (g_ascii_strcasecmp (option[1], "true") == 0) {
          snap_option.model_encrypted = true;
        }
      } else if (g_ascii_strcasecmp (option[0], "ExecutionDataType") == 0) {
        if (g_ascii_strcasecmp (option[1], "FLOAT32") == 0) {
          snap_option.exec_data_type = snap_sdk::ExecutionDataType::FLOAT32;
        } else if (g_ascii_strcasecmp (option[1], "FLOAT16") == 0) {
          snap_option.exec_data_type = snap_sdk::ExecutionDataType::FLOAT16;
        } else if (g_ascii_strcasecmp (option[1], "QASYMM16") == 0) {
          snap_option.exec_data_type = snap_sdk::ExecutionDataType::QASYMM16;
        } else if (g_ascii_strcasecmp (option[1], "QASYMM8") == 0) {
          snap_option.exec_data_type = snap_sdk::ExecutionDataType::QASYMM8;
        } else {
          snap_logw ("Unknown execution type (%s), set float32 as default.",
              options[op]);
          snap_option.exec_data_type = snap_sdk::ExecutionDataType::FLOAT32;
        }
      } else if (g_ascii_strcasecmp (option[0], "InputFormat") == 0) {
        /* data formats with separator "-" (e.g., NHWC-NHWC for 2 inputs) */
        if (!parse_format_string (option[1], true, snap_option.input_format)) {
          snap_loge ("Failed to set the input data layout.");
          invalid_option = true;
        }
      } else if (g_ascii_strcasecmp (option[0], "OutputFormat") == 0) {
        if (!parse_format_string (option[1], false, snap_option.output_format)) {
          snap_loge ("Failed to set the output data layout.");
          invalid_option = true;
        }
      } else if (g_ascii_strcasecmp (option[0], "ComputingUnit") == 0) {
        if (g_ascii_strcasecmp (option[1], "CPU") == 0) {
          snap_option.computing_unit = snap_sdk::ComputingUnit::CPU;
        } else if (g_ascii_strcasecmp (option[1], "GPU") == 0) {
          snap_option.computing_unit = snap_sdk::ComputingUnit::GPU;
        } else if (g_ascii_strcasecmp (option[1], "NPU") == 0) {
          snap_option.computing_unit = snap_sdk::ComputingUnit::NPU;
        } else if (g_ascii_strcasecmp (option[1], "DSP") == 0) {
          snap_option.computing_unit = snap_sdk::ComputingUnit::DSP;
        } else {
          snap_logw ("Unknown computing unit (%s), set CPU as default.",
              options[op]);
          snap_option.computing_unit = snap_sdk::ComputingUnit::CPU;
        }
      } else if (g_ascii_strcasecmp (option[0], "CpuThreadCount") == 0) {
        snap_option.cpu_thread_count =
            (int) g_ascii_strtoll (option[1], NULL, 10);
      } else if (g_ascii_strcasecmp (option[0], "GpuCacheSource") == 0) {
        gchar *path = option[1];

        /* append dir separator */
        if (!G_IS_DIR_SEPARATOR (path[strlen (path) - 1])) {
          path = g_strconcat (path, G_DIR_SEPARATOR_S, NULL);

          g_free (option[1]);
          option[1] = path;
        }

        snap_option.gpu_cache_src = std::string (path);
      } else {
        snap_logw ("Unknown option (%s).", options[op]);
      }
    }

    g_strfreev (option);

    if (invalid_option)
      break;
  }

  g_strfreev (options);
  return !invalid_option;
}

/**
 * @brief Internal method to set the options to open SNAP session.
 */
bool
tensor_filter_snap::configure_option (const GstTensorFilterProperties *prop,
    snap_option_s &snap_option)
{
  /* init options */
  snap_option.fw_type = snap_sdk::ModelFWType::MAX_ENUM;
  snap_option.exec_data_type = snap_sdk::ExecutionDataType::FLOAT32;
  snap_option.computing_unit = snap_sdk::ComputingUnit::CPU;
  snap_option.cpu_thread_count = -1;
  snap_option.model_encrypted = false;
  snap_option.gpu_cache_src = "";
  snap_option.input_format.clear ();
  snap_option.output_format.clear ();

  if (!parse_custom_prop (prop->custom_properties, snap_option)) {
    snap_loge ("Cannot get the options to open SNAP session.");
    return false;
  }

  if (snap_option.fw_type == snap_sdk::ModelFWType::MAX_ENUM) {
    snap_loge ("Failed to identify the FW type.");
    return false;
  }

  /* configure in/out data format */
  if (snap_option.input_format.size () == 0 &&
      !convert_nns_layouts (prop->input_layout, prop->input_meta.num_tensors,
          snap_option.input_format)) {
    snap_loge ("Failed to configure input data format.");
    return false;
  }

  if (prop->input_meta.num_tensors != snap_option.input_format.size ()) {
    snap_loge ("The size of input data format is not matched.");
    return false;
  }

  if (snap_option.output_format.size () == 0 &&
      !convert_nns_layouts (prop->output_layout, prop->output_meta.num_tensors,
          snap_option.output_format)) {
    snap_loge ("Failed to configure output data format.");
    return false;
  }

  if (prop->output_meta.num_tensors != snap_option.output_format.size ()) {
    snap_loge ("The size of output data format is not matched.");
    return false;
  }

  /* check computing unit */
  if (snap_option.computing_unit == snap_sdk::ComputingUnit::GPU &&
      snap_option.gpu_cache_src.empty ()) {
    snap_loge ("GPU cache path is not defined.");
    return false;
  }

  return true;
}

/**
 * @brief Internal method to configure input info from the properties and options.
 */
bool
tensor_filter_snap::configure_input_meta (const GstTensorFilterProperties *prop,
    snap_option_s &snap_option)
{
  snap_sdk::ErrCode status;
  guint i, j;
  gulong rank;

  if (session_ == nullptr) {
    snap_loge ("Invalid state, SNAP session is not opened.");
    return false;
  }

  for (i = 0; i < prop->input_meta.num_tensors; i++) {
    snap_data_info_s snap_info;
    snap_sdk::SnapData in_data;

    snap_info.format = snap_option.input_format[i];

    if (!convert_nns_type (prop->input_meta.info[i].type, snap_info.type)) {
      snap_logw ("Failed to convert input type.");
      return false;
    }

    /* Note that, returned shape is NCHW format. */
    status = session_->GetModelInputShape (i, &snap_info.shape);
    if (status != snap_sdk::ErrCode::OK) {
      snap_logw ("Failed to get the input shape with index %u.", i);
      return false;
    }

    rank = snap_info.shape.size ();
    snap_logd ("The rank of model input shape is %lu.", rank);

    if (rank == 0) {
      /**
       * Cannot get the exact dimension (e.g., tensorflow).
       * Set the dimension from input tensors info.
       */
      rank = NNS_TENSOR_RANK_LIMIT;
      for (j = 0; j < rank; j++) {
        int s = (int) prop->input_meta.info[i].dimension[rank - j - 1];
        snap_info.shape.push_back (s);
      }
    } else {
      /* SNAP supports rank 4 */
      if (rank > NNS_TENSOR_RANK_LIMIT) {
        for (j = NNS_TENSOR_RANK_LIMIT; j < rank; j++) {
          if (snap_info.shape[j] != 0) {
            snap_logd ("The model input shape at %d is %d.", j,
                snap_info.shape[j]);
            snap_loge ("The rank of model input shape is not supported.");
            return false;
          }
        }

        snap_info.shape.erase (snap_info.shape.begin () + NNS_TENSOR_RANK_LIMIT,
            snap_info.shape.end ());
        rank = NNS_TENSOR_RANK_LIMIT;
      }

      if (snap_info.format != snap_sdk::DataFormat::NCHW) {
        /* convert data format (NCHW > NHWC) */
        int c = snap_info.shape[1];
        snap_info.shape[1] = snap_info.shape[2];
        snap_info.shape[2] = snap_info.shape[3];
        snap_info.shape[3] = c;
      }
    }

    /* add input meta */
    in_info_.push_back (snap_info);
  }

  return true;
}

/**
 * @brief Internal method to get the tensor type from SNAP data type.
 */
bool
tensor_filter_snap::get_nns_type (snap_sdk::DataType snap_type,
    tensor_type &nns_type)
{
  /** @todo snap v2.0 only supports float32 type */
  if (snap_type == snap_sdk::DataType::FLOAT32) {
    nns_type = _NNS_FLOAT32;
    return true;
  }

  snap_logw ("The data type %d is not supported.", static_cast<int>(snap_type));
  return false;
}

/**
 * @brief Internal method to convert the tensor type to SNAP data type.
 */
bool
tensor_filter_snap::convert_nns_type (tensor_type nns_type,
    snap_sdk::DataType &snap_type)
{
  /** @todo snap v2.0 only supports float32 type */
  if (nns_type == _NNS_FLOAT32) {
    snap_type = snap_sdk::DataType::FLOAT32;
    return true;
  }

  snap_logw ("The tensor type %d is not supported.", nns_type);
  return false;
}

/**
 * @brief Internal method to get the tensor layout from SNAP data format.
 */
bool
tensor_filter_snap::get_nns_layout (snap_sdk::DataFormat snap_format,
    tensor_layout &nns_layout)
{
  if (snap_format == snap_sdk::DataFormat::NCHW) {
    nns_layout = _NNS_LAYOUT_NCHW;
  } else if (snap_format == snap_sdk::DataFormat::NHWC) {
    nns_layout = _NNS_LAYOUT_NHWC;
  } else {
    snap_logw ("The data format %d is not supported.",
        static_cast<int>(snap_format));
    nns_layout = _NNS_LAYOUT_NONE;
  }

  return (nns_layout != _NNS_LAYOUT_NONE);
}

/**
 * @brief Internal method to convert the tensor layout to SNAP data format.
 */
bool
tensor_filter_snap::get_snap_data_format (tensor_layout nns_layout,
    snap_sdk::DataFormat &snap_format)
{
  switch (nns_layout) {
    case _NNS_LAYOUT_NHWC:
      snap_format = snap_sdk::DataFormat::NHWC;
      break;
    case _NNS_LAYOUT_NCHW:
      snap_format = snap_sdk::DataFormat::NCHW;
      break;
    default:
      snap_logw ("The tensor layout %d is not supported.", nns_layout);
      snap_format = snap_sdk::DataFormat::MAX_ENUM;
      break;
  }

  return (snap_format != snap_sdk::DataFormat::MAX_ENUM);
}

/**
 * @brief Internal method to get the SNAP data formats from the array of tensor layouts.
 */
bool
tensor_filter_snap::convert_nns_layouts (const tensors_layout nns_layout,
    unsigned int len, std::vector<snap_sdk::DataFormat> &format)
{
  guint i;

  for (i = 0; i < len; i++) {
    snap_sdk::DataFormat snap_format = snap_sdk::DataFormat::MAX_ENUM;

    if (!get_snap_data_format (nns_layout[i], snap_format)) {
      snap_logw ("Failed to get data format, given layout with index %u is %d.",
          i, nns_layout[i]);
      return false;
    }

    format.push_back (snap_format);
  }

  return true;
}

/**
 * @brief Internal method to get the SNAP data from from the string.
 */
bool
tensor_filter_snap::parse_format_string (const gchar *format_str, bool is_input,
    std::vector<snap_sdk::DataFormat> &format)
{
  gchar **formats;
  guint i, len;
  bool is_valid = false;

  formats = g_strsplit (format_str, "-", -1);
  len = g_strv_length (formats);

  for (i = 0; i < len; i++) {
    snap_sdk::DataFormat snap_format;

    if (g_ascii_strcasecmp (formats[i], "NHWC") == 0) {
      snap_format = snap_sdk::DataFormat::NHWC;
    } else if (g_ascii_strcasecmp (formats[i], "NCHW") == 0) {
      snap_format = snap_sdk::DataFormat::NCHW;
    } else {
      snap_logw ("Invalid data format %s.", formats[i]);
      goto done;
    }

    format.push_back (snap_format);
  }

  is_valid = true;

done:
  g_strfreev (formats);
  return is_valid;
}

/**
 * @brief Internal method to get the tensor dimension from SNAP data shape.
 */
bool
tensor_filter_snap::parse_dimension (const std::vector<int> &shape,
    tensor_dim dim)
{
  unsigned long i, rank;

  rank = shape.size ();
  if (rank > NNS_TENSOR_RANK_LIMIT) {
    /* supposed max rank is less than 4 (NNS_TENSOR_RANK_LIMIT) */
    snap_loge ("The rank is invalid (%lu).", rank);
    return false;
  }

  for (i = 0; i < rank; i++) {
    snap_logd ("snap data shape[%lu] : %d", i, shape[i]);
    dim[rank - i - 1] = (unsigned int) shape[i];
  }

  /* fill the remnants with 1 */
  for (i = rank; i < NNS_TENSOR_RANK_LIMIT; i++) {
    dim[i] = 1;
  }

  return true;
}

/**
 * @brief Internal method to get the tensor names from tensors information.
 */
bool
tensor_filter_snap::convert_names (const GstTensorsInfo *info,
    std::vector<std::string> &names)
{
  guint i;

  for (i = 0; i < info->num_tensors; ++i) {
    if (info->info[i].name == nullptr) {
      /* failed */
      snap_loge ("Given tensor name with index %d is invalid.", i);
      return false;
    }

    names.push_back (std::string (info->info[i].name));
  }

  return true;
}

/**
 * @brief Internal method to validate the tensors information.
 */
bool
tensor_filter_snap::compare_meta (const GstTensorFilterProperties *prop,
    std::vector<snap_sdk::SnapData> &data, bool is_input)
{
  const GstTensorsInfo *nns_info;
  const tensor_layout *nns_layout;
  guint i;

  if (is_input) {
    nns_info = &prop->input_meta;
    nns_layout = prop->input_layout;
  } else {
    nns_info = &prop->output_meta;
    nns_layout = prop->output_layout;
  }

  if (data.size () != nns_info->num_tensors) {
    return false;
  }

  for (i = 0; i < nns_info->num_tensors; ++i) {
    GstTensorInfo snap_info;
    tensor_layout snap_layout = _NNS_LAYOUT_NONE;

    gst_tensor_info_init (&snap_info);

    snap_logi ("The data format at index %d is %s.", i,
        (data[i].GetFormat () == snap_sdk::DataFormat::NCHW) ? "NCHW" : "NHWC");

    if (!get_nns_type (data[i].GetType (), snap_info.type) ||
        !get_nns_layout (data[i].GetFormat (), snap_layout) ||
        !parse_dimension (data[i].GetShapes (), snap_info.dimension)) {
      snap_loge ("Failed to parse the tensor meta.");
      return false;
    }

    if (!gst_tensor_info_is_equal (&nns_info->info[i], &snap_info)) {
      snap_logw ("Given tensor info is not equal.");
      return false;
    }

    if (nns_layout[i] != snap_layout && nns_layout[i] != _NNS_LAYOUT_ANY) {
      snap_logw ("The layout at index %d is not matched (layout %d:%d)", i,
          nns_layout[i], snap_layout);
      return false;
    }
  }

  return true;
}

G_BEGIN_DECLS

/**
 * @brief Register the sub-plugin for SNAP.
 */
void
init_filter_snap (void)
{
  const GstTensorFilterFramework *snpe = nnstreamer_filter_find ("snpe");

  if (snpe) {
    snap_loge ("Found SNPE sub-plugin. NNStreamer does not support SNAP with SNPE runtime.");
    return;
  }

  tensor_filter_snap::register_snap ();
}

/**
 * @brief Destruct the sub-plugin for SNAP.
 */
void
fini_filter_snap (void)
{
  tensor_filter_snap::unregister_snap ();
}

G_END_DECLS

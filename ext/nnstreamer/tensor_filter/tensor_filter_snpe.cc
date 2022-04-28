/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * NNStreamer tensor_filter, sub-plugin for SNPE
 * Copyright (C) 2020 Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 */
/**
 * @file	tensor_filter_snpe.cc
 * @date	24 Apr 2020
 * @brief	NNStreamer tensor-filter sub-plugin for SNPE (Qualcomm Neural Processing SDK)
 * @see		http://github.com/nnstreamer/nnstreamer
 * @see		https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk
 * @author	Yongjoo Ahn <yongjoo1.ahn@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (SNPE) for tensor_filter.
 *
 * @todo This supports only ITensor for input. Do support IUserBuffer.
 * @todo This supports float32 input output only. Do support Tf8 using IUserBuffer.
 * @todo This supports only CPU runtime on linux-x86_64. Do support others.
 */

#include <iostream>
#include <string>

#include <glib.h>
#include <nnstreamer_cppplugin_api_filter.hh>
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include <tensor_common.h>

#include <DlContainer/IDlContainer.hpp>
#include <DlSystem/ITensorFactory.hpp>
#include <DlSystem/RuntimeList.hpp>
#include <DlSystem/TensorMap.hpp>
#include <DlSystem/IUserBufferFactory.hpp>
#include <SNPE/SNPE.hpp>
#include <SNPE/SNPEBuilder.hpp>
#include <SNPE/SNPEFactory.hpp>

#if defined(__ANDROID__)
#include <jni.h>
#endif

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

namespace nnstreamer
{
namespace tensor_filter_snpe
{

extern "C" {
#if defined(__ANDROID__)
void init_filter_snpe (JNIEnv *env, jobject context);
#else
void init_filter_snpe (void) __attribute__((constructor));
#endif
void fini_filter_snpe (void) __attribute__((destructor));
}

/** @brief tensor-filter-subplugin concrete class for SNPE */
class snpe_subplugin final : public tensor_filter_subplugin
{
  private:
  bool empty_model;
  char *model_path; /**< The model *.dlc file */
  GstTensorsInfo inputInfo; /**< Input tensors metadata */
  GstTensorsInfo outputInfo; /**< Output tensors metadata */
#if (DBG)
  gint64 total_frames;
  gint64 invoke_time_total;
#endif

  /* options for snpe builder */
  zdl::DlSystem::RuntimeList runtime_list;
  bool use_cpu_fallback;
  zdl::DlSystem::StringList output_tensor_names_list, input_tensor_names_list;
  size_t max_resizable_dim;

  std::unique_ptr<zdl::DlContainer::IDlContainer> container;
  std::unique_ptr<zdl::SNPE::SNPE> snpe;

  zdl::DlSystem::TensorMap input_tensor_map;
  zdl::DlSystem::TensorMap output_tensor_map;
  std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> input_tensors;

  static const char *name;
  static snpe_subplugin *registeredRepresentation;

  void cleanup ();
  bool configure_option (const GstTensorFilterProperties *prop);
  bool parse_custom_prop (const char *custom_prop);
  bool set_output_tensor_names (const GstTensorsInfo *info);
  void configureUserBuffer (zdl::DlSystem::UserBufferMap &buffer_map, const zdl::DlSystem::StringList strList);
  void setTensorProp (GstTensorsInfo &tensor_meta, const zdl::DlSystem::StringList strList, tensor_type data_type);
  static const char *runtimeToString (zdl::DlSystem::Runtime_t runtime);

  tensor_type input_data_type;
  tensor_type output_data_type;

  bool use_user_buffer;
  zdl::DlSystem::UserBufferMap input_buffer_map;
  zdl::DlSystem::UserBufferMap output_buffer_map;
  std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> user_input_buffers;
  std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> user_output_buffers;

  public:
  static void init_filter_snpe ();
  static void fini_filter_snpe ();

  snpe_subplugin ();
  ~snpe_subplugin ();

  tensor_filter_subplugin &getEmptyInstance ();
  void configure_instance (const GstTensorFilterProperties *prop);
  void invoke (const GstTensorMemory *input, GstTensorMemory *output);
  void getFrameworkInfo (GstTensorFilterFrameworkInfo &info);
  int getModelInfo (model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info);
  int eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data);
};

const char *snpe_subplugin::name = "snpe";

/**
 * @brief Constructor for snpe_subplugin.
 */
snpe_subplugin::snpe_subplugin ()
    : tensor_filter_subplugin (), empty_model (true), model_path (nullptr),
      runtime_list (zdl::DlSystem::Runtime_t::CPU), use_cpu_fallback (false),
      output_tensor_names_list (), input_tensor_names_list (), max_resizable_dim (0U),
      container (nullptr), snpe (nullptr), input_data_type (_NNS_FLOAT32),
      output_data_type (_NNS_FLOAT32), use_user_buffer (false)
{
  gst_tensors_info_init (std::addressof (inputInfo));
  gst_tensors_info_init (std::addressof (outputInfo));
  input_tensors.reserve (NNS_TENSOR_RANK_LIMIT);
#if (DBG)
  invoke_time_total = total_frames = 0;
#endif
}

/**
 * @brief Method to cleanup snpe subplugin.
 */
void
snpe_subplugin::cleanup ()
{
  if (model_path) {
    g_free (model_path);
    model_path = nullptr;
  }

  if (empty_model)
    return;

  if (container) {
    container = nullptr;
  }

  if (snpe) {
    snpe.reset ();
    snpe = nullptr;
  }

  gst_tensors_info_free (std::addressof (inputInfo));
  gst_tensors_info_free (std::addressof (outputInfo));

  runtime_list.clear ();
  input_tensors.clear ();
  input_tensor_map.clear ();
  output_tensor_map.clear ();
  output_tensor_names_list = zdl::DlSystem::StringList ();
  input_tensor_names_list = zdl::DlSystem::StringList ();

  empty_model = true;
}

/**
 * @brief Destructor for snpe subplugin.
 */
snpe_subplugin::~snpe_subplugin ()
{
#if (DBG)
  nns_logd ("Average Invoke latency: %" G_GINT64_FORMAT "us, for total: %" G_GINT64_FORMAT
            " frames, used model: %s, used runtime: %s",
      (invoke_time_total / total_frames), total_frames, model_path,
      runtimeToString (runtime_list[0]));
#endif

  cleanup ();
}

/**
 * @brief Method to get empty object.
 */
tensor_filter_subplugin &
snpe_subplugin::getEmptyInstance ()
{
  return *(new snpe_subplugin ());
}

/**
 * @brief Method to get string of SNPE runtime.
 */
const char *
snpe_subplugin::runtimeToString (zdl::DlSystem::Runtime_t runtime)
{
  switch (runtime) {
  case zdl::DlSystem::Runtime_t::CPU:
    return "CPU";
  case zdl::DlSystem::Runtime_t::GPU:
    return "GPU";
  case zdl::DlSystem::Runtime_t::DSP:
    return "DSP";
  case zdl::DlSystem::Runtime_t::AIP_FIXED8_TF:
    return "AIP_FIXED8_TF";
  default:
    return "invalid_runtime...";
  }
}

/**
 * @brief Internal method to get names of output layers from tensors information.
 */
bool
snpe_subplugin::set_output_tensor_names (const GstTensorsInfo *info)
{
  if (output_tensor_names_list.size () > 0) {
    output_tensor_names_list = zdl::DlSystem::StringList ();
  }
  for (unsigned int i = 0; i < info->num_tensors; ++i) {
    if (info->info[i].name == nullptr || info->info[i].name[0] == '\0') {
      /* failed */
      nns_loge ("Given output tensor name with index %u is invalid, it is null.", i);
      return false;
    }
    output_tensor_names_list.append (info->info[i].name);
  }
  return true;
}

/**
 * @brief Internal method to parse custom options.
 */
bool
snpe_subplugin::parse_custom_prop (const char *custom_prop)
{
  gchar **options;
  bool invalid_option = false;

  if (!custom_prop) {
    /* no custom properties were given */
    return true;
  }

  options = g_strsplit (custom_prop, ",", -1);

  for (guint op = 0; op < g_strv_length (options); ++op) {
    gchar **option = g_strsplit (options[op], ":", -1);

    if (g_strv_length (option) > 1) {
      g_strstrip (option[0]);
      g_strstrip (option[1]);

      if (g_ascii_strcasecmp (option[0], "Runtime") == 0) {
        zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
        if (g_ascii_strcasecmp (option[1], "CPU") == 0) {
          runtime = zdl::DlSystem::Runtime_t::CPU;
        } else if (g_ascii_strcasecmp (option[1], "GPU") == 0) {
          runtime = zdl::DlSystem::Runtime_t::GPU;
        } else if (g_ascii_strcasecmp (option[1], "DSP") == 0) {
          runtime = zdl::DlSystem::Runtime_t::DSP;
        } else if (g_ascii_strcasecmp (option[1], "NPU") == 0) {
          runtime = zdl::DlSystem::Runtime_t::AIP_FIXED8_TF;
        } else {
          nns_logw ("Unknown runtime (%s), set CPU as default.", options[op]);
          invalid_option = true;
        }
        if (zdl::SNPE::SNPEFactory::isRuntimeAvailable (runtime)) {
          runtime_list.clear ();
          nns_logi ("Set runtime to %s", runtimeToString (runtime));
          runtime_list.add (runtime);
        } else {
          nns_loge ("All runtime is not available...");
        }
      } else if (g_ascii_strcasecmp (option[0], "CPUFallback") == 0) {
        if (g_ascii_strcasecmp (option[1], "true") == 0) {
          use_cpu_fallback = true;
          nns_logd ("Enable CPU fallback.");
        } else if (g_ascii_strcasecmp (option[1], "false") == 0) {
          use_cpu_fallback = false;
        } else {
          nns_loge ("Unknown cpu_fallback option");
          invalid_option = true;
        }
      } else if (g_ascii_strcasecmp (option[0], "OutputTensor") == 0) {
        /* the tensor name may contain ':' */
        gchar *_ot_str = g_strjoinv (":", &option[1]);
        gchar **names = g_strsplit (_ot_str, ";", -1);
        guint num_names = g_strv_length (names);
        for (guint i = 0; i < num_names; ++i) {
          if (g_strcmp0 (names[i], "") == 0) {
            nns_loge ("Given tensor name with index %u is invalid.", i);
            invalid_option = true;
            break;
          }
          nns_logd ("Add output tensor name of %s", names[i]);
          output_tensor_names_list.append (names[i]);
        }
        g_free (_ot_str);
        g_strfreev (names);
      } else if (g_ascii_strcasecmp (option[0], "InputType") == 0) {
        if (g_ascii_strcasecmp (option[1], "uint8") == 0) {
          input_data_type = _NNS_UINT8;
          nns_logi ("Set input data type as uint8");
        } else {
          input_data_type = _NNS_FLOAT32;
          nns_logi ("Set input data type as default (float32)");
        }
      } else if (g_ascii_strcasecmp (option[0], "OutputType") == 0) {
        if (g_ascii_strcasecmp (option[1], "uint8") == 0) {
          output_data_type = _NNS_UINT8;
          nns_logi ("Set output data type as uint8");
        } else {
          output_data_type = _NNS_FLOAT32;
          nns_logi ("Set output data type as default (float32)");
        }
      } else if (g_ascii_strcasecmp (option[0], "UserBuffer") == 0) {
        if (g_ascii_strcasecmp (option[1], "true") == 0) {
          use_user_buffer = true;
          nns_logi ("Use user supplied buffer for input/output");
        } else {
          use_user_buffer = false;
          nns_logi ("Use ITENSOR for input/output (default option)");
        }
      } else if (g_ascii_strcasecmp (option[0], "MaxResizableDim") == 0) {
        max_resizable_dim = (size_t) g_ascii_strtoll (option[1], NULL, 10);
        nns_logi ("Set %zu for max number of resizable dim", max_resizable_dim);
        if (max_resizable_dim == 0) {
          nns_loge ("max_resizable_dim should be greater than 0");
          invalid_option = true;
        }
      } else {
        nns_logw ("Unknown option (%s).", options[op]);
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
 * @brief Internal method to set the options for SNPE instance.
 */
bool
snpe_subplugin::configure_option (const GstTensorFilterProperties *prop)
{
  if (!parse_custom_prop (prop->custom_properties)) {
    nns_loge ("Cannot get the proper custom properties.");
    return false;
  }

  return true;
}

/**
 * @brief Method to prepare/configure SNPE instance.
 */
void
snpe_subplugin::configure_instance (const GstTensorFilterProperties *prop)
{
  nns_logi ("SNPE Version: %s",
      zdl::SNPE::SNPEFactory::getLibraryVersion ().asString ().c_str ());

  if (!set_output_tensor_names (&prop->output_meta)) {
    nns_loge ("Failed to set output layer names");
    cleanup ();
    return;
  }

  if (!configure_option (prop)) {
    cleanup ();
    throw std::invalid_argument ("Failed to configure SNPE option.");
    return;
  }

  if (!empty_model) {
    /* Already opened */

    if (!prop->model_files[0] || prop->model_files[0][0] == '\0') {
      std::cerr << "Model path is not given." << std::endl;
      throw std::invalid_argument ("Model path is not given.");
    }

    cleanup ();
  }

  if (!g_file_test (prop->model_files[0], G_FILE_TEST_IS_REGULAR)) {
    const std::string err_msg = "Given file " + (std::string) prop->model_files[0] + " is not valid";
    std::cerr << err_msg << std::endl;
    cleanup ();
    throw std::invalid_argument (err_msg);
  }

  assert (model_path == nullptr);

  model_path = g_strdup (prop->model_files[0]);

  container = zdl::DlContainer::IDlContainer::open (model_path);

  zdl::SNPE::SNPEBuilder snpe_builder (container.get());
  if (output_tensor_names_list.size () > 0) {
    nns_logd ("Use user given output tensor names");
    snpe_builder.setOutputTensors (output_tensor_names_list);
  }
  snpe_builder.setUseUserSuppliedBuffers (use_user_buffer);
  snpe_builder.setInitCacheMode (false);
  snpe_builder.setRuntimeProcessorOrder (runtime_list);
  snpe_builder.setCPUFallbackMode (use_cpu_fallback);

  snpe = snpe_builder.build ();
  if (snpe == nullptr) {
    cleanup ();
    throw std::runtime_error ("fail to build snpe");
  }

  /** configure input and output tensor names */
  if (output_tensor_names_list.size () == 0) {
    /** when no output tensor names are given -> use default output tensors */
    nns_logd ("No options are given for output tensors, use default output tensors of the model");
    output_tensor_names_list = snpe->getOutputTensorNames ();
  }

  input_tensor_names_list = snpe->getInputTensorNames ();

  /** user buffer mode */
  if (use_user_buffer) {
    if (input_data_type != _NNS_FLOAT32 || output_data_type != _NNS_FLOAT32) {
      cleanup ();
      throw std::invalid_argument ("user buffer mode only support float32 type");
    }

    /* Configure input and output */
    configureUserBuffer (input_buffer_map, input_tensor_names_list);
    configureUserBuffer (output_buffer_map, output_tensor_names_list);

  } else {  /** ITENSOR mode */
    for (size_t i = 0; i < input_tensor_names_list.size (); ++i) {
      const zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> &inputDims_opt
          = snpe->getInputDimensions (input_tensor_names_list.at (i));
      const zdl::DlSystem::TensorShape &input_shape = *inputDims_opt;

      input_tensors.emplace_back (
          zdl::SNPE::SNPEFactory::getTensorFactory ().createTensor (input_shape));
      input_tensor_map.add (input_tensor_names_list.at (i), input_tensors[i].get ());
    }
  }

  setTensorProp (inputInfo, input_tensor_names_list, input_data_type);
  setTensorProp (outputInfo, output_tensor_names_list, output_data_type);

  empty_model = false;
}

/**
 * @brief Method to execute the model.
 */
void
snpe_subplugin::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  assert (!empty_model);
  assert (snpe);

#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  if (use_user_buffer) {
    for (unsigned int i = 0; i < inputInfo.num_tensors; ++i) {
      input_buffer_map.getUserBuffer (inputInfo.info[i].name)->setBufferAddress (input[i].data);
    }

    for (unsigned int i = 0; i < outputInfo.num_tensors; ++i) {
      output_buffer_map.getUserBuffer (outputInfo.info[i].name)->setBufferAddress (output[i].data);
    }

    snpe->execute (input_buffer_map, output_buffer_map);
  } else {
    /* Configure inputs */
    for (unsigned int i = 0; i < inputInfo.num_tensors; ++i) {
      size_t fsize = input_tensors[i].get ()->getSize ();

      switch (input_data_type) {
        case _NNS_FLOAT32: {
          float *inbuf = (float *) input[i].data;
          std::copy (inbuf, inbuf + fsize, input_tensors[i].get ()->begin ());
          break;
        }
        case _NNS_UINT8: {
          uint8_t *inbuf = (uint8_t *) input[i].data;
          std::copy (inbuf, inbuf + fsize, input_tensors[i].get ()->begin ());
          break;
        }
        default:
          throw std::runtime_error ("Got invalid input data type");
      }
    }

    output_tensor_map.clear ();
    snpe->execute (input_tensor_map, output_tensor_map);

    for (unsigned int i = 0; i < outputInfo.num_tensors; ++i) {
      zdl::DlSystem::ITensor *output_tensor = output_tensor_map.getTensor (output_tensor_names_list.at (i));
      switch (output_data_type) {
        case _NNS_FLOAT32: {
          float *outbuf = (float *) output[i].data;
          std::copy (output_tensor->cbegin (), output_tensor->cend (), outbuf);
          break;
        }
        case _NNS_UINT8: {
          uint8_t *outbuf = (uint8_t *) output[i].data;
          std::copy (output_tensor->cbegin (), output_tensor->cend (), outbuf);
          break;
        }
        default:
          throw std::runtime_error ("Got invalid output data type");
      }
    }
  }

#if (DBG)
  gint64 stop_time = g_get_real_time ();

  invoke_time_total += (stop_time - start_time);
  total_frames++;
#endif
}

/**
 * @brief Method to get the information of SNPE subplugin.
 */
void
snpe_subplugin::getFrameworkInfo (GstTensorFilterFrameworkInfo &info)
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
snpe_subplugin::getModelInfo (
    model_info_ops ops, GstTensorsInfo &in_info, GstTensorsInfo &out_info)
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
snpe_subplugin::eventHandler (event_ops ops, GstTensorFilterFrameworkEventData &data)
{
  UNUSED (ops);
  UNUSED (data);
  return -ENOENT;
}

/**
 * @brief Method to configure user_buffer_map with given strList of tensor names
 */
void
snpe_subplugin::configureUserBuffer (zdl::DlSystem::UserBufferMap &buffer_map, const zdl::DlSystem::StringList strList)
{
  zdl::DlSystem::IUserBufferFactory& ubFactory = zdl::SNPE::SNPEFactory::getUserBufferFactory ();
  for (const char *name : strList) {
    auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes (name);
    const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims ();
    std::vector<size_t> strides (bufferShape.rank ());
    strides[strides.size () - 1] = sizeof (float);
    for (size_t i = strides.size () - 1; i > 0; --i) {
      if (bufferShape[i] == 0) {
        if (max_resizable_dim == 0) {
          throw std::runtime_error (std::string ("zero dim is detected in tensor ") + name + std::string (". User should provide MaxResizableDim with custom option"));
        }
        nns_logi ("zero dim (resizable dim) is detected in %zu-th dim of tensor %s. Set it as user given max number: %zu",
            i, name, max_resizable_dim);
        bufferShape[i] = max_resizable_dim;
      }
      strides[i - 1] = strides[i] * bufferShape[i];
    }

    size_t bufSize = sizeof (float);
    for (size_t i = 0; i < bufferShape.rank (); ++i) {
      bufSize *= bufferShape[i];
    }

    std::unique_ptr<zdl::DlSystem::UserBufferEncoding> userBufferEncoding;
    userBufferEncoding = std::unique_ptr<zdl::DlSystem::UserBufferEncodingFloat>(new zdl::DlSystem::UserBufferEncodingFloat ());

    user_input_buffers.push_back (ubFactory.createUserBuffer (NULL, bufSize, strides, userBufferEncoding.get ()));

    if (user_input_buffers.back () == nullptr) {
      throw std::runtime_error ("Error while creating user buffer.");
    }

    buffer_map.add (name, user_input_buffers.back ().get ());
  }
}

/**
 * @brief Method to set tensor properties with user_buffer.
 */
void
snpe_subplugin::setTensorProp (GstTensorsInfo &tensor_meta, const zdl::DlSystem::StringList strList, tensor_type data_type)
{
  unsigned int idx = 0;
  tensor_meta.num_tensors = strList.size ();

  for (const char *name : strList) {
    tensor_meta.info[idx].type = data_type;
    tensor_meta.info[idx].name = g_strdup (name);
    auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes (name);
    const zdl::DlSystem::TensorShape& bufferShape = (*bufferAttributesOpt)->getDims ();
    for (size_t j = 0; j < bufferShape.rank (); ++j) {
      if (bufferShape[bufferShape.rank () - j - 1] == 0) {
        if (max_resizable_dim == 0) {
          throw std::runtime_error (std::string ("zero dim is detected in tensor ") + name + std::string (". User should provide MaxResizableDim with custom option"));
        }
        bufferShape[bufferShape.rank () - j - 1] = max_resizable_dim;
      }
      tensor_meta.info[idx].dimension[j] = bufferShape[bufferShape.rank () - j - 1];
    }
    for (size_t j = bufferShape.rank (); j < NNS_TENSOR_RANK_LIMIT; ++j) {
      tensor_meta.info[idx].dimension[j] = 1;
    }
    idx++;
  }
}

snpe_subplugin *snpe_subplugin::registeredRepresentation = nullptr;

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
snpe_subplugin::init_filter_snpe (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<snpe_subplugin> ();
  nnstreamer_filter_set_custom_property_desc (name,
      "Runtime",
      "Designate hardware resource {'CPU' (default), 'GPU', 'DSP', 'NPU'}",
      "CPUFallback",
      "Set true to enable CPU fallback {'true' (default), 'false'}",
      "OutputLayer",
      "Layer names for the output, separated by ';'. E.g., 'layer0;layer1;layer2'",
      "InputType",
      "Set the data type of the input {'float32 (default)', 'uint8'}",
      "OutputType",
      "Set the data type of the output {'float32 (default)', 'uint8'}",
      "UserBuffer",
      "Use user supplied buffers for input/output tensors {'false (default)', 'true'}",
      "MaxResizableDim",
      "Max number for resizable dim (should be greater than 0)",
      NULL);
}

/** @brief Destruct the subplugin */
void
snpe_subplugin::fini_filter_snpe (void)
{
  assert (registeredRepresentation != nullptr);
  tensor_filter_subplugin::unregister_subplugin (registeredRepresentation);
}

#if defined(__ANDROID__)
/**
 * @brief Set additional environment (ADSP_LIBRARY_PATH) for snpe
 */
static gboolean
_snpe_set_env (JNIEnv *env, jobject context)
{
  gboolean snpe_failed = TRUE;
  jclass context_class = NULL;
  jmethodID get_application_info_method_id = NULL;
  jobject application_info_object = NULL;
  jclass application_info_object_class = NULL;
  jfieldID native_library_dir_field_id = NULL;
  jstring native_library_dir_path = NULL;

  const gchar *native_library_dir_path_str;
  gchar *new_path;

  g_return_val_if_fail (env != NULL, FALSE);
  g_return_val_if_fail (context != NULL, FALSE);

  context_class = env->GetObjectClass (context);
  if (!context_class) {
    nns_loge ("Failed to get context class.");
    goto done;
  }

  get_application_info_method_id = env->GetMethodID (context_class,
      "getApplicationInfo", "()Landroid/content/pm/ApplicationInfo;");
  if (!get_application_info_method_id) {
    nns_loge ("Failed to get method ID for `ApplicationInfo()`.");
    goto done;
  }

  application_info_object = env->CallObjectMethod (context, get_application_info_method_id);
  if (env->ExceptionCheck ()) {
    env->ExceptionDescribe ();
    env->ExceptionClear ();
    nns_loge ("Failed to call method `ApplicationInfo()`.");
    goto done;
  }

  application_info_object_class = env->GetObjectClass (application_info_object);
  if (!application_info_object_class) {
    nns_loge ("Failed to get `ApplicationInfo` object class");
    goto done;
  }

  native_library_dir_field_id = env->GetFieldID (
      application_info_object_class, "nativeLibraryDir", "Ljava/lang/String;");
  if (!native_library_dir_field_id) {
    nns_loge ("Failed to get field ID for `nativeLibraryDir`.");
    goto done;
  }

  native_library_dir_path = static_cast<jstring> (
      env->GetObjectField (application_info_object, native_library_dir_field_id));
  if (!native_library_dir_path) {
    nns_loge ("Failed to get field `nativeLibraryDir`.");
    goto done;
  }

  native_library_dir_path_str = env->GetStringUTFChars (native_library_dir_path, NULL);
  if (env->ExceptionCheck ()) {
    env->ExceptionDescribe ();
    env->ExceptionClear ();
    nns_loge ("Failed to get string `nativeLibraryDir`");
    goto done;
  }

  new_path = g_strconcat (native_library_dir_path_str,
      ";/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp", NULL);

  /**
   *  See https://developer.qualcomm.com/docs/snpe/dsp_runtime.html for details
   */
  nns_logi ("Set env ADSP_LIBRARY_PATH for snpe DSP/AIP runtime: %s", new_path);
  g_setenv ("ADSP_LIBRARY_PATH", new_path, TRUE);

  g_free (new_path);
  env->ReleaseStringUTFChars (native_library_dir_path, native_library_dir_path_str);

  snpe_failed = FALSE;

done:

  if (native_library_dir_path) {
    env->DeleteLocalRef (native_library_dir_path);
  }

  if (application_info_object_class) {
    env->DeleteLocalRef (application_info_object_class);
  }

  if (application_info_object) {
    env->DeleteLocalRef (application_info_object);
  }

  if (context_class) {
    env->DeleteLocalRef (context_class);
  }

  return !(snpe_failed);
}

/**
 * @brief Register the sub-plugin for SNPE in Android.
 */
void
init_filter_snpe (JNIEnv *env, jobject context)
{
  if (nnstreamer_filter_find ("snap")) {
    nns_loge ("Cannot use SNPE and SNAP both. Won't register this SNPE subplugin.");
    return;
  }

  if (!_snpe_set_env (env, context)) {
    nns_loge ("Failed to set extra env");
    return;
  }

  snpe_subplugin::init_filter_snpe ();
}
#else
/**
 * @brief Register the sub-plugin for SNPE.
 */
void
init_filter_snpe ()
{
  snpe_subplugin::init_filter_snpe ();
}
#endif

/**
 * @brief Destruct the sub-plugin for SNPE.
 */
void
fini_filter_snpe ()
{
  snpe_subplugin::fini_filter_snpe ();
}

} /* namespace nnstreamer::tensor_filter_snpe */
} /* namespace nnstreamer */

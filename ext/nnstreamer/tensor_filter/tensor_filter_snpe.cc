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
#include <tensor_common.h>

#include <DlContainer/IDlContainer.hpp>
#include <DlSystem/ITensorFactory.hpp>
#include <DlSystem/RuntimeList.hpp>
#include <DlSystem/TensorMap.hpp>
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
void init_filter_snpe (void) __attribute__ ((constructor));
#endif
void fini_filter_snpe (void) __attribute__ ((destructor));
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
  std::vector<std::string> output_layer_names;

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
  bool set_output_layer_names (const GstTensorsInfo *info);
  static void setTensorProp (GstTensorsInfo &tensor_meta, zdl::DlSystem::TensorMap &tensor_map);
  static const char *runtimeToString (zdl::DlSystem::Runtime_t runtime);

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
      container (nullptr), snpe (nullptr)
{
  inputInfo.num_tensors = 0;
  outputInfo.num_tensors = 0;
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
  if (empty_model)
    return;

  if (container) {
    container = nullptr;
  }

  if (snpe) {
    snpe.reset ();
    snpe = nullptr;
  }

  if (model_path)
    delete model_path;

  runtime_list.clear ();
  input_tensors.clear ();
  input_tensor_map.clear ();
  output_tensor_map.clear ();

  model_path = nullptr;
  inputInfo.num_tensors = 0;
  outputInfo.num_tensors = 0;
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
snpe_subplugin::set_output_layer_names (const GstTensorsInfo *info)
{
  for (unsigned int i = 0; i < info->num_tensors; ++i) {
    if (info->info[i].name == nullptr) {
      /* failed */
      nns_loge ("Given layer name with index %u is invalid.", i);
      output_layer_names.clear ();
      return false;
    }
    nns_logd ("Add output layer name of %s", info->info[i].name);
    output_layer_names.emplace_back (std::string (info->info[i].name));
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
      } else if (g_ascii_strcasecmp (option[0], "OutputLayer") == 0) {
        gchar **names = g_strsplit (option[1], ";", -1);
        guint num_names = g_strv_length (names);
        for (guint i = 0; i < num_names; ++i) {
          if (g_strcmp0 (names[i], "") == 0) {
            nns_loge ("Given layer name with index %u is invalid.", i);
            output_layer_names.clear ();
            invalid_option = true;
            break;
          }
          nns_logd ("Add output layer name of %s", names[i]);
          output_layer_names.emplace_back (std::string (names[i]));
        }
        g_strfreev (names);
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

  if (!set_output_layer_names (&prop->output_meta)) {
    nns_loge ("Failed to set output layer names");
    return;
  }

  if (!configure_option (prop)) {
    throw std::invalid_argument ("Failed to configure SNPE option.");
    return;
  }

  if (!empty_model) {
    /* Already opend */

    if (!prop->model_files[0] || prop->model_files[0][0] == '\0') {
      std::cerr << "Model path is not given." << std::endl;
      throw std::invalid_argument ("Model path is not given.");
    }

    cleanup ();
  }

  assert (model_path == nullptr);

  model_path = g_strdup (prop->model_files[0]);

  container = zdl::DlContainer::IDlContainer::open (model_path);

  zdl::DlSystem::StringList _output_layer_names;
  for (size_t i = 0; i < output_layer_names.size (); ++i) {
    _output_layer_names.append (output_layer_names.at(i).c_str ());
  }

  zdl::SNPE::SNPEBuilder snpe_builder (container.get());
  snpe_builder.setOutputLayers (_output_layer_names);
  snpe_builder.setUseUserSuppliedBuffers (false);
  snpe_builder.setInitCacheMode (false);
  snpe_builder.setRuntimeProcessorOrder (runtime_list);
  snpe_builder.setCPUFallbackMode (use_cpu_fallback);

  snpe = snpe_builder.build ();
  if (snpe == nullptr) {
    nns_loge ("fail to build snpe");
  }

  const zdl::DlSystem::Optional<zdl::DlSystem::StringList> &strList_opt
      = snpe->getInputTensorNames ();

  assert (strList_opt);

  const zdl::DlSystem::StringList &strList = *strList_opt;

  for (size_t i = 0; i < strList.size (); ++i) {
    const zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> &inputDims_opt
        = snpe->getInputDimensions (strList.at (i));
    const zdl::DlSystem::TensorShape &input_shape = *inputDims_opt;

    input_tensors.emplace_back (
        zdl::SNPE::SNPEFactory::getTensorFactory ().createTensor (input_shape));
    input_tensor_map.add (strList.at (i), input_tensors[i].get ());
  }

  /* do execution for get info of output tensors */
  snpe->execute (input_tensor_map, output_tensor_map);

  setTensorProp (inputInfo, input_tensor_map);
  setTensorProp (outputInfo, output_tensor_map);

  output_tensor_map.clear ();

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

  /* Configure inputs */
  for (unsigned int i = 0; i < inputInfo.num_tensors; ++i) {
    float *finput = (float *)input[i].data;
    size_t fsize = input_tensors[i].get ()->getSize ();
    std::copy (finput, finput + fsize, input_tensors[i].get ()->begin ());
  }

  output_tensor_map.clear ();
  snpe->execute (input_tensor_map, output_tensor_map);

  for (unsigned int i = 0; i < outputInfo.num_tensors; ++i) {
    zdl::DlSystem::ITensor *output_tensor
        = output_tensor_map.getTensor (output_tensor_map.getTensorNames ().at (i));
    float *foutput = (float *)output[i].data;
    std::copy (output_tensor->cbegin (), output_tensor->cend (), foutput);
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
  return -ENOENT;
}

/**
 * @brief Method to set tensor properties.
 */
void
snpe_subplugin::setTensorProp (GstTensorsInfo &tensor_meta, zdl::DlSystem::TensorMap &tensor_map)
{
  tensor_meta.num_tensors = tensor_map.size ();
  for (unsigned int i = 0; i < tensor_map.size (); ++i) {
    tensor_meta.info[i].name = g_strdup (tensor_map.getTensorNames ().at (i));
    tensor_meta.info[i].type = _NNS_FLOAT32;

    unsigned int rank
        = tensor_map.getTensor (tensor_meta.info[i].name)->getShape ().rank ();
    for (unsigned int j = 0; j < rank; ++j) {
      tensor_meta.info[i].dimension[j]
          = tensor_map.getTensor (tensor_meta.info[i].name)->getShape ()[rank - j - 1];
    }
    for (unsigned int j = rank; j < NNS_TENSOR_RANK_LIMIT; ++j) {
      tensor_meta.info[i].dimension[j] = 1;
    }
  }
}

snpe_subplugin *snpe_subplugin::registeredRepresentation = nullptr;

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
snpe_subplugin::init_filter_snpe (void)
{
  registeredRepresentation
      = tensor_filter_subplugin::register_subplugin<snpe_subplugin> ();
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

  /* See https://developer.qualcomm.com/docs/snpe/dsp_runtime.html for details
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

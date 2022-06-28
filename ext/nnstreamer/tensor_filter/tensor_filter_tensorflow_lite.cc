/**
 * GStreamer Tensor_Filter, Tensorflow-Lite Module
 * Copyright (C) 2018 Samsung Electronics Co., Ltd. All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file   tensor_filter_tensorflow_lite.cc
 * @date   7 May 2018
 * @brief  Tensorflow-lite module for tensor_filter gstreamer plugin
 * @see    http://github.com/nnstreamer/nnstreamer
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (tensorflow-lite, tensorflow2-lite)
 * for tensor_filter. The meson build system generates two .so files
 * (e.g., TF-Lite and TF2-Lite) from this source code.
 */

#include <algorithm>
#include <limits.h>
#include <thread>
#include <unistd.h>

#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api_util.h>
#define NO_ANONYMOUS_NESTED_STRUCT
#include <nnstreamer_plugin_api_filter.h>
#undef NO_ANONYMOUS_NESTED_STRUCT
#include <nnstreamer_conf.h>
#include <nnstreamer_util.h>

#if TFLITE_VERSION_MAJOR >= 2 || TFLITE_VERSION_MINOR >= 13
#  if USE_TENSORFLOW2_HEADER_PATH
#    include <tensorflow2/lite/kernels/register.h>
#    include <tensorflow2/lite/model.h>
#  else
#    include <tensorflow/lite/kernels/register.h>
#    include <tensorflow/lite/model.h>
#  endif
#else
#  include <tensorflow/contrib/lite/kernels/register.h>
#  include <tensorflow/contrib/lite/model.h>
#endif

/** control delegate headers */
#ifdef TFLITE_XNNPACK_DELEGATE_SUPPORTED
#  if USE_TENSORFLOW2_HEADER_PATH
#    include <tensorflow2/lite/delegates/xnnpack/xnnpack_delegate.h>
#  else
#    include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>
#  endif
#endif

#ifdef TFLITE_GPU_DELEGATE_SUPPORTED
#  if USE_TENSORFLOW2_HEADER_PATH
#    include <tensorflow2/lite/delegates/gpu/delegate.h>
#  else
#    include <tensorflow/lite/delegates/gpu/delegate.h>
#  endif
#endif

#ifdef TFLITE_NNAPI_DELEGATE_SUPPORTED
#  if USE_TENSORFLOW2_HEADER_PATH
#    include <tensorflow2/lite/delegates/nnapi/nnapi_delegate.h>
#  else
#    include <tensorflow/lite/delegates/nnapi/nnapi_delegate.h>
#  endif
#endif

#ifdef TFLITE_EXTERNAL_DELEGATE_SUPPORTED
#  if USE_TENSORFLOW2_HEADER_PATH
#    include <tensorflow2/lite/delegates/external/external_delegate.h>
#  else
#    include <tensorflow/lite/delegates/external/external_delegate.h>
#  endif
#endif

#if !defined(TFLITE_SUBPLUGIN_NAME)
#warning "The sub-plugin name for tensorflow-lite is not defined."
#define TFLITE_SUBPLUGIN_NAME "tensorflow-lite"
#endif

/**
 * @brief prevent usage by TFLite of default delegates that may not be supported
 */
#if TFLITE_VERSION_MAJOR >= 2 && TFLITE_VERSION_MINOR >= 4
#define TFLITE_RESOLVER_WITHOUT_DEFAULT_DELEGATES
#endif

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief Possible tensorflow-lite delegates.
 */
typedef enum {
  TFLITE_DELEGATE_NONE = 0,
  TFLITE_DELEGATE_GPU,
  TFLITE_DELEGATE_NNAPI,
  TFLITE_DELEGATE_XNNPACK,
  TFLITE_DELEGATE_EXTERNAL,

  TFLITE_DELEGATE_MAX
} tflite_delegate_e;

/**
 * @brief Option to open tf-lite model.
 */
typedef struct {
  const gchar *model_file; /**< path to tensorflow-lite model file */
  const gchar *accelerators; /**< accelerators set for this subplugin */
  tflite_delegate_e delegate; /**< tensorflow-lite delegate */
  gint num_threads; /**< the number of threads */
  const gchar *ext_delegate_path; /**< path to external delegate lib */
  GHashTable *ext_delegate_kv_table; /**< external delegate key values options */
} tflite_option_s;

/**
 * @brief Possible accelerators.
 */
static const gchar *tflite_accl_support[] = { ACCL_CPU_NEON_STR,
  ACCL_CPU_SIMD_STR, ACCL_CPU_STR, ACCL_GPU_STR, ACCL_NPU_STR, NULL };

#if defined(__x86_64__) || defined(__aarch64__) || defined(__arm__)
static const gchar *tflite_accl_auto = ACCL_CPU_SIMD_STR;
#else
static const gchar *tflite_accl_auto = ACCL_CPU_STR;
#endif
static const gchar *tflite_accl_default = ACCL_CPU_STR;

static GstTensorFilterFrameworkStatistics tflite_internal_stats = {
  .total_invoke_num = 0,
  .total_invoke_latency = 0,
  .total_overhead_latency = 0,
};

/**
 * @brief Wrapper class for TFLite Interpreter to support model switching
 */
class TFLiteInterpreter
{
  public:
  TFLiteInterpreter ();
  ~TFLiteInterpreter ();

  int invoke (const GstTensorMemory *input, GstTensorMemory *output);
  int loadModel (int num_threads, tflite_delegate_e delegate);

  int setInputTensorProp ();
  int setOutputTensorProp ();
  int setInputTensorsInfo (const GstTensorsInfo *info);

  void setModelPath (const char *model_path);
  void setExtDelegate (const char *lib_path, GHashTable *key_val);
  void getExtDelegate (const char **lib_path, GHashTable **key_val);
  /** @brief get current model path */
  const char *getModelPath ()
  {
    return model_path;
  }

  /** @brief return input tensor meta */
  const GstTensorsInfo *getInputTensorsInfo ()
  {
    return &inputTensorMeta;
  }
  /** @brief return output tensor meta */
  const GstTensorsInfo *getOutputTensorsInfo ()
  {
    return &outputTensorMeta;
  }

  /** @brief lock this interpreter */
  void lock ()
  {
    g_mutex_lock (&mutex);
  }
  /** @brief unlock this interpreter */
  void unlock ()
  {
    g_mutex_unlock (&mutex);
  }
  /** @brief cache input and output tensor ptr before invoke */
  int cacheInOutTensorPtr ();

  /** @brief set delegate for the tflite interpreter */
  void setDelegate (TfLiteDelegate *delegate, void (*deleter) (TfLiteDelegate *))
  {
    delegate_ptr = tflite::Interpreter::TfLiteDelegatePtr (delegate, deleter);
  }

  /** @brief get delegate for the tflite interpreter */
  TfLiteDelegate *getDelegate ()
  {
    return delegate_ptr.get ();
  }

  private:
  GMutex mutex;
  char *model_path;
  bool is_cached_after_first_invoke; /**< To cache again after first invoke */
  bool is_xnnpack_delegated; /**< To check if XNNPACK delegate is used */
  char *ext_delegate_path; /**< path to external delegate lib */
  GHashTable *ext_delegate_kv_table; /**< external delegate key values options */

  std::unique_ptr<tflite::Interpreter> interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model;

  GstTensorsInfo inputTensorMeta; /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta; /**< The tensor info of output tensors */
  std::vector<TfLiteTensor *> inputTensorPtr;
  std::vector<TfLiteTensor *> outputTensorPtr;

  tensor_type getTensorType (TfLiteType tfType);
  int getTensorDim (int tensor_idx, tensor_dim dim);
  int setTensorProp (const std::vector<int> &tensor_idx_list, GstTensorsInfo *tensorMeta);

  tflite::Interpreter::TfLiteDelegatePtr delegate_ptr; /**< single delegate supported */
};

/**
 * @brief	ring cache structure
 */
class TFLiteCore
{
  public:
  TFLiteCore (const GstTensorFilterProperties *prop);
  ~TFLiteCore ();
  int init (tflite_option_s *option);
  int loadModel ();
  gboolean compareModelPath (const char *model_path);
  int setInputTensorProp ();
  int setOutputTensorProp ();
  int getInputTensorDim (GstTensorsInfo *info);
  int getOutputTensorDim (GstTensorsInfo *info);
  int setInputTensorDim (const GstTensorsInfo *info);
  int reloadModel (const char *model_path);
  int invoke (const GstTensorMemory *input, GstTensorMemory *output);
  /** @brief cache input and output tensor ptr before invoke */
  int cacheInOutTensorPtr ();
  /** @brief callback method to delete interpreter for shared model */
  friend void free_interpreter (void *instance);
  /** @brief callback method to replace interpreter for shared model */
  friend void replace_interpreter (void *instance, void *interperter);

  private:
  int num_threads;
  accl_hw accelerator;
  tflite_delegate_e delegate;

  TFLiteInterpreter *interpreter;
  TFLiteInterpreter *interpreter_sub;

  gchar *shared_tensor_filter_key;
  gboolean checkSharedInterpreter (const GstTensorFilterProperties *prop);
  int reloadInterpreter (TFLiteInterpreter * new_interpreter);
  void setAccelerator (const char *accelerators, tflite_delegate_e d);
};

extern "C" {
void init_filter_tflite (void) __attribute__ ((constructor));
void fini_filter_tflite (void) __attribute__ ((destructor));
}

G_LOCK_DEFINE_STATIC (slock);

/**
 * @brief TFLiteInterpreter constructor
 */
TFLiteInterpreter::TFLiteInterpreter ()
: delegate_ptr (nullptr, [] (TfLiteDelegate *) {})
{
  interpreter = nullptr;
  model = nullptr;
  model_path = nullptr;
  ext_delegate_path = nullptr;
  ext_delegate_kv_table = nullptr;

  g_mutex_init (&mutex);

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);

  is_cached_after_first_invoke = false;
  is_xnnpack_delegated = false;
}

/**
 * @brief TFLiteInterpreter desctructor
 */
TFLiteInterpreter::~TFLiteInterpreter ()
{
  g_mutex_clear (&mutex);
  g_free (model_path);
  g_free (ext_delegate_path);
  if (ext_delegate_kv_table)
    g_hash_table_unref(ext_delegate_kv_table);

  gst_tensors_info_free (&inputTensorMeta);
  gst_tensors_info_free (&outputTensorMeta);
}

/**
 * @brief Internal implementation of TFLiteCore's invoke()
 */
int
TFLiteInterpreter::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  int64_t start_time, stop_time;
  TfLiteTensor *tensor_ptr;
  TfLiteStatus status;

  start_time = g_get_monotonic_time ();

  /**
   * XNNPACK Delegate uses fixed buffer address for input/output tensors.
   * Therefore tensor data is to be manually copied from/to input/output GStreamer
   * buffers memory whose address changes at every round.
   */
  if (is_xnnpack_delegated) {
    for (unsigned int i = 0; i < inputTensorMeta.num_tensors; ++i) {
      tensor_ptr = inputTensorPtr[i];
      g_assert(tensor_ptr->bytes == input[i].size);
      memcpy (tensor_ptr->data.raw, input[i].data, input[i].size);
    }
  } else {
    for (unsigned int i = 0; i < inputTensorMeta.num_tensors; ++i) {
      tensor_ptr = inputTensorPtr[i];
      tensor_ptr->data.raw = (char *) input[i].data;
    }

    for (unsigned int i = 0; i < outputTensorMeta.num_tensors; ++i) {
      tensor_ptr = outputTensorPtr[i];
      tensor_ptr->data.raw = (char *) output[i].data;
    }
  }

  stop_time = g_get_monotonic_time ();

  tflite_internal_stats.total_overhead_latency += stop_time - start_time;

  start_time = g_get_monotonic_time ();
  status = interpreter->Invoke ();

  if (is_xnnpack_delegated) {
    for (unsigned int i = 0; i < outputTensorMeta.num_tensors; ++i) {
      tensor_ptr = outputTensorPtr[i];
      g_assert(tensor_ptr->bytes == output[i].size);
      memcpy (output[i].data, tensor_ptr->data.raw, output[i].size);
    }
  }

  stop_time = g_get_monotonic_time ();

  tflite_internal_stats.total_invoke_latency += stop_time - start_time;
  tflite_internal_stats.total_invoke_num += 1;

#if (DBG)
  ml_logi ("Invoke() is finished: %" G_GINT64_FORMAT "ms, model path: %s",
      (stop_time - start_time) / 1000, getModelPath ());
  ml_logi ("%" G_GINT64_FORMAT " invoke average %" G_GINT64_FORMAT
           ", total overhead %" G_GINT64_FORMAT,
      tflite_internal_stats.total_invoke_num,
      (tflite_internal_stats.total_invoke_latency / tflite_internal_stats.total_invoke_num),
      tflite_internal_stats.total_overhead_latency);
#endif

  if (status != kTfLiteOk) {
    ml_loge ("Failed to invoke");
    return -1;
  }

  if (!is_cached_after_first_invoke) {
    if (cacheInOutTensorPtr () == 0) {
      is_cached_after_first_invoke = true;
    } else {
      ml_logw ("Failed to cache tensor memory ptr");
    }
  }

  return 0;
}

/**
 * @brief Internal implementation of TFLiteCore's loadModel()
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteInterpreter::loadModel (int num_threads, tflite_delegate_e delegate_e)
{
  TfLiteDelegate *delegate;
#if (DBG)
  gint64 start_time, stop_time;
  start_time = g_get_monotonic_time ();
#endif

  model = tflite::FlatBufferModel::BuildFromFile (model_path);
  if (!model) {
    ml_loge ("Failed to mmap model\n");
    return -1;
  }

  /**
   * If got any trouble at model, active below code. It'll be help to analyze.
   * model->error_reporter ();
   */

  interpreter = nullptr;

#ifdef TFLITE_RESOLVER_WITHOUT_DEFAULT_DELEGATES
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
#else
  tflite::ops::builtin::BuiltinOpResolver resolver;
#endif
  tflite::InterpreterBuilder (*model, resolver) (&interpreter);
  if (!interpreter) {
    ml_loge ("Failed to construct interpreter\n");
    return -2;
  }

  if (num_threads > 0) {
    int n = static_cast<int> (std::thread::hardware_concurrency ());

    num_threads = MIN (n, num_threads);
    ml_logi ("Set the number of threads (%d)", num_threads);
    interpreter->SetNumThreads (num_threads);
  }

  /** set delegate after the accelerator prop */
  switch (delegate_e) {
    case TFLITE_DELEGATE_XNNPACK:
    {
#if TFLITE_XNNPACK_DELEGATE_SUPPORTED
      /* set xnnpack delegate */
      TfLiteXNNPackDelegateOptions xnnpack_options =
          TfLiteXNNPackDelegateOptionsDefault();
      xnnpack_options.num_threads = (num_threads > 1) ? num_threads : 0;

      is_xnnpack_delegated = true;
      ml_logw ("Input/output tensors should be memcpy-ed rather than explicitly assigning its ptr when XNNPACK Delegate is used.");
      ml_logw ("This could cause performance degradation if sizes of input/output tensors are large");

      delegate = TfLiteXNNPackDelegateCreate (&xnnpack_options);
      void (* deleter) (TfLiteDelegate *) =
              [] (TfLiteDelegate *delegate_) {
                  TfLiteXNNPackDelegateDelete (delegate_);
              };

      setDelegate (delegate, deleter);
#else
      ml_logw ("NNStreamer was built without XNNPACK delegate. Given delegate option XNNPACK is ignored.");
#endif
      break;
    }
    case TFLITE_DELEGATE_GPU:
    {
#if TFLITE_GPU_DELEGATE_SUPPORTED
      /* set gpu delegate when accelerator set to GPU */
      TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default ();
      options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;
      options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT;

      /**
       * NNStreamer filter for TFLite2 GPU delegate only supports OpenCL backend
       * since GLES v3.1 backend has a constraint that
       * Invoke() must be called from the same EGLContext.
       */
      options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
      options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
      options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
      options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;

      delegate = TfLiteGpuDelegateV2Create (&options);
      void (* deleter) (TfLiteDelegate *) =
              [] (TfLiteDelegate *delegate_) {
                  TfLiteGpuDelegateV2Delete (delegate_);
              };

      setDelegate (delegate, deleter);
#else
      ml_logw ("NNStreamer was built without GPU delegate. Given delegate option GPU is ignored.");
#endif
      break;
    }
    case TFLITE_DELEGATE_NNAPI:
    {
#if TFLITE_NNAPI_DELEGATE_SUPPORTED
      /* set nnapi delegate when accelerator set to auto (cpu.neon in Android) or NPU */
      delegate = new tflite::StatefulNnApiDelegate ();
      void (* deleter) (TfLiteDelegate *) =
              [] (TfLiteDelegate *delegate_) {
                  delete reinterpret_cast<tflite::StatefulNnApiDelegate *> (delegate_);
              };

      setDelegate (delegate, deleter);
#else
      ml_logw ("NNStreamer was built without NNAPI delegate. Given delegate option NNAPI is ignored.");
#endif
      break;
    }
    case TFLITE_DELEGATE_EXTERNAL:
    {
#ifdef TFLITE_EXTERNAL_DELEGATE_SUPPORTED
      TfLiteExternalDelegateOptions options;

      options = TfLiteExternalDelegateOptionsDefault (ext_delegate_path);

      /* Add optional key values to delegate configuration */
      if (ext_delegate_kv_table) {
        GHashTable *table = ext_delegate_kv_table;
        GHashTableIter iter;
        gchar *key, *value;

        g_hash_table_iter_init (&iter, table);
        while (g_hash_table_iter_next (&iter, (gpointer *) &key, (gpointer *) &value))
           options.insert (&options, key, value);
      }

      delegate = TfLiteExternalDelegateCreate (&options);
      void (* deleter) (TfLiteDelegate *) =
              [] (TfLiteDelegate *delegate_) {
                  TfLiteExternalDelegateDelete (delegate_);
              };

      setDelegate (delegate, deleter);
#else
      ml_logw ("NNStreamer was built without external delegate. Given delegate option external is ignored.");
#endif
      break;
    }
    default:
      break;
  }

  delegate = getDelegate ();
  if (delegate != nullptr) {
    if (interpreter->ModifyGraphWithDelegate (delegate) != kTfLiteOk) {
      ml_loge ("Failed to apply delegate\n");
      return -2;
    }
  }

  if (interpreter->AllocateTensors () != kTfLiteOk) {
    ml_loge ("Failed to allocate tensors\n");
    return -2;
  }

#if (DBG)
  stop_time = g_get_monotonic_time ();
  ml_logi ("Model is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
  return 0;
}

/**
 * @brief	return the data type of the tensor
 * @param tfType	: the defined type of Tensorflow Lite
 * @return the enum of defined _NNS_TYPE
 */
tensor_type
TFLiteInterpreter::getTensorType (TfLiteType tfType)
{
  switch (tfType) {
  case kTfLiteFloat32:
    return _NNS_FLOAT32;
  case kTfLiteUInt8:
    return _NNS_UINT8;
  case kTfLiteInt32:
    return _NNS_INT32;
  case kTfLiteBool:
#ifdef TFLITE_INT8
  case kTfLiteInt8:
#endif
    return _NNS_INT8;
#ifdef TFLITE_INT16
  case kTfLiteInt16:
    return _NNS_INT16;
#endif
  case kTfLiteInt64:
    return _NNS_INT64;
  case kTfLiteString:
#ifdef TFLITE_COMPLEX64
  case kTfLiteComplex64:
#endif
#ifdef TFLITE_FLOAT16
  case kTfLiteFloat16:
#endif
  default:
    ml_loge ("Not supported Tensorflow Data Type: [%d].", tfType);
    /** @todo Support other types */
    break;
  }

  return _NNS_END;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param tensor_idx	: the real index of model of the tensor
 * @param[out] dim	: the array of the tensor
 * @return 0 if OK. non-zero if error.
 * @note assume that the interpreter lock was already held.
 */
int
TFLiteInterpreter::getTensorDim (int tensor_idx, tensor_dim dim)
{
  TfLiteIntArray *tensor_dims = interpreter->tensor (tensor_idx)->dims;
  int len = tensor_dims->size;
  if (len > NNS_TENSOR_RANK_LIMIT)
    return -EPERM;

  /* the order of dimension is reversed at CAPS negotiation */
  std::reverse_copy (tensor_dims->data, tensor_dims->data + len, dim);

  /* fill the remnants with 1 */
  for (int i = len; i < NNS_TENSOR_RANK_LIMIT; ++i) {
    dim[i] = 1;
  }

  return 0;
}

/**
 * @brief extract and store the information of given tensor list
 * @param tensor_idx_list list of index of tensors in tflite interpreter
 * @param[out] tensorMeta tensors to set the info into
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteInterpreter::setTensorProp (
    const std::vector<int> &tensor_idx_list, GstTensorsInfo *tensorMeta)
{
  tensorMeta->num_tensors = tensor_idx_list.size ();

  for (unsigned int i = 0; i < tensorMeta->num_tensors; ++i) {
    int idx = tensor_idx_list[i];

    if (getTensorDim (idx, tensorMeta->info[i].dimension)) {
      ml_loge ("failed to get the dimension of input tensors");
      return -1;
    }
    tensorMeta->info[i].type = getTensorType (interpreter->tensor (idx)->type);
    tensorMeta->info[i].name = g_strdup (interpreter->tensor (idx)->name);

#if (DBG)
    gchar *dim_str = gst_tensor_get_dimension_string (tensorMeta->info[i].dimension);
    ml_logi ("tensorMeta[%d] >> name[%s], type[%d], dim[%s]", i,
        tensorMeta->info[i].name, tensorMeta->info[i].type, dim_str);
    g_free (dim_str);
#endif
  }
  return 0;
}

/**
 * @brief extract and store the information of input tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteInterpreter::setInputTensorProp ()
{
  return setTensorProp (interpreter->inputs (), &inputTensorMeta);
}

/**
 * @brief extract and store the information of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteInterpreter::setOutputTensorProp ()
{
  return setTensorProp (interpreter->outputs (), &outputTensorMeta);
}

/**
 * @brief set the Dimension for Input Tensor.
 * @param info Structure for input tensor info.
 * @return 0 if OK. non-zero if error.
 * @note rank can be changed dependent on the model
 */
int
TFLiteInterpreter::setInputTensorsInfo (const GstTensorsInfo *info)
{
  TfLiteStatus status = kTfLiteOk;
  const std::vector<int> &input_idx_list = interpreter->inputs ();

  /** Cannot change the number of inputs */
  if (info->num_tensors != input_idx_list.size ())
    return -EINVAL;

  for (unsigned int tensor_idx = 0; tensor_idx < info->num_tensors; ++tensor_idx) {
    tensor_type tf_type;
    const GstTensorInfo *tensor_info;
    int input_rank;

    tensor_info = &info->info[tensor_idx];

    /** cannot change the type of input */
    tf_type = getTensorType (interpreter->tensor (input_idx_list[tensor_idx])->type);
    if (tf_type != tensor_info->type)
      return -EINVAL;

    /**
     * Given that the rank intended by the user cannot be exactly determined,
     * iterate over all possible ranks starting from MAX rank to the actual rank
     * of the dimension array. In case of none of these ranks work, return error
     */
    input_rank = gst_tensor_info_get_rank (tensor_info);
    for (int rank = NNS_TENSOR_RANK_LIMIT; rank >= input_rank; rank--) {
      std::vector<int> dims (rank);
      /* the order of dimension is reversed at CAPS negotiation */
      for (int idx = 0; idx < rank; ++idx) {
        /** check overflow when storing uint32_t in int container */
        if (tensor_info->dimension[rank - idx - 1] > INT_MAX)
          return -ERANGE;
        dims[idx] = tensor_info->dimension[rank - idx - 1];
      }

      status = interpreter->ResizeInputTensor (input_idx_list[tensor_idx], dims);
      if (status != kTfLiteOk)
        continue;

      break;
    }

    /** return error when none of the ranks worked */
    if (status != kTfLiteOk)
      return -EPERM;
  }

  status = interpreter->AllocateTensors ();
  if (status != kTfLiteOk)
    return -EPERM;

  return 0;
}

/**
 * @brief update the model path
 */
void
TFLiteInterpreter::setModelPath (const char *_model_path)
{
  if (_model_path) {
    g_free (model_path);
    model_path = g_strdup (_model_path);
  }
}

/**
 * @brief update external delegate library path and options
 */
void
TFLiteInterpreter::setExtDelegate (const char *lib_path, GHashTable *key_val)
{
  g_free (ext_delegate_path);
  if (lib_path)
    ext_delegate_path = g_strdup (lib_path);
  else
    ext_delegate_path = nullptr;

  if (ext_delegate_kv_table)
    g_hash_table_unref (ext_delegate_kv_table);

  if (key_val) {
    g_hash_table_ref (key_val);
    ext_delegate_kv_table = key_val;
  } else
    ext_delegate_kv_table = nullptr;
}

/**
 * @brief get external delegate library path and options
 */
void
TFLiteInterpreter::getExtDelegate (const char **lib_path, GHashTable **key_val)
{
  *lib_path = ext_delegate_path;
  *key_val = ext_delegate_kv_table;
}

/**
 * @brief cache input and output tensor ptr before invoke
 * @return 0 on success. -errno on failure.
 */
int
TFLiteInterpreter::cacheInOutTensorPtr ()
{
  int tensor_idx;
  TfLiteTensor *tensor_ptr;

  inputTensorPtr.clear ();
  inputTensorPtr.reserve (inputTensorMeta.num_tensors);
  for (unsigned int i = 0; i < inputTensorMeta.num_tensors; ++i) {
    tensor_idx = interpreter->inputs ()[i];
    tensor_ptr = interpreter->tensor (tensor_idx);

    if (tensor_ptr->bytes != gst_tensor_info_get_size (&inputTensorMeta.info[i]))
      goto fail_exit;

    inputTensorPtr.push_back (tensor_ptr);
  }

  outputTensorPtr.clear ();
  outputTensorPtr.reserve (outputTensorMeta.num_tensors);
  for (unsigned int i = 0; i < outputTensorMeta.num_tensors; ++i) {
    tensor_idx = interpreter->outputs ()[i];
    tensor_ptr = interpreter->tensor (tensor_idx);

    if (tensor_ptr->bytes != gst_tensor_info_get_size (&outputTensorMeta.info[i]))
      goto fail_exit;

    outputTensorPtr.push_back (tensor_ptr);
  }

  return 0;

fail_exit:
  inputTensorPtr.clear ();
  outputTensorPtr.clear ();
  return -EINVAL;
}

/**
 * @brief	TFLiteCore constructor
 */
TFLiteCore::TFLiteCore (const GstTensorFilterProperties *prop)
{
  num_threads = -1;
  accelerator = ACCL_NONE;
  delegate = TFLITE_DELEGATE_NONE;
  interpreter_sub = nullptr;
  shared_tensor_filter_key = NULL;

  if (prop->shared_tensor_filter_key) {
    shared_tensor_filter_key =
        g_strdup (prop->shared_tensor_filter_key);
    if (!checkSharedInterpreter (prop))
      interpreter = new TFLiteInterpreter ();
  }
  else
    interpreter = new TFLiteInterpreter ();
}

/**
 * @brief callback method to destroy interpreter for shared model
 * @param interpreter TFLiteInterpreter*
 */
void free_interpreter (void * interpreter) {
  TFLiteInterpreter * self = reinterpret_cast <TFLiteInterpreter *> (interpreter);
  delete self;
}

/**
 * @brief	TFLiteCore destructor
 */
TFLiteCore::~TFLiteCore ()
{
  if (shared_tensor_filter_key) {
    G_LOCK (slock);
    if (!nnstreamer_filter_shared_model_remove (this, shared_tensor_filter_key, free_interpreter)) {
      nns_loge ("failed to remove shared model");
    }
    G_UNLOCK (slock);
    g_free (shared_tensor_filter_key);
  }
  else {
    delete interpreter;
  }
}

/**
 * @brief	check the shared interpreter
 * The shared model representation (interpreter) is allocated or shared.
 * If `shared_tensor_filter_key` is already existed, it will share the TFLiteInterpreter.
 * in the opposite case, the new TFLiteInterpreter will allocated and registered at the shared table.
 */
gboolean
TFLiteCore::checkSharedInterpreter (const GstTensorFilterProperties * prop)
{
  G_LOCK (slock);
  interpreter = (TFLiteInterpreter *) nnstreamer_filter_shared_model_get (this, shared_tensor_filter_key);

  if (!interpreter) {
    /* create new interpreter */
    TFLiteInterpreter *new_interpreter = new TFLiteInterpreter ();
    interpreter = (TFLiteInterpreter *) nnstreamer_filter_shared_model_insert_and_get (this, shared_tensor_filter_key, new_interpreter);
    if (!interpreter) {
      G_UNLOCK (slock);
      ml_loge ("Failed to insert the model representation!");
      g_free (shared_tensor_filter_key);
      shared_tensor_filter_key = NULL;
      delete new_interpreter;
      return FALSE;
    }
  }
  /* shared model exists */
  else if (g_strcmp0 (prop->model_files[0], interpreter->getModelPath ()) != 0) {
    ml_logw ("The model paths are not equal, models are not shared.");
    nnstreamer_filter_shared_model_remove (this, shared_tensor_filter_key, free_interpreter);
    G_UNLOCK (slock);
    g_free (shared_tensor_filter_key);
    shared_tensor_filter_key = NULL;
    return FALSE;
  }
  G_UNLOCK (slock);

  ml_logd ("The model representation is shared: key=[%s]", shared_tensor_filter_key);
  return TRUE;
}

/**
 * @brief	Set the accelerator for the tf engine
 */
void
TFLiteCore::setAccelerator (const char *accelerators, tflite_delegate_e d)
{
  accelerator = parse_accl_hw (
      accelerators, tflite_accl_support, tflite_accl_auto, tflite_accl_default);

  delegate = d;

  /* set possible tensorflow-lite delegate from accelerator */
  if (delegate == TFLITE_DELEGATE_NONE) {
    /** @todo update condition to set delegate from accl hw */
    switch (accelerator) {
      case ACCL_GPU:
        delegate = TFLITE_DELEGATE_GPU;
        break;
      default:
        break;
    }
  }

  ml_logd ("Set tensorflow-lite delegate %d", delegate);
  return;
}

/**
 * @brief	initialize the object with tflite model
 * @param	option options to initialize tf-lite model
 * @return 0 if OK. non-zero if error.
 *        -1 if the model is not loaded.
 *        -2 if the initialization of input tensor is failed.
 *        -3 if the initialization of output tensor is failed.
 *        -4 if the caching of input and output tensors failed.
 */
int
TFLiteCore::init (tflite_option_s *option)
{
  interpreter->setModelPath (option->model_file);
  interpreter->setExtDelegate (option->ext_delegate_path, option->ext_delegate_kv_table);
  num_threads = option->num_threads;
  int err;

  setAccelerator (option->accelerators, option->delegate);
  g_message ("accl = %s", get_accl_hw_str (accelerator));

  if ((err = loadModel ())) {
    ml_loge ("Failed to load model (TensorFlow-lite interpreter->loadModel() has returned %d. Please check if the model, '%s', is accessible and compatible with the given TensorFlow-lite instance. For example, this TensorFlow-lite's version might not support the given model.\n", err, option->model_file);
    return -1;
  }
  if (setInputTensorProp ()) {
    ml_loge ("Failed to initialize input tensor\n");
    return -2;
  }
  if (setOutputTensorProp ()) {
    ml_loge ("Failed to initialize output tensor\n");
    return -3;
  }
  if (cacheInOutTensorPtr ()) {
    ml_loge ("Failed to cache input and output tensors storage\n");
    return -4;
  }
  return 0;
}

/**
 * @brief	compare the model path
 * @return TRUE if tflite core has the same model path
 */
gboolean
TFLiteCore::compareModelPath (const char *model_path)
{
  gboolean is_same;

  interpreter->lock ();
  is_same = (g_strcmp0 (model_path, interpreter->getModelPath ()) == 0);
  interpreter->unlock ();

  return is_same;
}

/**
 * @brief	load the tflite model
 * @note	the model will be loaded
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::loadModel ()
{
  int err;

  interpreter->lock ();
  err = interpreter->loadModel (num_threads, delegate);
  interpreter->unlock ();

  return err;
}

/**
 * @brief extract and store the information of input tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::setInputTensorProp ()
{
  int err;

  interpreter->lock ();
  err = interpreter->setInputTensorProp ();
  interpreter->unlock ();

  return err;
}

/**
 * @brief extract and store the information of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::setOutputTensorProp ()
{
  int err;

  interpreter->lock ();
  err = interpreter->setOutputTensorProp ();
  interpreter->unlock ();

  return err;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getInputTensorDim (GstTensorsInfo *info)
{
  interpreter->lock ();
  gst_tensors_info_copy (info, interpreter->getInputTensorsInfo ());
  interpreter->unlock ();

  return 0;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getOutputTensorDim (GstTensorsInfo *info)
{
  interpreter->lock ();
  gst_tensors_info_copy (info, interpreter->getOutputTensorsInfo ());
  interpreter->unlock ();

  return 0;
}

/**
 * @brief set the Dimension for Input Tensor.
 * @param info Structure for input tensor info.
 * @return 0 if OK. non-zero if error.
 * @note rank can be changed dependent on the model
 */
int
TFLiteCore::setInputTensorDim (const GstTensorsInfo *info)
{
  int err;

  interpreter->lock ();
  err = interpreter->setInputTensorsInfo (info);
  interpreter->unlock ();

  return err;
}

/**
 * @brief Replace the interpreter, called by reloadModel
 *        Check input/output tensors have the same info
 * @param new_interpreter new interpreter to replace with
 * @return int 0 if ok, non-zero if error
 */
int
TFLiteCore::reloadInterpreter (TFLiteInterpreter * new_interpreter)
{
  TFLiteInterpreter *old_interpreter = interpreter;
  gboolean in_matched, out_matched;
  int ret = 0;

  old_interpreter->lock ();
  new_interpreter->lock ();

  in_matched = gst_tensors_info_is_equal (old_interpreter->getInputTensorsInfo (),
      new_interpreter->getInputTensorsInfo ());
  out_matched = gst_tensors_info_is_equal (old_interpreter->getOutputTensorsInfo (),
      new_interpreter->getOutputTensorsInfo ());

  if (!in_matched || !out_matched) {
    ml_loge ("The model has unmatched tensors info\n");
    ret = -EINVAL;
  } else {
    interpreter = new_interpreter;
  }

  new_interpreter->unlock ();
  old_interpreter->unlock ();

  return ret;
}

/**
 * @brief callback method to replace interpreter for shared model
 */
void replace_interpreter (void * instance, void * interperter) {
  TFLiteCore * core = reinterpret_cast <TFLiteCore *> (instance);
  TFLiteInterpreter * interpreter_new = reinterpret_cast <TFLiteInterpreter *> (interperter);
  if (core->reloadInterpreter (interpreter_new) != 0)
    nns_loge ("Failed to replace interpreter");
}

/**
 * @brief	reload a model
 * @param	tflite	: the class object
 * @param[in] model_path : the path of model file
 * @return 0 if OK. non-zero if error.
 * @note reloadModel() is asynchronously called with other callbacks. But, it requires
 *       extra memory size enough to temporarily hold both models during this function.
 */
int
TFLiteCore::reloadModel (const char *_model_path)
{
  TFLiteInterpreter * interpreter_temp = interpreter;
  const char *_ext_delegate_path;
  GHashTable *_ext_delegate_kv;

  if (!g_file_test (_model_path, G_FILE_TEST_IS_REGULAR)) {
    ml_loge ("The path of model file(s), %s, to reload is invalid.", _model_path);
    return -EINVAL;
  }
  interpreter_sub = new TFLiteInterpreter ();
  interpreter_sub->setModelPath (_model_path);
  interpreter->getExtDelegate(&_ext_delegate_path, &_ext_delegate_kv);
  interpreter_sub->setExtDelegate(_ext_delegate_path, _ext_delegate_kv);

  /**
   * load a model into sub interpreter. This loading overhead is independent
   * with main one's activities.
   */
  if (interpreter_sub->loadModel (num_threads, delegate) != 0) {
    ml_loge ("Failed to load model %s\n", _model_path);
    return -EINVAL;
  }
  if (interpreter_sub->setInputTensorProp () != 0) {
    ml_loge ("Failed to initialize input tensor\n");
    return -EINVAL;
  }
  if (interpreter_sub->setOutputTensorProp () != 0) {
    ml_loge ("Failed to initialize output tensor\n");
    return -EINVAL;
  }
  if (interpreter_sub->cacheInOutTensorPtr () != 0) {
    ml_loge ("Failed to cache input and output tensors storage\n");
    return -EINVAL;
  }

  if (shared_tensor_filter_key) {
    /* update cores with new interpreter that has shared key */
    nnstreamer_filter_shared_model_replace (this, shared_tensor_filter_key,
        interpreter_sub, replace_interpreter, free_interpreter);
  }
  else {
    if (reloadInterpreter (interpreter_sub) != 0) {
      ml_loge ("Failed replace interpreter\n");
      return -EINVAL;
    }
    delete interpreter_temp;
  }

  return 0;
}

/**
 * @brief	run the model with the input.
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::invoke (const GstTensorMemory *input, GstTensorMemory *output)
{
  int err;

  interpreter->lock ();
  err = interpreter->invoke (input, output);
  interpreter->unlock ();

  return err;
}

/**
 * @brief cache input and output tensor ptr before invoke
 */
int
TFLiteCore::cacheInOutTensorPtr ()
{
  int err;

  interpreter->lock ();
  err = interpreter->cacheInOutTensorPtr ();
  interpreter->unlock ();

  return err;
}

/**
 * @brief Internal function to get the option for tf-lite model.
 */
static int
tflite_parseCustomOption (const GstTensorFilterProperties *prop, tflite_option_s *option)
{
  if (prop->num_models != 1 || prop->model_files[0] == NULL)
    return -1;

  option->model_file = prop->model_files[0];
  option->accelerators = prop->accl_str;
  option->delegate = TFLITE_DELEGATE_NONE;
  option->num_threads = -1;
  option->ext_delegate_path = nullptr;
  option->ext_delegate_kv_table = nullptr;

  if (prop->custom_properties) {
    gchar **strv;
    guint i, len;

    strv = g_strsplit (prop->custom_properties, ",", -1);
    len = g_strv_length (strv);

    for (i = 0; i < len; ++i) {
      gchar **pair = g_strsplit (strv[i], ":", -1);

      if (g_strv_length (pair) > 1) {
        g_strstrip (pair[0]);
        g_strstrip (pair[1]);

        if (g_ascii_strcasecmp (pair[0], "NumThreads") == 0) {
          option->num_threads = (int)g_ascii_strtoll (pair[1], NULL, 10);
        } else if (g_ascii_strcasecmp (pair[0], "Delegate") == 0) {
          if (g_ascii_strcasecmp (pair[1], "NNAPI") == 0)
            option->delegate = TFLITE_DELEGATE_NNAPI;
          else if (g_ascii_strcasecmp (pair[1], "GPU") == 0)
            option->delegate = TFLITE_DELEGATE_GPU;
          else if (g_ascii_strcasecmp (pair[1], "XNNPACK") == 0)
            option->delegate = TFLITE_DELEGATE_XNNPACK;
          else if (g_ascii_strcasecmp (pair[1], "External") == 0)
            option->delegate = TFLITE_DELEGATE_EXTERNAL;
          else
            ml_logw ("Unknown option to set tensorflow-lite delegate (%s).", pair[1]);
        } else if (g_ascii_strcasecmp (pair[0], "ExtDelegateLib") == 0) {
          option->ext_delegate_path = g_strdup (pair[1]);
        } else if (g_ascii_strcasecmp (pair[0], "ExtDelegateKeyVal") == 0) {
          gchar **kvpairs;
          guint j, kvnum;
          GHashTable *table = option->ext_delegate_kv_table;

          kvpairs = g_strsplit (pair[1], ";", -1);
          kvnum = g_strv_length (kvpairs);

          for (j = 0; j < kvnum; j++) {
            gchar **kv = g_strsplit (kvpairs[j], "#", -1);

            if (g_strv_length (kv) > 1) {
              g_strstrip (kv[0]);
              g_strstrip (kv[1]);
              if (!table) {
                table = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, g_free);
                option->ext_delegate_kv_table = table;
              }
              g_hash_table_insert (table, g_strdup (kv[0]), g_strdup (kv[1]));
            }
            g_strfreev (kv);
          }
          g_strfreev (kvpairs);
        } else {
          ml_logw ("Unknown option (%s).", strv[i]);
        }
      }

      g_strfreev (pair);
    }

    g_strfreev (strv);
  }

  if (option->delegate == TFLITE_DELEGATE_EXTERNAL
      && option->ext_delegate_path == NULL) {
    ml_logw ("No shared lib for external delegate.");
    option->delegate = TFLITE_DELEGATE_NONE;
  }

  return 0;
}

/**
 * @brief Free privateData and move on.
 */
static void
tflite_close (const GstTensorFilterProperties *prop, void **private_data)
{
  TFLiteCore *core = static_cast<TFLiteCore *> (*private_data);
  UNUSED (prop);

  if (!core)
    return;

  delete core;
  *private_data = NULL;
}

/**
 * @brief Load tensorflow lite modelfile
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @return 0 if successfully loaded. 1 if skipped (already loaded).
 *        -1 if the object construction is failed.
 *        -2 if the object initialization if failed
 */
static int
tflite_loadModelFile (const GstTensorFilterProperties *prop, void **private_data)
{
  int ret = 0;
  TFLiteCore *core;
  tflite_option_s option = {};

  if (tflite_parseCustomOption (prop, &option) != 0) {
    g_printerr ("Failed to parse options to initialize tensorflow-lite model.");
    ret = -1;
    goto done;
  }

  core = static_cast<TFLiteCore *> (*private_data);

  if (core != NULL) {
    if (core->compareModelPath (option.model_file)) {
      ret = 1; /* skipped */
      goto done;
    }

    tflite_close (prop, private_data);
  }

  core = new TFLiteCore (prop);
  if (core == NULL) {
    g_printerr ("Failed to allocate memory for filter subplugin.");
    ret = -1;
    goto done;
  }

  if (core->init (&option) != 0) {
    *private_data = NULL;
    delete core;

    g_printerr ("failed to initialize the object: Tensorflow-lite");
    ret = -2;
    goto done;
  }

  *private_data = core;

done:
  g_free ((gpointer) option.ext_delegate_path);
  option.ext_delegate_path = nullptr;

  if (option.ext_delegate_kv_table)
    g_hash_table_unref (option.ext_delegate_kv_table);
  option.ext_delegate_kv_table = nullptr;

  return ret;
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 */
static int
tflite_open (const GstTensorFilterProperties *prop, void **private_data)
{
  int status = tflite_loadModelFile (prop, private_data);

  return status;
}

/**
 * @brief The mandatory callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @param[in] input The array of input tensors
 * @param[out] output The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
static int
tflite_invoke (const GstTensorFilterProperties *prop, void **private_data,
    const GstTensorMemory *input, GstTensorMemory *output)
{
  TFLiteCore *core = static_cast<TFLiteCore *> (*private_data);
  g_return_val_if_fail (core && input && output, -EINVAL);
  UNUSED (prop);

  return core->invoke (input, output);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @param[out] info The dimesions and types of input tensors
 */
static int
tflite_getInputDim (const GstTensorFilterProperties *prop, void **private_data,
    GstTensorsInfo *info)
{
  TFLiteCore *core = static_cast<TFLiteCore *> (*private_data);
  g_return_val_if_fail (core && info, -EINVAL);
  UNUSED (prop);

  return core->getInputTensorDim (info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @param[out] info The dimesions and types of output tensors
 */
static int
tflite_getOutputDim (const GstTensorFilterProperties *prop, void **private_data,
    GstTensorsInfo *info)
{
  TFLiteCore *core = static_cast<TFLiteCore *> (*private_data);
  g_return_val_if_fail (core && info, -EINVAL);
  UNUSED (prop);

  return core->getOutputTensorDim (info);
}

#define tryRecovery(failedAt, status, location, exp) \
  do {                                               \
    status = (exp);                                  \
    if (status != 0) {                               \
      failedAt = location;                           \
      goto recovery_fail;                            \
    }                                                \
  } while (0)

/**
 * @brief A fallback function to recover input tensor dimensions
 */
static void
tflite_setInputDim_recovery (
    TFLiteCore *core, GstTensorsInfo *cur_in_info, const char *reason, int mode)
{
  int failedAt, status;

  tryRecovery (failedAt, status, __LINE__, core->setInputTensorDim (cur_in_info));
  if (mode >= 1)
    tryRecovery (failedAt, status, __LINE__, core->setInputTensorProp ());
  if (mode >= 2)
    tryRecovery (failedAt, status, __LINE__, core->setOutputTensorProp ());
  tryRecovery (failedAt, status, __LINE__, core->cacheInOutTensorPtr ());

  return;

recovery_fail:
  ml_logf ("Tensorflow-lite's setInputDim failed (%s) and its recovery failed (at %d line with error %d), too. "
           "The behavior will be unstable.\n",
      reason, failedAt, status);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @param in_info The dimesions and types of input tensors
 * @param[out] out_info The dimesions and types of output tensors
 * @detail Output Tensor info is recalculated based on the set Input Tensor Info
 */
static int
tflite_setInputDim (const GstTensorFilterProperties *prop, void **private_data,
    const GstTensorsInfo *in_info, GstTensorsInfo *out_info)
{
  TFLiteCore *core = static_cast<TFLiteCore *> (*private_data);
  GstTensorsInfo cur_in_info;
  int status;
  UNUSED (prop);

  g_return_val_if_fail (core, -EINVAL);
  g_return_val_if_fail (in_info, -EINVAL);
  g_return_val_if_fail (out_info, -EINVAL);

  /** get current input tensor info for resetting */
  status = core->getInputTensorDim (&cur_in_info);
  if (status != 0)
    return status;

  /** set new input tensor info */
  status = core->setInputTensorDim (in_info);
  if (status != 0) {
    tflite_setInputDim_recovery (core, &cur_in_info, "while setting input tensor info", 0);
    return status;
  }

  /** update input tensor info */
  if ((status = core->setInputTensorProp ()) != 0) {
    tflite_setInputDim_recovery (core, &cur_in_info, "while updating input tensor info", 1);
    return status;
  }

  /** update output tensor info */
  if ((status = core->setOutputTensorProp ()) != 0) {
    tflite_setInputDim_recovery (core, &cur_in_info, "while updating output tensor info", 2);
    return status;
  }

  /** update the input and output tensor cache */
  status = core->cacheInOutTensorPtr ();
  if (status != 0) {
    tflite_setInputDim_recovery (
        core, &cur_in_info, "while updating input and output tensor cache", 2);
    return status;
  }

  /** get output tensor info to be returned */
  status = core->getOutputTensorDim (out_info);
  if (status != 0) {
    tflite_setInputDim_recovery (
        core, &cur_in_info, "while retreiving update output tensor info", 2);
    return status;
  }

  return 0;
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @return 0 if OK. non-zero if error.
 */
static int
tflite_reloadModel (const GstTensorFilterProperties *prop, void **private_data)
{
  TFLiteCore *core = static_cast<TFLiteCore *> (*private_data);
  g_return_val_if_fail (core, -EINVAL);

  if (prop->num_models != 1)
    return -1;

  return core->reloadModel (prop->model_files[0]);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param[in] hw backend accelerator hardware
 * @return 0 if supported. -errno if not supported.
 */
static int
tflite_checkAvailability (accl_hw hw)
{
  if (g_strv_contains (tflite_accl_support, get_accl_hw_str (hw)))
    return 0;

  return -ENOENT;
}

static gchar filter_subplugin_tensorflow_lite[] = TFLITE_SUBPLUGIN_NAME;

static GstTensorFilterFramework NNS_support_tensorflow_lite
    = { .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
        .open = tflite_open,
        .close = tflite_close,
        { .v0 = {
              .name = filter_subplugin_tensorflow_lite,
              .allow_in_place = FALSE, /** @todo: support this to optimize performance later. */
              .allocate_in_invoke = FALSE,
              .run_without_model = FALSE,
              .verify_model_path = TRUE,
              .statistics = &tflite_internal_stats,
              .invoke_NN = tflite_invoke,
              .getInputDimension = tflite_getInputDim,
              .getOutputDimension = tflite_getOutputDim,
              .setInputDimension = tflite_setInputDim,
              .destroyNotify = nullptr,
              .reloadModel = tflite_reloadModel,
              .handleEvent = nullptr,
              .checkAvailability = tflite_checkAvailability,
              .allocateInInvoke = nullptr,
          } } };

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_tflite (void)
{
  nnstreamer_filter_probe (&NNS_support_tensorflow_lite);
  nnstreamer_filter_set_custom_property_desc (
      NNS_support_tensorflow_lite.v0.name,
      "NumThreads", "Number of threads. Set 0 for default behaviors.",
      "Delegate", "TF-Lite delegation options: {'NNAPI', 'GPU', 'XNNPACK', 'External'}."
      " Do not specify to disable delegation.",
      "ExtDelegateLib", "Path to external delegate shared library",
      "ExtDelegateKeyVal", "key/values pairs optional parameters for delegate."
      " Format ExtDelegateKeyVal=key1#value1;key2#value2...",
      NULL);
}

/** @brief Destruct the subplugin */
void
fini_filter_tflite (void)
{
  nnstreamer_filter_exit (NNS_support_tensorflow_lite.v0.name);
}

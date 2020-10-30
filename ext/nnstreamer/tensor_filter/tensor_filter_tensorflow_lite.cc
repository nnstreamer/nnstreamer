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

#include <unistd.h>
#include <limits.h>
#include <algorithm>
#include <thread>

#include <nnstreamer_log.h>
#include <nnstreamer_plugin_api.h>
#define NO_ANONYMOUS_NESTED_STRUCT
#include <nnstreamer_plugin_api_filter.h>
#undef NO_ANONYMOUS_NESTED_STRUCT
#include <nnstreamer_conf.h>

#if TFLITE_VERSION_MAJOR >= 2 || TFLITE_VERSION_MINOR >= 13
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#else
#include <tensorflow/contrib/lite/model.h>
#include <tensorflow/contrib/lite/kernels/register.h>
#endif

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

/**
 * @brief Option to open tf-lite model.
 */
typedef struct
{
  const gchar *model_file; /**< path to tensorflow-lite model file */
  const gchar *accelerators; /**< accelerators set for this subplugin */
  gint num_threads; /**< the number of threads */
} tflite_option_s;

/**
 * @brief Possible accelerators.
 */
static const gchar *tflite_accl_support[] = {
  ACCL_CPU_NEON_STR,
  ACCL_CPU_SIMD_STR,
  ACCL_CPU_STR,
  ACCL_GPU_STR,
  ACCL_NPU_STR,
  NULL
};

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

  int invoke (const GstTensorMemory * input, GstTensorMemory * output);
  int loadModel (int num_threads, accl_hw accelerator);
  void moveInternals (TFLiteInterpreter& interp);

  int setInputTensorProp ();
  int setOutputTensorProp ();
  int setInputTensorsInfo (const GstTensorsInfo * info);

  void setModelPath (const char *model_path);
  /** @brief get current model path */
  const char *getModelPath () { return model_path; }

  /** @brief return input tensor meta */
  const GstTensorsInfo* getInputTensorsInfo () { return &inputTensorMeta; }
  /** @brief return output tensor meta */
  const GstTensorsInfo* getOutputTensorsInfo () { return &outputTensorMeta; }

  /** @brief lock this interpreter */
  void lock () { g_mutex_lock (&mutex); }
  /** @brief unlock this interpreter */
  void unlock () { g_mutex_unlock (&mutex); }
  /** @brief cache input and output tensor ptr before invoke */
  int cacheInOutTensorPtr ();

  /** @brief set delegate for the tflite interpreter */
  void setDelegate (TfLiteDelegate* delegate) { delegate_ = delegate; }

private:
  GMutex mutex;
  char *model_path;
  bool is_cached_after_first_invoke;  /**< To cache again after first invoke */

  std::unique_ptr <tflite::Interpreter> interpreter;
  std::unique_ptr <tflite::FlatBufferModel> model;

  GstTensorsInfo inputTensorMeta;  /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta;  /**< The tensor info of output tensors */
  std::vector<TfLiteTensor *> inputTensorPtr;
  std::vector<TfLiteTensor *> outputTensorPtr;

  tensor_type getTensorType (TfLiteType tfType);
  int getTensorDim (int tensor_idx, tensor_dim dim);
  int setTensorProp (const std::vector<int> &tensor_idx_list,
      GstTensorsInfo * tensorMeta);

  TfLiteDelegate* delegate_ = nullptr; /**< The delegate for tflite interpreter */

#if defined(__ANDROID__) && (TFLITE_VERSION_MAJOR >= 2 || TFLITE_VERSION_MINOR >= 14)
  std::unique_ptr<tflite::StatefulNnApiDelegate> stateful_nnapi_delegate; /**< The pointer of NNAPI delegate */
#endif
};

/**
 * @brief	ring cache structure
 */
class TFLiteCore
{
public:
  int init (tflite_option_s * option);
  int loadModel ();
  gboolean compareModelPath (const char *model_path);
  int setInputTensorProp ();
  int setOutputTensorProp ();
  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int setInputTensorDim (const GstTensorsInfo * info);
  int reloadModel (const char * model_path);
  int invoke (const GstTensorMemory * input, GstTensorMemory * output);
  /** @brief cache input and output tensor ptr before invoke */
  int cacheInOutTensorPtr ();

private:
  int num_threads;
  accl_hw accelerator;

  TFLiteInterpreter interpreter;
  TFLiteInterpreter interpreter_sub;

  void setAccelerator (const char * accelerators);
};

extern "C" { /* accessed by android api */
  void init_filter_tflite (void) __attribute__ ((constructor));
  void fini_filter_tflite (void) __attribute__ ((destructor));
}

/**
 * @brief TFLiteInterpreter constructor
 */
TFLiteInterpreter::TFLiteInterpreter ()
{
  interpreter = nullptr;
  model = nullptr;
  model_path = nullptr;

  g_mutex_init (&mutex);

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);

  is_cached_after_first_invoke = false;
}

/**
 * @brief TFLiteInterpreter desctructor
 */
TFLiteInterpreter::~TFLiteInterpreter ()
{
  g_mutex_clear (&mutex);
  g_free (model_path);

  gst_tensors_info_free (&inputTensorMeta);
  gst_tensors_info_free (&outputTensorMeta);
}

/**
 * @brief Internal implementation of TFLiteCore's invoke()
 */
int
TFLiteInterpreter::invoke (const GstTensorMemory * input,
    GstTensorMemory * output)
{
  int64_t start_time, stop_time;
  TfLiteTensor *tensor_ptr;
  TfLiteStatus status;

  start_time = g_get_monotonic_time ();
  for (unsigned int i = 0; i < outputTensorMeta.num_tensors; ++i) {
    tensor_ptr = outputTensorPtr[i];
    tensor_ptr->data.raw = (char *) output[i].data;
  }

  for (unsigned int i = 0; i < inputTensorMeta.num_tensors; ++i) {
    tensor_ptr = inputTensorPtr[i];
    tensor_ptr->data.raw = (char *) input[i].data;
  }
  stop_time = g_get_monotonic_time ();

  tflite_internal_stats.total_overhead_latency += stop_time - start_time;

  start_time = g_get_monotonic_time ();
  status = interpreter->Invoke ();
  stop_time = g_get_monotonic_time ();

  tflite_internal_stats.total_invoke_latency += stop_time - start_time;
  tflite_internal_stats.total_invoke_num += 1;

#if (DBG)
  g_critical ("Invoke() is finished: %" G_GINT64_FORMAT,
      (stop_time - start_time));
  g_critical ("%ld invoke average %" G_GINT64_FORMAT ", total overhead %" G_GINT64_FORMAT,
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
TFLiteInterpreter::loadModel (int num_threads, accl_hw accelerator)
{
#if (DBG)
  gint64 start_time = g_get_monotonic_time ();
#endif

  model = tflite::FlatBufferModel::BuildFromFile (model_path);
  if (!model) {
    ml_loge ("Failed to mmap model\n");
    return -1;
  }
  /* If got any trouble at model, active below code. It'll be help to analyze. */
  /* model->error_reporter (); */

  interpreter = nullptr;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder (*model, resolver) (&interpreter);
  if (!interpreter) {
    ml_loge ("Failed to construct interpreter\n");
    return -2;
  }

  if (num_threads > 0) {
    int n = static_cast<int> (std::thread::hardware_concurrency ());

    n = MIN (n, num_threads);
    ml_logi ("Set the number of threads (%d)", n);
    interpreter->SetNumThreads (n);
  }

  /** set nnapi delegate when accelerator set to auto (cpu.neon in Android), GPU or NPU */
  if (accelerator == ACCL_CPU_NEON || accelerator == ACCL_GPU ||
          accelerator == ACCL_NPU) {
#if defined(__ANDROID__) && (TFLITE_VERSION_MAJOR >= 2 || TFLITE_VERSION_MINOR >= 14)
    stateful_nnapi_delegate.reset (new tflite::StatefulNnApiDelegate ());
    setDelegate (stateful_nnapi_delegate.get ());
#else
    ml_logw ("NNAPI delegate is available only in Android with tflite v1.14.0 or higher");
#endif
  }

  if (delegate_ != nullptr) {
    if (interpreter->ModifyGraphWithDelegate (delegate_) != kTfLiteOk) {
      ml_loge ("Failed to allocate tensors with NNAPI delegate\n");
      return -2;
    }
  } else {
    if (interpreter->AllocateTensors () != kTfLiteOk) {
      ml_loge ("Failed to allocate tensors\n");
      return -2;
    }
  }

#if (DBG)
  gint64 stop_time = g_get_monotonic_time ();
  g_message ("Model is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
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
TFLiteInterpreter::setTensorProp (const std::vector<int> &tensor_idx_list,
    GstTensorsInfo * tensorMeta)
{
  tensorMeta->num_tensors = tensor_idx_list.size ();

  for (unsigned int i = 0; i < tensorMeta->num_tensors; ++i) {
    if (getTensorDim (tensor_idx_list[i], tensorMeta->info[i].dimension)) {
      ml_loge ("failed to get the dimension of input tensors");
      return -1;
    }
    tensorMeta->info[i].type =
        getTensorType (interpreter->tensor (tensor_idx_list[i])->type);

#if (DBG)
    gchar *dim_str =
        gst_tensor_get_dimension_string (tensorMeta->info[i].dimension);
    g_message ("tensorMeta[%d] >> type:%d, dim[%s]",
        i, tensorMeta->info[i].type, dim_str);
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
TFLiteInterpreter::setInputTensorsInfo (const GstTensorsInfo * info)
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
    tf_type = getTensorType (
        interpreter->tensor (input_idx_list[tensor_idx])->type);
    if (tf_type != tensor_info->type)
      return -EINVAL;

    /**
     * Given that the rank intended by the user cannot be exactly determined,
     * iterate over all possible ranks starting from MAX rank to the actual rank
     * of the dimension array. In case of none of these ranks work, return error
     */
    input_rank = gst_tensor_info_get_rank (&info->info[tensor_idx]);
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
 * @brief Move the ownership of interpreter internal members
 */
void
TFLiteInterpreter::moveInternals (TFLiteInterpreter& interp)
{
  interpreter = std::move (interp.interpreter);
  model = std::move (interp.model);
  inputTensorPtr = std::move (interp.inputTensorPtr);
  outputTensorPtr = std::move (interp.outputTensorPtr);
  setModelPath (interp.getModelPath ());
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
 * @brief	Set the accelerator for the tf engine
 */
void
TFLiteCore::setAccelerator (const char * accelerators)
{
  accelerator = parse_accl_hw (accelerators, tflite_accl_support,
      tflite_accl_auto, tflite_accl_default);

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
TFLiteCore::init (tflite_option_s * option)
{
  interpreter.setModelPath (option->model_file);
  num_threads = option->num_threads;

  setAccelerator (option->accelerators);
  g_message ("accl = %s", get_accl_hw_str (accelerator));

  if (loadModel ()) {
    ml_loge ("Failed to load model\n");
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

  interpreter.lock ();
  is_same = (g_strcmp0 (model_path, interpreter.getModelPath ()) == 0);
  interpreter.unlock ();

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

  interpreter.lock ();
  err = interpreter.loadModel (num_threads, accelerator);
  interpreter.unlock ();

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

  interpreter.lock ();
  err = interpreter.setInputTensorProp ();
  interpreter.unlock ();

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

  interpreter.lock ();
  err = interpreter.setOutputTensorProp ();
  interpreter.unlock ();

  return err;
}

/**
 * @brief	return the Dimension of Input Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getInputTensorDim (GstTensorsInfo * info)
{
  interpreter.lock ();
  gst_tensors_info_copy (info, interpreter.getInputTensorsInfo ());
  interpreter.unlock ();

  return 0;
}

/**
 * @brief	return the Dimension of Tensor.
 * @param[out] info Structure for tensor info.
 * @todo return whole array rather than index 0
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::getOutputTensorDim (GstTensorsInfo * info)
{
  interpreter.lock ();
  gst_tensors_info_copy (info, interpreter.getOutputTensorsInfo ());
  interpreter.unlock ();

  return 0;
}

/**
 * @brief set the Dimension for Input Tensor.
 * @param info Structure for input tensor info.
 * @return 0 if OK. non-zero if error.
 * @note rank can be changed dependent on the model
 */
int
TFLiteCore::setInputTensorDim (const GstTensorsInfo * info)
{
  int err;

  interpreter.lock ();
  err = interpreter.setInputTensorsInfo (info);
  interpreter.unlock ();

  return err;
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
TFLiteCore::reloadModel (const char * _model_path)
{
  int err;

  if (!g_file_test (_model_path, G_FILE_TEST_IS_REGULAR)) {
    ml_loge ("The path of model file(s), %s, to reload is invalid.",
        _model_path);
    return -EINVAL;
  }

  interpreter_sub.lock ();
  interpreter_sub.setModelPath (_model_path);

  /**
   * load a model into sub interpreter. This loading overhead is indenendent
   * with main one's activities.
   */
  err = interpreter_sub.loadModel (num_threads, accelerator);
  if (err != 0) {
    ml_loge ("Failed to load model %s\n", _model_path);
    goto out_unlock;
  }
  err = interpreter_sub.setInputTensorProp ();
  if (err != 0) {
    ml_loge ("Failed to initialize input tensor\n");
    goto out_unlock;
  }
  err = interpreter_sub.setOutputTensorProp ();
  if (err != 0) {
    ml_loge ("Failed to initialize output tensor\n");
    goto out_unlock;
  }
  err = interpreter_sub.cacheInOutTensorPtr ();
  if (err != 0) {
    ml_loge ("Failed to cache input and output tensors storage\n");
    goto out_unlock;
  }

  /* Also, we need to check input/output tensors have the same info */
  if (!gst_tensors_info_is_equal (
        interpreter.getInputTensorsInfo (),
        interpreter_sub.getInputTensorsInfo ()) ||
      !gst_tensors_info_is_equal (
        interpreter.getOutputTensorsInfo (),
        interpreter_sub.getOutputTensorsInfo ())) {
    ml_loge ("The model has unmatched tensors info\n");
    err = -EINVAL;
    goto out_unlock;
  }

  /**
   * Everything is ready. let's move the model in sub interpreter to main one.
   * But, it needs to wait if main interpreter is busy (e.g., invoke()).
   */
  interpreter.lock ();
  interpreter.moveInternals (interpreter_sub);
  /* after this, all callbacks will handle operations for the reloaded model */
  interpreter.unlock ();

out_unlock:
  interpreter_sub.unlock ();

  return err;
}

/**
 * @brief	run the model with the input.
 * @param[in] input : The array of input tensors
 * @param[out]  output : The array of output tensors
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteCore::invoke (const GstTensorMemory * input, GstTensorMemory * output)
{
  int err;

  interpreter.lock ();
  err = interpreter.invoke (input, output);
  interpreter.unlock ();

  return err;
}

/**
 * @brief cache input and output tensor ptr before invoke
 */
int
TFLiteCore::cacheInOutTensorPtr ()
{
  int err;

  interpreter.lock ();
  err = interpreter.cacheInOutTensorPtr ();
  interpreter.unlock ();

  return err;
}

/**
 * @brief Internal function to get the option for tf-lite model.
 */
static int
tflite_parseCustomOption (const GstTensorFilterProperties * prop,
    tflite_option_s * option)
{
  if (prop->num_models != 1 || prop->model_files[0] == NULL)
    return -1;

  option->model_file = prop->model_files[0];
  option->accelerators = prop->accl_str;
  option->num_threads = -1;

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
          option->num_threads = (int) g_ascii_strtoll (pair[1], NULL, 10);
        } else {
          g_warning ("Unknown option (%s).", strv[i]);
        }
      }

      g_strfreev (pair);
    }

    g_strfreev (strv);
  }

  return 0;
}

/**
 * @brief Free privateData and move on.
 */
static void
tflite_close (const GstTensorFilterProperties * prop, void **private_data)
{
  TFLiteCore *core = static_cast<TFLiteCore *>(*private_data);

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
tflite_loadModelFile (const GstTensorFilterProperties * prop,
    void **private_data)
{
  TFLiteCore *core;
  tflite_option_s option = { 0, };

  if (tflite_parseCustomOption (prop, &option) != 0) {
    g_printerr ("Failed to parse options to initialize tensorflow-lite model.");
    return -1;
  }

  core = static_cast<TFLiteCore *>(*private_data);

  if (core != NULL) {
    if (core->compareModelPath (option.model_file))
      return 1; /* skipped */

    tflite_close (prop, private_data);
  }

  core = new TFLiteCore ();
  if (core == NULL) {
    g_printerr ("Failed to allocate memory for filter subplugin.");
    return -1;
  }

  if (core->init (&option) != 0) {
    *private_data = NULL;
    delete core;

    g_printerr ("failed to initialize the object: Tensorflow-lite");
    return -2;
  }

  *private_data = core;

  return 0;
}

/**
 * @brief The open callback for GstTensorFilterFramework. Called before anything else
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 */
static int
tflite_open (const GstTensorFilterProperties * prop, void **private_data)
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
tflite_invoke (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorMemory * input, GstTensorMemory * output)
{
  TFLiteCore *core = static_cast<TFLiteCore *>(*private_data);
  g_return_val_if_fail (core && input && output, -EINVAL);

  return core->invoke (input, output);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @param[out] info The dimesions and types of input tensors
 */
static int
tflite_getInputDim (const GstTensorFilterProperties * prop, void **private_data,
    GstTensorsInfo * info)
{
  TFLiteCore *core = static_cast<TFLiteCore *>(*private_data);
  g_return_val_if_fail (core && info, -EINVAL);

  return core->getInputTensorDim (info);
}

/**
 * @brief The optional callback for GstTensorFilterFramework
 * @param prop property of tensor_filter instance
 * @param private_data : tensorflow lite plugin's private data
 * @param[out] info The dimesions and types of output tensors
 */
static int
tflite_getOutputDim (const GstTensorFilterProperties * prop,
    void **private_data, GstTensorsInfo * info)
{
  TFLiteCore *core = static_cast<TFLiteCore *>(*private_data);
  g_return_val_if_fail (core && info, -EINVAL);

  return core->getOutputTensorDim (info);
}

#define tryRecovery(failedAt, status, location, exp) \
  do { \
    status = (exp); \
    if (status != 0) { \
      failedAt = location; \
      goto recovery_fail; \
    } \
  } while (0)

/**
 * @brief A fallback function to recover input tensor dimensions
 */
static void
tflite_setInputDim_recovery (TFLiteCore *core, GstTensorsInfo *cur_in_info,
    const char * reason, int mode)
{
  int failedAt, status;

  tryRecovery (failedAt, status, __LINE__,
      core->setInputTensorDim (cur_in_info));
  if (mode >= 1)
    tryRecovery (failedAt, status, __LINE__,
        core->setInputTensorProp ());
  if (mode >= 2)
    tryRecovery (failedAt, status, __LINE__,
        core->setOutputTensorProp ());
  tryRecovery (failedAt, status, __LINE__,
      core->cacheInOutTensorPtr ());

  return;

recovery_fail:
  ml_logf
      ("Tensorflow-lite's setInputDim failed (%s) and its recovery failed (at %d line with error %d), too. "
       "The behavior will be unstable.\n", reason, failedAt, status);
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
tflite_setInputDim (const GstTensorFilterProperties * prop, void **private_data,
    const GstTensorsInfo * in_info, GstTensorsInfo * out_info)
{
  TFLiteCore *core = static_cast<TFLiteCore *>(*private_data);
  GstTensorsInfo cur_in_info;
  int status;

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
    tflite_setInputDim_recovery (core, &cur_in_info,
        "while setting input tensor info", 0);
    return status;
  }

  /** update input tensor info */
  if ((status = core->setInputTensorProp ()) != 0) {
    tflite_setInputDim_recovery (core, &cur_in_info,
        "while updating input tensor info", 1);
    return status;
  }

  /** update output tensor info */
  if ((status = core->setOutputTensorProp ()) != 0) {
    tflite_setInputDim_recovery (core, &cur_in_info,
        "while updating output tensor info", 2);
    return status;
  }

  /** update the input and output tensor cache */
  status = core->cacheInOutTensorPtr ();
  if (status != 0) {
    tflite_setInputDim_recovery (core, &cur_in_info,
        "while updating input and output tensor cache", 2);
    return status;
  }

  /** get output tensor info to be returned */
  status = core->getOutputTensorDim (out_info);
  if (status != 0) {
    tflite_setInputDim_recovery (core, &cur_in_info,
        "while retreiving update output tensor info", 2);
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
tflite_reloadModel (const GstTensorFilterProperties * prop, void **private_data)
{
  TFLiteCore *core = static_cast<TFLiteCore *>(*private_data);
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

#if TFLITE_VERSION_MAJOR == 2 && !defined(__ANDROID__)
static gchar filter_subplugin_tensorflow_lite[] = "tensorflow2-lite";
#else
static gchar filter_subplugin_tensorflow_lite[] = "tensorflow-lite";
#endif

static GstTensorFilterFramework NNS_support_tensorflow_lite = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = tflite_open,
  .close = tflite_close,
  {
    .v0 = {
      .name = filter_subplugin_tensorflow_lite,
      .allow_in_place = FALSE,  /** @todo: support this to optimize performance later. */
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
      .checkAvailability = tflite_checkAvailability,
      .allocateInInvoke = nullptr,
    }
  }
};

/** @brief Initialize this object for tensor_filter subplugin runtime register */
void
init_filter_tflite (void)
{
  nnstreamer_filter_probe (&NNS_support_tensorflow_lite);
}

/** @brief Destruct the subplugin */
void
fini_filter_tflite (void)
{
  nnstreamer_filter_exit (NNS_support_tensorflow_lite.v0.name);
}

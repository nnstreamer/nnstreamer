/**
 * GStreamer Tensor_Filter, Tensorflow-Lite Module
 * Copyright (C) 2018 Samsung Electronics Co., Ltd. All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
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
 * @see    http://github.com/nnsuite/nnstreamer
 * @author HyoungJoo Ahn <hello.ahn@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (tensorflow-lite) for tensor_filter.
 */

#include <unistd.h>
#include <limits.h>
#include <algorithm>

#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>
#include <nnstreamer_conf.h>

#include <tensorflow/contrib/lite/model.h>
#include <tensorflow/contrib/lite/kernels/register.h>

#ifdef ENABLE_TFLITE_NNAPI_DELEGATE
#include "tflite/ext/nnapi_delegate.h"
#endif

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

/** Match all accelerators for nnapi at once */
#define REGEX_ACCL_NNAPI \
  "(^(true)[:]?([(]?(" \
  REGEX_ACCL_AUTO "|" \
  REGEX_ACCL_DEF "|" \
  REGEX_ACCL_CPU "|" \
  REGEX_ACCL_GPU "|" \
  REGEX_ACCL_NPU "|" \
  REGEX_ACCL_NEON ")*[)]?))"

/** Match accelerator for nnapi one by one */
#define REGEX_ACCL_NNAPI_ELEM \
  "(" \
  "(?<!!)" ACCL_AUTO_STR "|" \
  "(?<!!)" ACCL_DEF_STR "|" \
  "(?<!!)" ACCL_CPU_STR "|" \
  "(?<!!)" ACCL_GPU_STR "|" \
  "(?<!!)" ACCL_NPU_STR "|" \
  "(?<!!)" ACCL_NEON_STR ")?"

/**
 * @brief Wrapper class for TFLite Interpreter to support model switching
 */
class TFLiteInterpreter
{
public:
  TFLiteInterpreter ();
  ~TFLiteInterpreter ();

  int invoke (const GstTensorMemory * input, GstTensorMemory * output, bool use_nnapi);
  int loadModel (bool use_nnapi);
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

private:
  GMutex mutex;
  char *model_path;

  std::unique_ptr <tflite::Interpreter> interpreter;
  std::unique_ptr <tflite::FlatBufferModel> model;
#ifdef ENABLE_TFLITE_NNAPI_DELEGATE
  std::unique_ptr <nnfw::tflite::NNAPIDelegate> nnfw_delegate;
#endif

  GstTensorsInfo inputTensorMeta;  /**< The tensor info of input tensors */
  GstTensorsInfo outputTensorMeta;  /**< The tensor info of output tensors */

  tensor_type getTensorType (TfLiteType tfType);
  int getTensorDim (int tensor_idx, tensor_dim dim);
  int setTensorProp (const std::vector<int> &tensor_idx_list,
      GstTensorsInfo * tensorMeta);
};

/**
 * @brief	ring cache structure
 */
class TFLiteCore
{
public:
  TFLiteCore (const char *_model_path, const char *accelerators);

  int init ();
  int loadModel ();
  gboolean compareModelPath (const char *model_path);
  int setInputTensorProp ();
  int setOutputTensorProp ();
  int getInputTensorDim (GstTensorsInfo * info);
  int getOutputTensorDim (GstTensorsInfo * info);
  int setInputTensorDim (const GstTensorsInfo * info);
  int reloadModel (const char * model_path);
  int invoke (const GstTensorMemory * input, GstTensorMemory * output);

private:
  bool use_nnapi;
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
#ifdef ENABLE_TFLITE_NNAPI_DELEGATE
  nnfw_delegate = nullptr;
#endif
  model_path = nullptr;

  g_mutex_init (&mutex);

  gst_tensors_info_init (&inputTensorMeta);
  gst_tensors_info_init (&outputTensorMeta);
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
    GstTensorMemory * output, bool use_nnapi)
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  std::vector <int> tensors_idx;
  int tensor_idx;
  TfLiteTensor *tensor_ptr;
  TfLiteStatus status;

  for (unsigned int i = 0; i < outputTensorMeta.num_tensors; ++i) {
    tensor_idx = interpreter->outputs ()[i];
    tensor_ptr = interpreter->tensor (tensor_idx);

    g_assert (tensor_ptr->bytes == output[i].size);
    tensor_ptr->data.raw = (char *) output[i].data;
    tensors_idx.push_back (tensor_idx);
  }

  for (unsigned int i = 0; i < inputTensorMeta.num_tensors; ++i) {
    tensor_idx = interpreter->inputs ()[i];
    tensor_ptr = interpreter->tensor (tensor_idx);

    g_assert (tensor_ptr->bytes == input[i].size);
    tensor_ptr->data.raw = (char *) input[i].data;
    tensors_idx.push_back (tensor_idx);
  }

#ifdef ENABLE_TFLITE_NNAPI_DELEGATE
  if (use_nnapi)
    status = nnfw_delegate->Invoke (interpreter.get());
  else
#endif
    status = interpreter->Invoke ();

  /** if it is not `nullptr`, tensorflow makes `free()` the memory itself. */
  int tensorSize = tensors_idx.size ();
  for (int i = 0; i < tensorSize; ++i) {
    interpreter->tensor (tensors_idx[i])->data.raw = nullptr;
  }

#if (DBG)
  gint64 stop_time = g_get_real_time ();
  g_message ("Invoke() is finished: %" G_GINT64_FORMAT,
      (stop_time - start_time));
#endif

  if (status != kTfLiteOk) {
    g_critical ("Failed to invoke");
    return -1;
  }

  return 0;
}

/**
 * @brief Internal implementation of TFLiteCore's loadModel()
 * @return 0 if OK. non-zero if error.
 */
int
TFLiteInterpreter::loadModel (bool use_nnapi)
{
#if (DBG)
  gint64 start_time = g_get_real_time ();
#endif

  if (!g_file_test (model_path, G_FILE_TEST_IS_REGULAR)) {
    g_critical ("the file of model_path (%s) is not valid (not regular)\n", model_path);
    return -1;
  }
  model = tflite::FlatBufferModel::BuildFromFile (model_path);
  if (!model) {
    g_critical ("Failed to mmap model\n");
    return -1;
  }
  /* If got any trouble at model, active below code. It'll be help to analyze. */
  /* model->error_reporter (); */

  interpreter = nullptr;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder (*model, resolver) (&interpreter);
  if (!interpreter) {
    g_critical ("Failed to construct interpreter\n");
    return -2;
  }

  interpreter->UseNNAPI (use_nnapi);

#ifdef ENABLE_TFLITE_NNAPI_DELEGATE
  if (use_nnapi) {
    nnfw_delegate.reset (new ::nnfw::tflite::NNAPIDelegate);
    if (nnfw_delegate->BuildGraph (interpreter) != kTfLiteOk) {
      g_critical ("Fail to BuildGraph");
      return -3;
    }
  }
#endif

  /** set allocation type to dynamic for in/out tensors */
  int tensor_idx;

  int tensorSize = interpreter->inputs ().size ();
  for (int i = 0; i < tensorSize; ++i) {
    tensor_idx = interpreter->inputs ()[i];
    interpreter->tensor (tensor_idx)->allocation_type = kTfLiteDynamic;
  }

  tensorSize = interpreter->outputs ().size ();
  for (int i = 0; i < tensorSize; ++i) {
    tensor_idx = interpreter->outputs ()[i];
    interpreter->tensor (tensor_idx)->allocation_type = kTfLiteDynamic;
  }

  if (interpreter->AllocateTensors () != kTfLiteOk) {
    g_critical ("Failed to allocate tensors\n");
    return -2;
  }
#if (DBG)
  gint64 stop_time = g_get_real_time ();
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
      g_critical ("failed to get the dimension of input tensors");
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
  TfLiteStatus status;
  const std::vector<int> &input_idx_list = interpreter->inputs();

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
			std::vector<int> dims(rank);
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
      return -EINVAL;
  }

  status = interpreter->AllocateTensors ();
  if (status != kTfLiteOk)
    return -EINVAL;

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
#ifdef ENABLE_TFLITE_NNAPI_DELEGATE
  nnfw_delegate = std::move (interp.nnfw_delegate);
#endif
  setModelPath (interp.getModelPath ());
}

/**
 * @brief	TFLiteCore creator
 * @param	_model_path	: the logical path to '{model_name}.tflite' file
 * @param	accelerators  : the accelerators property set for this subplugin
 * @note	the model of _model_path will be loaded simultaneously
 * @return	Nothing
 */
TFLiteCore::TFLiteCore (const char * _model_path, const char * accelerators)
{
  g_assert (_model_path != NULL);

  interpreter.setModelPath (_model_path);

  setAccelerator (accelerators);

#if (DBG)
  g_message ("nnapi = %d, accl = %s", use_nnapi, get_accl_hw_str(accelerator));
#endif
}

/**
 * @brief	Set the accelerator for the tf engine
 */
void TFLiteCore::setAccelerator (const char * accelerators)
{
  use_nnapi = TRUE;
  accelerator = parse_accl_hw (accelerators, REGEX_ACCL_NNAPI,
      REGEX_ACCL_NNAPI_ELEM);
  if (accelerators == NULL || accelerator == ACCL_NONE)
    goto use_nnapi_ini;

  return;

use_nnapi_ini:
  use_nnapi = nnsconf_get_custom_value_bool ("tensorflowlite", "enable_nnapi",
      FALSE);
  if (use_nnapi == FALSE) {
    accelerator = ACCL_NONE;
  } else {
    accelerator = ACCL_AUTO;
  }
}

/**
 * @brief	initialize the object with tflite model
 * @return 0 if OK. non-zero if error.
 *        -1 if the model is not loaded.
 *        -2 if the initialization of input tensor is failed.
 *        -3 if the initialization of output tensor is failed.
 */
int
TFLiteCore::init ()
{
  if (loadModel ()) {
    g_critical ("Failed to load model\n");
    return -1;
  }
  if (setInputTensorProp ()) {
    g_critical ("Failed to initialize input tensor\n");
    return -2;
  }
  if (setOutputTensorProp ()) {
    g_critical ("Failed to initialize output tensor\n");
    return -3;
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
  err = interpreter.loadModel (use_nnapi);
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
  gst_tensors_info_copy (info, interpreter.getInputTensorsInfo());
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
  gst_tensors_info_copy (info, interpreter.getOutputTensorsInfo());
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

  interpreter_sub.lock ();
  interpreter_sub.setModelPath (_model_path);

  /**
   * load a model into sub interpreter. This loading overhead is indenendent
   * with main one's activities.
   */
  err = interpreter_sub.loadModel (use_nnapi);
  if (err != 0) {
    g_critical ("Failed to load model %s\n", _model_path);
    goto out_unlock;
  }
  err = interpreter_sub.setInputTensorProp ();
  if (err != 0) {
    g_critical ("Failed to initialize input tensor\n");
    goto out_unlock;
  }
  err = interpreter_sub.setOutputTensorProp ();
  if (err != 0) {
    g_critical ("Failed to initialize output tensor\n");
    goto out_unlock;
  }

  /* Also, we need to check input/output tensors have the same info */
  if (!gst_tensors_info_is_equal (
        interpreter.getInputTensorsInfo (),
        interpreter_sub.getInputTensorsInfo ()) ||
      !gst_tensors_info_is_equal (
        interpreter.getOutputTensorsInfo (),
        interpreter_sub.getOutputTensorsInfo ())) {
    g_critical ("The model has unmatched tensors info\n");
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
  err = interpreter.invoke (input, output, use_nnapi);
  interpreter.unlock ();

  return err;
}

/**
 * @brief Free privateData and move on.
 */
static void
tflite_close (const GstTensorFilterProperties * prop, void **private_data)
{
  TFLiteCore *core = static_cast<TFLiteCore *>(*private_data);

  g_assert (core);
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
  const gchar *model_file;

  if (prop->num_models != 1)
    return -1;

  core = static_cast<TFLiteCore *>(*private_data);
  model_file = prop->model_files[0];

  if (core != NULL) {
    if (core->compareModelPath (model_file))
      return 1; /* skipped */

    tflite_close (prop, private_data);
  }

  core = new TFLiteCore (model_file, prop->accl_str);
  if (core == NULL) {
    g_printerr ("Failed to allocate memory for filter subplugin.");
    return -1;
  }

  if (core->init () != 0) {
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

  g_assert (status >= 0);       /** This must be called only once */

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

  g_assert (core);

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

  g_assert (core);

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

  g_assert (core);

  return core->getOutputTensorDim (info);
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

  g_assert (core);

  /** get current input tensor info for resetting */
  status = core->getInputTensorDim (&cur_in_info);
  if (status != 0)
    return status;

  /** set new input tensor info */
  status = core->setInputTensorDim (in_info);
  if (status != 0) {
    g_assert (core->setInputTensorDim (&cur_in_info) == 0);
    return status;
  }

  /** update input tensor info */
  if ((status = core->setInputTensorProp ()) != 0) {
    g_assert (core->setInputTensorDim (&cur_in_info) == 0);
    g_assert (core->setInputTensorProp () == 0);
    return status;
  }

  /** update output tensor info */
  if ((status = core->setOutputTensorProp ()) != 0) {
    g_assert (core->setInputTensorDim (&cur_in_info) == 0);
    g_assert (core->setInputTensorProp () == 0);
    g_assert (core->setOutputTensorProp () == 0);
    return status;
  }

  /** get output tensor info to be returned */
  status = core->getOutputTensorDim (out_info);
  if (status != 0) {
    g_assert (core->setInputTensorDim (&cur_in_info) == 0);
    g_assert (core->setInputTensorProp () == 0);
    g_assert (core->setOutputTensorProp () == 0);
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

  g_assert (core);

  if (prop->num_models != 1)
    return -1;

  return core->reloadModel (prop->model_files[0]);
}

static gchar filter_subplugin_tensorflow_lite[] = "tensorflow-lite";

static GstTensorFilterFramework NNS_support_tensorflow_lite = {
  .name = filter_subplugin_tensorflow_lite,
  .allow_in_place = FALSE,      /** @todo: support this to optimize performance later. */
  .allocate_in_invoke = FALSE,
  .run_without_model = FALSE,
  .invoke_NN = tflite_invoke,
  .getInputDimension = tflite_getInputDim,
  .getOutputDimension = tflite_getOutputDim,
  .setInputDimension = tflite_setInputDim,
  .open = tflite_open,
  .close = tflite_close,
  .destroyNotify = NULL,
  .reloadModel = tflite_reloadModel,
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
  nnstreamer_filter_exit (NNS_support_tensorflow_lite.name);
}
